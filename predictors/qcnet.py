# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from itertools import chain
from itertools import compress
from pathlib import Path
from typing import Optional
import os
import pickle

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData

from losses import MixtureNLLLoss
from losses import NLLLoss
from metrics import Brier
from metrics import MR
from metrics import minADE
from metrics import minAHE
from metrics import minFDE
from metrics import minFHE
from metrics.utils import topk
from metrics.utils import valid_filter
from modules import QCNetDecoder
from modules import QCNetEncoder
import logging

try:
    from av2.datasets.motion_forecasting.eval.submission import ChallengeSubmission
except ImportError:
    ChallengeSubmission = object


class QCNet(pl.LightningModule):

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 output_head: bool,
                 num_historical_steps: int,
                 num_future_steps: int,
                 num_modes: int,
                 num_recurrent_steps: int,
                 num_freq_bands: int,
                 num_map_layers: int,
                 num_agent_layers: int,
                 num_dec_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 pl2pl_radius: float,
                 time_span: Optional[int],
                 pl2a_radius: float,
                 a2a_radius: float,
                 num_t2m_steps: Optional[int],
                 pl2m_radius: float,
                 a2m_radius: float,
                 lr: float,
                 weight_decay: float,
                 T_max: int,
                 submission_dir: str,
                 submission_file_name: str,
                 **kwargs) -> None:
        super(QCNet, self).__init__()
        self.save_hyperparameters()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_head = output_head
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_modes = num_modes
        self.num_recurrent_steps = num_recurrent_steps
        self.num_freq_bands = num_freq_bands
        self.num_map_layers = num_map_layers
        self.num_agent_layers = num_agent_layers
        self.num_dec_layers = num_dec_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.pl2pl_radius = pl2pl_radius
        self.time_span = time_span
        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.num_t2m_steps = num_t2m_steps
        self.pl2m_radius = pl2m_radius
        self.a2m_radius = a2m_radius
        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max
        self.submission_dir = submission_dir
        self.submission_file_name = submission_file_name

        self.encoder = QCNetEncoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            pl2pl_radius=pl2pl_radius,
            time_span=time_span,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_map_layers=num_map_layers,
            num_agent_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        self.decoder = QCNetDecoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            output_head=output_head,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            num_modes=num_modes,
            num_recurrent_steps=num_recurrent_steps,
            num_t2m_steps=num_t2m_steps,
            pl2m_radius=pl2m_radius,
            a2m_radius=a2m_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_dec_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )

        self.reg_loss = NLLLoss(component_distribution=['laplace'] * output_dim + ['von_mises'] * output_head,
                                reduction='none')
        self.cls_loss = MixtureNLLLoss(component_distribution=['laplace'] * output_dim + ['von_mises'] * output_head,
                                       reduction='none')
        self.cls_loss_region = MixtureNLLLoss(component_distribution=['laplace'] * output_dim + ['von_mises'] * output_head,
                                       reduction='none') # 这个loss不应该是Mixture

        self.Brier = Brier(max_guesses=6)
        self.minADE = minADE(max_guesses=6)
        self.minAHE = minAHE(max_guesses=6)
        self.minFDE = minFDE(max_guesses=6)
        self.minFHE = minFHE(max_guesses=6)
        self.MR = MR(max_guesses=6)

        self.test_predictions = dict()

    def forward(self, data: HeteroData):
        scene_enc = self.encoder(data)
        pred = self.decoder(data, scene_enc)
        return pred
    # 函数头只能传入batch和batch_idx，其中这里的data就是dataloader传进的batch数据；
    def training_step(self,
                      data,
                      batch_idx):
        # torch.cuda.empty_cache() # 清除gpu缓存空间，防止显存溢出发生OOM；torch.cuda.empty_cache()
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        reg_mask = data['agent']['predict_mask'][:, self.num_historical_steps:] # shape(agents, 60)
        cls_mask = data['agent']['predict_mask'][:, -1]
        pred = self(data) # 执行forward函数预测；
        if self.output_head:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['loc_propose_head'],
                                      pred['scale_propose_pos'][..., :self.output_dim],
                                      pred['conc_propose_head']], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['scale_propose_pos'][..., :self.output_dim]], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
            multi_traj_propose = torch.cat([pred['multi_loc_proposal'][..., :self.output_dim],
                                      pred['multi_loc_propose_scale'][..., :self.output_dim]], dim=-1)
        pi = pred['pi']
        proposal_pi = pred['proposal_pi']
        gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)
        # l2_norm得到所有agent的所有模态和GT的全部预测时间步的l2损失；sum之后会减少一个维度；l2 shape(agents, modes)，存储全部agents的每个模态的轨迹损失之和；
        l2_norm = (torch.norm(traj_propose[..., :self.output_dim] -
                              gt[..., :self.output_dim].unsqueeze(1), p=2, dim=-1) * reg_mask.unsqueeze(1)).sum(dim=-1) # unsqueeze的原因是建议轨迹他有多模态，而gt少了一个模态的维度；扩充之后的gt轨迹shape (agents, modes, t, 2)
        best_mode = l2_norm.argmin(dim=-1) # 通过建议轨迹来选择最优轨迹，而不是通过优化轨迹选择最优轨迹；shape(agents) 每一个元素对应每一个agent的最优的模态索引的数值；
        traj_propose_best = traj_propose[torch.arange(traj_propose.size(0)), best_mode] # 全部agents的最优模态的轨迹 shape(agents, 60, 2+2) 均值加方差，方差座位后面的概率来计算分类损失；
        traj_refine_best = traj_refine[torch.arange(traj_refine.size(0)), best_mode] # shape(num_nodes, 60, 4)
        # 在求损失的时候会计算所有agents的损失，只要这个agents在一个时间步有坐标就会计算其最佳轨迹的损失；
        reg_loss_propose = self.reg_loss(traj_propose_best,
                                         gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask # shape(num_nodes, 60)，只计算那些prediction_mask为1的那些l2损失；
        reg_loss_propose = reg_loss_propose.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1) # 就是将所有agents的60个时间步的损失在dim=0的维度上相加再求均值，这里的均值不是除以agents数，而是这个时间步有轨迹点的车辆数；shape(60)；
        # 实际的一个例子，如下：
        # tensor([3.3210, 4.2741, 4.8465, 5.5470, 5.9303, 6.1051, 6.4817, 6.6103,
        #         6.8038, 7.0390, 7.1521, 7.3535, 7.4517, 7.6924, 7.8528, 8.0690,
        #         8.1827, 8.3465, 8.4981, 8.5819, 8.6852, 8.8003, 8.8330, 8.9245,
        #         9.0252, 9.0772, 9.1675, 9.1695, 9.2319, 9.3215, 9.3539, 9.4295,
        #         9.4495, 9.5541, 9.6215, 9.6794, 9.7315, 9.8093, 9.8677, 9.9057,
        #         9.9677, 10.0425, 10.0628, 10.1085, 10.1605, 10.1911, 10.2420, 10.2410,
        #         10.2754, 10.3249, 10.3450, 10.3803, 10.4631, 10.5273, 10.5732, 10.6345,
        #         10.6641, 10.7074, 10.7408, 10.7239]
        reg_loss_propose = reg_loss_propose.mean() # shape(1)，上面的例子对应的损失的均值：8.83588981628418；
        reg_loss_refine = self.reg_loss(traj_refine_best,
                                        gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
        reg_loss_refine = reg_loss_refine.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
        reg_loss_refine = reg_loss_refine.mean()
        cls_loss = self.cls_loss(pred=traj_refine[:, :, -1:].detach(),
                                 target=gt[:, -1:, :self.output_dim + self.output_head],
                                 prob=pi,
                                 mask=reg_mask[:, -1:]) * cls_mask
        cls_loss = cls_loss.sum() / cls_mask.sum().clamp_(min=1)
        if proposal_pi is not None:
            cls_loss_region = self.cls_loss_region(pred=multi_traj_propose[:, :, -1:][(pred['num_equal']>=6)].detach(),
                                     target=gt[:, -1:, :self.output_dim + self.output_head][(pred['num_equal']>=6)],
                                     prob=proposal_pi,
                                     mask=reg_mask[:, -1:][(pred['num_equal']>=6)]) * ((cls_mask) & (pred['num_equal']>=6))[(pred['num_equal']>=6)]
            cls_loss_region = cls_loss_region.sum() / ((cls_mask) & (pred['num_equal'] >= 6)).sum().clamp_(min=1)
            loss = reg_loss_propose + reg_loss_refine + (0.5 * cls_loss_region + 0.5 * cls_loss)
        else:
            loss = reg_loss_propose + reg_loss_refine + 0.5 * cls_loss
            cls_loss_region = 0.

        self.log('train_reg_loss_propose', reg_loss_propose, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_reg_loss_refine', reg_loss_refine, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_cls_loss', cls_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_cls_region_loss', cls_loss_region, prog_bar=False, on_step=True, on_epoch=True,
                 batch_size=1)

        return loss

    def on_train_epoch_end(self):
        torch.cuda.empty_cache()

    # # 预测难度模块：应该对所有category=2和3的轨迹进行难度评估并标注（就像预测区域标注一样），从而在训练的时候能够有足够的数据；
    # def validation_step(self,
    #                     data,
    #                     batch_idx):
    #     torch.cuda.empty_cache()  # 清理gpu存储空间
    #     logging.basicConfig(filename='complexity_estimate.txt', level=logging.INFO)
    #     if isinstance(data, Batch):
    #         data['agent']['av_index'] += data['agent']['ptr'][:-1]
    #     reg_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]
    #     cls_mask = data['agent']['predict_mask'][:, -1]
    #     pred = self(data)
    #     if self.output_head:
    #         traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
    #                                   pred['loc_propose_head'],
    #                                   pred['scale_propose_pos'][..., :self.output_dim],
    #                                   pred['conc_propose_head']], dim=-1)
    #         traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
    #                                  pred['loc_refine_head'],
    #                                  pred['scale_refine_pos'][..., :self.output_dim],
    #                                  pred['conc_refine_head']], dim=-1)
    #     else:
    #         traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
    #                                   pred['scale_propose_pos'][..., :self.output_dim]], dim=-1)
    #         traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
    #                                  pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
    #     pi = pred['pi']
    #     gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)
    #     # l2_norm = (torch.norm(traj_propose[..., :self.output_dim] -
    #     #                       gt[..., :self.output_dim].unsqueeze(1), p=2, dim=-1) * reg_mask.unsqueeze(1)).sum(
    #     #     dim=-1)
    #     # best_mode = l2_norm.argmin(dim=-1)
    #     # traj_propose_best = traj_propose[torch.arange(traj_propose.size(0)), best_mode]
    #     # traj_refine_best = traj_refine[torch.arange(traj_refine.size(0)), best_mode]
    #     # reg_loss_propose = self.reg_loss(traj_propose_best,
    #     #                                  gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
    #     # reg_loss_propose = reg_loss_propose.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
    #     # reg_loss_propose = reg_loss_propose.mean()
    #     # reg_loss_refine = self.reg_loss(traj_refine_best,
    #     #                                 gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
    #     # reg_loss_refine = reg_loss_refine.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
    #     # reg_loss_refine = reg_loss_refine.mean()
    #     # cls_loss = self.cls_loss(pred=traj_refine[:, :, -1:].detach(),
    #     #                          target=gt[:, -1:, :self.output_dim + self.output_head],
    #     #                          prob=pi,
    #     #                          mask=reg_mask[:, -1:]) * cls_mask
    #     # cls_loss = cls_loss.sum() / cls_mask.sum().clamp_(min=1)
    #     # self.log('val_reg_loss_propose', reg_loss_propose, prog_bar=True, on_step=False, on_epoch=True,
    #     #          batch_size=1,
    #     #          sync_dist=True)
    #     # self.log('val_reg_loss_refine', reg_loss_refine, prog_bar=True, on_step=False, on_epoch=True, batch_size=1,
    #     #          sync_dist=True)
    #     # self.log('val_cls_loss', cls_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1,
    #     #          sync_dist=True)
    #     # 把complexity先都填为-1，然后预测2和3的难度，生成fde shape(agents)；在按照eval_mask的方式选出那些类型的complexity行，然后填进去；如果是多个batch的图data，那么最后还需要按照场景id来划分不同的fde，存入对应的文件中来；
    #     complexity = torch.full((data['agent']['num_nodes'], ), -1, dtype=torch.long)
    #
    #     if self.dataset == 'argoverse_v2':
    #         eval_mask = (data['agent']['category'] == 2 | data['agent']['category'] == 3)  # 在验证的时候只验证轨迹类型为3的，一个场景中就一个，也就是focal_track这个轨迹类型的车辆；
    #     else:
    #         raise ValueError('{} is not a valid dataset'.format(self.dataset))
    #     valid_mask_eval = reg_mask[eval_mask]  # shape(scored_agents, 60) 轨迹类型为2或3的将被选出；
    #     traj_eval = traj_refine[eval_mask, :, :,
    #                 :self.output_dim + self.output_head]  # shape(scored_agents, num_modes, time_steps, 2)
    #
    #     if not self.output_head:
    #         traj_2d_with_start_pos_eval = torch.cat([traj_eval.new_zeros((traj_eval.size(0), self.num_modes, 1, 2)),
    #                                                  traj_eval[..., :2]], dim=-2)
    #         motion_vector_eval = traj_2d_with_start_pos_eval[:, :, 1:] - traj_2d_with_start_pos_eval[:, :, :-1]
    #         head_eval = torch.atan2(motion_vector_eval[..., 1], motion_vector_eval[..., 0])
    #         traj_eval = torch.cat([traj_eval, head_eval.unsqueeze(-1)], dim=-1)
    #     pi_eval = F.softmax(pi[eval_mask], dim=-1)  # pi_eval shape(scored_agents, 6)
    #     gt_eval = gt[eval_mask]  # shape(scored_agents, num_modes, time_steps, 2)
    #     # target shape(4, 60, 2) 4表示batch_size也就是筛选出4个场景的target车辆, 60表示t；pred shape(4, 6, 60, 2) 6表示six modes；
    #     pred, target, prob, valid_mask, _ = valid_filter(pred=traj_eval[..., :self.output_dim],
    #                                                      target=gt_eval[..., :self.output_dim], prob=pi_eval,
    #                                                      valid_mask=valid_mask_eval)  # 选出target车辆的预测轨迹；
    #     pred_topk, _ = topk(6, pred,
    #                         prob)  # 没做任何事情，只是验证预测的轨迹多模态的数量是不是等于第一个参数max_guesses=6；pred_topk shape(batch, 6, 60, 2)
    #     inds_last = (valid_mask * torch.arange(1, valid_mask.size(-1) + 1, device=self.device)).argmax(
    #         dim=-1)  # shape(batch) --> [59, 59, 59, 59]每一个元素都是当前预测的最后一个时间步的索引；
    #     fde6 = torch.norm(pred_topk[torch.arange(pred.size(0)), :, inds_last] -
    #                       target[torch.arange(pred.size(0)), inds_last].unsqueeze(-2),
    #                       p=2, dim=-1).min(dim=-1)[0]  # shape(scored_agents) 计算并选出距离GT endpoints最近的一个模态的距离，也就是min_fde6；
    #     complexity[eval_mask] = (fde6>1.250).to(torch.long) # 1困难，0简单；shape(scored_agents)
    #     logging.info(f'fde6: {fde6}')
    #     # 下面的方式只适用于batch_size=1的模型：
    #     new_processed_file_name = os.path.join('/space/renjx/qcnet_processed_datasets/',
    #                                            data['scenario_id'] + '.pkl')
    #     with open(new_processed_file_name, 'rb') as file:
    #         # 使用pickle.load()方法加载对象
    #         data_one_graph = pickle.load(file)
    #         data_one_graph['complexity'] = complexity
    #     with open(new_processed_file_name, 'wb') as file:
    #         pickle.dump(data_one_graph, file)
    #     # 下面是batch_size>1的模型：
    #     # if isinstance(data, Batch):  # 对于Batch的data而言，多个图的节点数据属性在纵向累积，而不像其他任务的batch一样单起一个维度；
    #     #     for i in range(data.num_graphs):
    #     #         new_processed_file_name = os.path.join('/space/renjx/qcnet_processed_datasets/',
    #     #                                                data['scenario_id'][i] + '.pkl')
    #     #         with open(new_processed_file_name, 'rb') as file:
    #     #             # 使用pickle.load()方法加载对象
    #     #             data_one_graph = pickle.load(file)
    #     #         if (fde6[i] > 1.250):
    #     #             data_one_graph['complexity'] = torch.tensor(1, dtype=torch.long)
    #     #         else:
    #     #             data_one_graph['complexity'] = torch.tensor(0, dtype=torch.long)
    #     #         with open(new_processed_file_name, 'wb') as file:
    #     #             pickle.dump(data_one_graph, file)
    #     # else:
    #     #     new_processed_file_name = os.path.join('/space/renjx/qcnet_processed_datasets/',
    #     #                                            data['scenario_id'] + '.pkl')
    #     #     if (fde6 > 1.250):
    #     #         data['complexity'] = torch.tensor(1, dtype=torch.long)
    #     #     else:
    #     #         data['complexity'] = torch.tensor(0, dtype=torch.long)
    #     #     with open(new_processed_file_name, 'wb') as file:
    #     #         pickle.dump(data, file)

    # # 预测难度模块：这里只是选出target车辆轨迹；
    # def validation_step(self,
    #                     data,
    #                     batch_idx):
    #     torch.cuda.empty_cache()  # 清理gpu存储空间
    #     logging.basicConfig(filename='complexity_estimate.txt', level=logging.INFO)
    #     if isinstance(data, Batch):
    #         data['agent']['av_index'] += data['agent']['ptr'][:-1]
    #     reg_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]
    #     cls_mask = data['agent']['predict_mask'][:, -1]
    #     pred = self(data)
    #     if self.output_head:
    #         traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
    #                                   pred['loc_propose_head'],
    #                                   pred['scale_propose_pos'][..., :self.output_dim],
    #                                   pred['conc_propose_head']], dim=-1)
    #         traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
    #                                  pred['loc_refine_head'],
    #                                  pred['scale_refine_pos'][..., :self.output_dim],
    #                                  pred['conc_refine_head']], dim=-1)
    #     else:
    #         traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
    #                                   pred['scale_propose_pos'][..., :self.output_dim]], dim=-1)
    #         traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
    #                                  pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
    #     pi = pred['pi']
    #     gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)
    #     l2_norm = (torch.norm(traj_propose[..., :self.output_dim] -
    #                           gt[..., :self.output_dim].unsqueeze(1), p=2, dim=-1) * reg_mask.unsqueeze(1)).sum(dim=-1)
    #     best_mode = l2_norm.argmin(dim=-1)
    #     traj_propose_best = traj_propose[torch.arange(traj_propose.size(0)), best_mode]
    #     traj_refine_best = traj_refine[torch.arange(traj_refine.size(0)), best_mode]
    #     reg_loss_propose = self.reg_loss(traj_propose_best,
    #                                      gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
    #     reg_loss_propose = reg_loss_propose.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
    #     reg_loss_propose = reg_loss_propose.mean()
    #     reg_loss_refine = self.reg_loss(traj_refine_best,
    #                                     gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
    #     reg_loss_refine = reg_loss_refine.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
    #     reg_loss_refine = reg_loss_refine.mean()
    #     cls_loss = self.cls_loss(pred=traj_refine[:, :, -1:].detach(),
    #                              target=gt[:, -1:, :self.output_dim + self.output_head],
    #                              prob=pi,
    #                              mask=reg_mask[:, -1:]) * cls_mask
    #     cls_loss = cls_loss.sum() / cls_mask.sum().clamp_(min=1)
    #     self.log('val_reg_loss_propose', reg_loss_propose, prog_bar=True, on_step=False, on_epoch=True, batch_size=1,
    #              sync_dist=True)
    #     self.log('val_reg_loss_refine', reg_loss_refine, prog_bar=True, on_step=False, on_epoch=True, batch_size=1,
    #              sync_dist=True)
    #     self.log('val_cls_loss', cls_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
    #
    #     if self.dataset == 'argoverse_v2':
    #         eval_mask = data['agent']['category'] == 3  # 在验证的时候只验证轨迹类型为3的，一个场景中就一个，也就是focal_track这个轨迹类型的车辆；
    #     else:
    #         raise ValueError('{} is not a valid dataset'.format(self.dataset))
    #     valid_mask_eval = reg_mask[eval_mask] # shape(batch, 60)
    #     traj_eval = traj_refine[eval_mask, :, :,
    #                 :self.output_dim + self.output_head]  # single prediction：shape(batch, num_modes, time_steps, 2)
    #
    #     if not self.output_head:
    #         traj_2d_with_start_pos_eval = torch.cat([traj_eval.new_zeros((traj_eval.size(0), self.num_modes, 1, 2)),
    #                                                  traj_eval[..., :2]], dim=-2)
    #         motion_vector_eval = traj_2d_with_start_pos_eval[:, :, 1:] - traj_2d_with_start_pos_eval[:, :, :-1]
    #         head_eval = torch.atan2(motion_vector_eval[..., 1], motion_vector_eval[..., 0])
    #         traj_eval = torch.cat([traj_eval, head_eval.unsqueeze(-1)], dim=-1)
    #     pi_eval = F.softmax(pi[eval_mask], dim=-1) # pi_eval shape(4, 6) 4表示4个batch的target车辆，6表示6条模态轨迹的概率；
    #     gt_eval = gt[eval_mask] # shape(batch, num_modes, time_steps, 2)
    #     # target shape(4, 60, 2) 4表示batch_size也就是筛选出4个场景的target车辆, 60表示t；pred shape(4, 6, 60, 2) 6表示six modes；
    #     pred, target, prob, valid_mask, _ = valid_filter(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval, valid_mask=valid_mask_eval)  # 选出target车辆的预测轨迹；
    #     pred_topk, _ = topk(6, pred, prob) # 没做任何事情，只是验证预测的轨迹多模态的数量是不是等于第一个参数max_guesses=6；pred_topk shape(batch, 6, 60, 2)
    #     inds_last = (valid_mask * torch.arange(1, valid_mask.size(-1) + 1, device=self.device)).argmax(dim=-1) # shape(batch) --> [59, 59, 59, 59]每一个元素都是当前预测的最后一个时间步的索引；
    #     fde6 = torch.norm(pred_topk[torch.arange(pred.size(0)), :, inds_last] -
    #                            target[torch.arange(pred.size(0)), inds_last].unsqueeze(-2),
    #                            p=2, dim=-1).min(dim=-1)[0] # shape(batch) 计算并选出距离GT endpoints最近的一个模态的距离，也就是min_fde6；
    #     logging.info(f'fde6: {fde6}')
    #     if isinstance(data, Batch): # 对于Batch的data而言，多个图的节点数据属性在纵向累积，而不像其他任务的batch一样单起一个维度；
    #         for i in range(data.num_graphs):
    #             new_processed_file_name = os.path.join('/space/renjx/qcnet_processed_datasets/', data['scenario_id'][i]+'.pkl')
    #             with open(new_processed_file_name, 'rb') as file:
    #                 # 使用pickle.load()方法加载对象
    #                 data_one_graph = pickle.load(file)
    #             if (fde6[i] > 1.250):
    #                 data_one_graph['complexity'] = torch.tensor(1, dtype=torch.long)
    #             else:
    #                 data_one_graph['complexity'] = torch.tensor(0, dtype=torch.long)
    #             with open(new_processed_file_name, 'wb') as file:
    #                 pickle.dump(data_one_graph, file)
    #     else:
    #         new_processed_file_name = os.path.join('/space/renjx/qcnet_processed_datasets/', data['scenario_id']+'.pkl')
    #         if (fde6 > 1.250):
    #             data['complexity'] = torch.tensor(1, dtype=torch.long)
    #         else:
    #             data['complexity'] = torch.tensor(0, dtype=torch.long)
    #         with open(new_processed_file_name, 'wb') as file:
    #             pickle.dump(data, file)




    def validation_step(self,
                        data,
                        batch_idx):
        # torch.cuda.empty_cache() # 清理gpu存储空间
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        reg_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]
        cls_mask = data['agent']['predict_mask'][:, -1]
        pred = self(data)
        if self.output_head:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['loc_propose_head'],
                                      pred['scale_propose_pos'][..., :self.output_dim],
                                      pred['conc_propose_head']], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['scale_propose_pos'][..., :self.output_dim]], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
        pi = pred['pi']
        gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)
        if self.dataset == 'argoverse_v2':
            eval_mask = data['agent']['category'] == 3 # 在验证的时候只验证轨迹类型为3的，一个场景中就一个，也就是focal_track这个轨迹类型的车辆；
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        valid_mask_eval = reg_mask[eval_mask]
        traj_eval = traj_refine[eval_mask, :, :, :self.output_dim + self.output_head] # single prediction：shape(num_modes, time_steps, 2)

        if not self.output_head:
            traj_2d_with_start_pos_eval = torch.cat([traj_eval.new_zeros((traj_eval.size(0), 6, 1, 2)),
                                                     traj_eval[..., :2]], dim=-2)
            motion_vector_eval = traj_2d_with_start_pos_eval[:, :, 1:] - traj_2d_with_start_pos_eval[:, :, :-1]
            head_eval = torch.atan2(motion_vector_eval[..., 1], motion_vector_eval[..., 0])
            traj_eval = torch.cat([traj_eval, head_eval.unsqueeze(-1)], dim=-1)
        pi_eval = F.softmax(pi[eval_mask], dim=-1)
        gt_eval = gt[eval_mask]

        self.Brier.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                          valid_mask=valid_mask_eval) # 每一次更新都会将这个metric加入到这个对象当中来，然后一个epoch结束之后，self.log函数就会通过自定义的compute操作来计算一个批次的指标值；
        self.minADE.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                           valid_mask=valid_mask_eval)
        self.minAHE.update(pred=traj_eval, target=gt_eval, prob=pi_eval, valid_mask=valid_mask_eval)
        self.minFDE.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                           valid_mask=valid_mask_eval)
        self.minFHE.update(pred=traj_eval, target=gt_eval, prob=pi_eval, valid_mask=valid_mask_eval)
        self.MR.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                       valid_mask=valid_mask_eval)
        self.log('val_Brier', self.Brier, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_minADE', self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_minAHE', self.minAHE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0)) # 第64个epoch的时候，一个epoch的平均minFDE是1.250，以这个指标为基准判断难易轨迹；
        self.log('val_minFHE', self.minFHE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_MR', self.MR, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))

    def on_validation_epoch_end(self):
        torch.cuda.empty_cache()


    # 真正提交测试的时候需要用到的代码；
    def test_step(self,
                  data,
                  batch_idx):
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        pred = self(data)
        if self.output_head:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
        pi = pred['pi']
        if self.dataset == 'argoverse_v2':
            eval_mask = data['agent']['category'] == 3
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        # 转换为全局坐标系的坐标；
        origin_eval = data['agent']['position'][eval_mask, self.num_historical_steps - 1]
        theta_eval = data['agent']['heading'][eval_mask, self.num_historical_steps - 1]
        cos, sin = theta_eval.cos(), theta_eval.sin()
        rot_mat = torch.zeros(eval_mask.sum(), 2, 2, device=self.device)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = sin
        rot_mat[:, 1, 0] = -sin
        rot_mat[:, 1, 1] = cos
        traj_eval = torch.matmul(traj_refine[eval_mask, :, :, :2],
                                 rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)
        pi_eval = F.softmax(pi[eval_mask], dim=-1)

        traj_eval = traj_eval.cpu().numpy()
        pi_eval = pi_eval.cpu().numpy()

        if self.dataset == 'argoverse_v2':
            eval_id = list(compress(list(chain(*data['agent']['id'])), eval_mask))
            if isinstance(data, Batch):
                for i in range(data.num_graphs):
                    self.test_predictions[data['scenario_id'][i]] = {eval_id[i]: (traj_eval[i], pi_eval[i])}
            else:
                self.test_predictions[data['scenario_id']] = {eval_id[0]: (traj_eval[0], pi_eval[0])} # {id:{track_id:{pred, prob}}}
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))

    def on_test_end(self):
        if self.dataset == 'argoverse_v2':
            ChallengeSubmission(self.test_predictions).to_parquet(
                Path(self.submission_dir) / f'{self.submission_file_name}.parquet')
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.T_max, eta_min=0.0) # 余弦衰减方式来随着epoch的增加而降低学习率；
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('QCNet') # 这组参数对象的名字是QCNet；
        parser.add_argument('--dataset', type=str, required=True)
        parser.add_argument('--input_dim', type=int, default=2)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--output_dim', type=int, default=2)
        parser.add_argument('--output_head', action='store_true')
        parser.add_argument('--num_historical_steps', type=int, required=True)
        parser.add_argument('--num_future_steps', type=int, required=True)
        parser.add_argument('--num_modes', type=int, default=30) # 这里修改模态数量；
        parser.add_argument('--num_recurrent_steps', type=int, required=True)
        parser.add_argument('--num_freq_bands', type=int, default=64)
        parser.add_argument('--num_map_layers', type=int, default=1)
        parser.add_argument('--num_agent_layers', type=int, default=2)
        parser.add_argument('--num_dec_layers', type=int, default=2)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--head_dim', type=int, default=16)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--pl2pl_radius', type=float, required=True)
        parser.add_argument('--time_span', type=int, default=None)
        parser.add_argument('--pl2a_radius', type=float, required=True)
        parser.add_argument('--a2a_radius', type=float, required=True)
        parser.add_argument('--num_t2m_steps', type=int, default=None)
        parser.add_argument('--pl2m_radius', type=float, required=True)
        parser.add_argument('--a2m_radius', type=float, required=True)
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--T_max', type=int, default=100)
        parser.add_argument('--submission_dir', type=str, default='./')
        parser.add_argument('--submission_file_name', type=str, default='submission')
        return parent_parser

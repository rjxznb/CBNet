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
import math
from typing import Dict, List, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import radius
from torch_cluster import radius_graph
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData # 异质图类型的一个图对象：节点的属性和节点间连接边表示的关系不同；
from torch_geometric.utils import dense_to_sparse

from layers import AttentionLayer
from layers import FourierEmbedding
from layers import MLPLayer
from utils import angle_between_2d_vectors
from utils import bipartite_dense_to_sparse
from utils import weight_init
from utils import wrap_angle
import joblib

class QCNetDecoder(nn.Module):

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
                 num_t2m_steps: Optional[int],
                 pl2m_radius: float,
                 a2m_radius: float,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float) -> None:
        super(QCNetDecoder, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_head = output_head
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_modes = num_modes
        self.num_recurrent_steps = num_recurrent_steps
        self.num_t2m_steps = num_t2m_steps if num_t2m_steps is not None else num_historical_steps
        self.pl2m_radius = pl2m_radius
        self.a2m_radius = a2m_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        input_dim_r_t = 4
        input_dim_r_pl2m = 3
        input_dim_r_a2m = 3
        self.kmeans = joblib.load('/space/renjx/qcnet_region/kmeans_model.pkl')
        # region_mlp区域预测头：
        region_mlp = torch.load('./region_proposal.pkl', map_location='cpu')
        self.region = MLPLayer(128, 128, 6)
        self.region.load_state_dict(region_mlp)
        self.region.requires_grad_(False)
        self.mode_emb = nn.Embedding(self.num_modes, self.hidden_dim) # 这里表示最初输入decoder的建议轨迹的模态数量，也就是随机生成的编码；
        self.r_t2m_emb = FourierEmbedding(input_dim=input_dim_r_t, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_pl2m_emb = FourierEmbedding(input_dim=input_dim_r_pl2m, hidden_dim=hidden_dim,
                                           num_freq_bands=num_freq_bands)
        self.r_a2m_emb = FourierEmbedding(input_dim=input_dim_r_a2m, hidden_dim=hidden_dim,
                                          num_freq_bands=num_freq_bands)
        self.y_emb = FourierEmbedding(input_dim=output_dim + output_head, hidden_dim=hidden_dim,
                                      num_freq_bands=num_freq_bands)
        self.traj_emb = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, bias=True,
                               batch_first=False, dropout=0.0, bidirectional=False)
        self.traj_emb_h0 = nn.Parameter(torch.zeros(1, hidden_dim))
        self.t2m_propose_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.pl2m_propose_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.a2m_propose_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.m2m_propose_attn_layer = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                                     dropout=dropout, bipartite=False, has_pos_emb=False)
        self.t2m_refine_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.pl2m_refine_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.a2m_refine_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.m2m_refine_attn_layer = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                                    dropout=dropout, bipartite=False, has_pos_emb=False)
        self.to_loc_propose_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                           output_dim=num_future_steps * output_dim // num_recurrent_steps)
        self.to_scale_propose_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                             output_dim=num_future_steps * output_dim // num_recurrent_steps)
        self.to_loc_refine_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                          output_dim=num_future_steps * output_dim)
        self.to_scale_refine_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                            output_dim=num_future_steps * output_dim)
        if output_head:
            self.to_loc_propose_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                output_dim=num_future_steps // num_recurrent_steps)
            self.to_conc_propose_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                 output_dim=num_future_steps // num_recurrent_steps)
            self.to_loc_refine_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=num_future_steps)
            self.to_conc_refine_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                output_dim=num_future_steps)
        else:
            self.to_loc_propose_head = None
            self.to_conc_propose_head = None
            self.to_loc_refine_head = None
            self.to_conc_refine_head = None
        self.to_pi = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=1)
        self.region_multimdoes_pi = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=1) # 用于为选中区域内的所有多模态轨迹计算概率，选择概率前6的轨迹；
        self.apply(weight_init) # 为所有层来初始化权重滴呦；

    def forward(self,
                data: HeteroData, # 读取的data如果是Batch对象，也就是多个图组合为data，那么所有的属性都是4个图的总数，比如data['agents']['num_nodes'] = 4个图的agents；data['agents']['postion']的shape(4个图的agents, 60, 2)
                scene_enc: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]: # scene_enc就是encoder的输出；
        # 预测每一辆车GT endpoints的所在区域：
        # x = scene_enc['x_a'][:, -1]
        # pred_region = self.region(x) # 可以尝试可学习和不可学习两种形式；
        # pred_region = torch.argmax(pred_region, dim=-1) # shape(agents)；但是在真正预测的时候需要考虑轨迹类型是否为2和3，在轨迹预测里是的single prediciton验证和测试任务是预测3，而不是2和3；训练时可以用任何的轨迹，在QCNet里面用的是类型为1，2，3的轨迹；

        pos_m = data['agent']['position'][:, self.num_historical_steps - 1, :self.input_dim]
        head_m = data['agent']['heading'][:, self.num_historical_steps - 1]
        head_vector_m = torch.stack([head_m.cos(), head_m.sin()], dim=-1)

        x_t = scene_enc['x_a'].reshape(-1, self.hidden_dim) # shape(agents x history_time_steps_, hidden_dims)
        x_pl = scene_enc['x_pl'][:, self.num_historical_steps - 1].repeat(self.num_modes, 1)
        x_pl_refine = scene_enc['x_pl'][:, self.num_historical_steps - 1].repeat(6, 1) # 为了在refine的时候6个模态能用
        x_a = scene_enc['x_a'][:, -1].repeat(self.num_modes, 1)
        x_a_refine = scene_enc['x_a'][:, -1].repeat(6, 1) # 为了在refine的时候6个模态能用；这里我改啦；
        m = self.mode_emb.weight.repeat(scene_enc['x_a'].size(0), 1) # 输入到decoder的最初多模态轨迹；

        mask_src = data['agent']['valid_mask'][:, :self.num_historical_steps].contiguous()
        mask_src[:, :self.num_historical_steps - self.num_t2m_steps] = False
        mask_dst = data['agent']['predict_mask'].any(dim=-1, keepdim=True).repeat(1, self.num_modes)
        mask_dst_refine = data['agent']['predict_mask'].any(dim=-1, keepdim=True).repeat(1, 6)

        pos_t = data['agent']['position'][:, :self.num_historical_steps, :self.input_dim].reshape(-1, self.input_dim)
        head_t = data['agent']['heading'][:, :self.num_historical_steps].reshape(-1)
        edge_index_t2m = bipartite_dense_to_sparse(mask_src.unsqueeze(2) & mask_dst[:, -1:].unsqueeze(1))
        edge_index_t2m_refine = bipartite_dense_to_sparse(mask_src.unsqueeze(2) & mask_dst_refine[:, -1:].unsqueeze(1))
        rel_pos_t2m = pos_t[edge_index_t2m[0]] - pos_m[edge_index_t2m[1]]
        rel_pos_t2m_refine = pos_t[edge_index_t2m_refine[0]] - pos_m[edge_index_t2m_refine[1]]
        rel_head_t2m = wrap_angle(head_t[edge_index_t2m[0]] - head_m[edge_index_t2m[1]])
        rel_head_t2m_refine = wrap_angle(head_t[edge_index_t2m_refine[0]] - head_m[edge_index_t2m_refine[1]])
        r_t2m = torch.stack(
            [torch.norm(rel_pos_t2m[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_t2m[1]], nbr_vector=rel_pos_t2m[:, :2]),
             rel_head_t2m,
             (edge_index_t2m[0] % self.num_historical_steps) - self.num_historical_steps + 1], dim=-1)
        r_t2m_refine = torch.stack(
            [torch.norm(rel_pos_t2m_refine[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_t2m_refine[1]], nbr_vector=rel_pos_t2m_refine[:, :2]),
             rel_head_t2m_refine,
             (edge_index_t2m_refine[0] % self.num_historical_steps) - self.num_historical_steps + 1], dim=-1)
        r_t2m = self.r_t2m_emb(continuous_inputs=r_t2m, categorical_embs=None)
        r_t2m_refine = self.r_t2m_emb(continuous_inputs=r_t2m_refine, categorical_embs=None)
        edge_index_t2m = bipartite_dense_to_sparse(mask_src.unsqueeze(2) & mask_dst.unsqueeze(1))
        edge_index_t2m_refine = bipartite_dense_to_sparse(mask_src.unsqueeze(2) & mask_dst_refine.unsqueeze(1))
        r_t2m = r_t2m.repeat_interleave(repeats=self.num_modes, dim=0)
        r_t2m_refine = r_t2m_refine.repeat_interleave(repeats=6, dim=0)

        pos_pl = data['map_polygon']['position'][:, :self.input_dim]
        orient_pl = data['map_polygon']['orientation']
        edge_index_pl2m = radius(
            x=pos_m[:, :2],
            y=pos_pl[:, :2],
            r=self.pl2m_radius,
            batch_x=data['agent']['batch'] if isinstance(data, Batch) else None,
            batch_y=data['map_polygon']['batch'] if isinstance(data, Batch) else None,
            max_num_neighbors=300)
        edge_index_pl2m = edge_index_pl2m[:, mask_dst[edge_index_pl2m[1], 0]]
        edge_index_pl2m_refine = edge_index_pl2m[:, mask_dst_refine[edge_index_pl2m[1], 0]]
        rel_pos_pl2m = pos_pl[edge_index_pl2m[0]] - pos_m[edge_index_pl2m[1]]
        rel_pos_pl2m_refine = pos_pl[edge_index_pl2m_refine[0]] - pos_m[edge_index_pl2m_refine[1]]
        rel_orient_pl2m = wrap_angle(orient_pl[edge_index_pl2m[0]] - head_m[edge_index_pl2m[1]])
        rel_orient_pl2m_refine = wrap_angle(orient_pl[edge_index_pl2m_refine[0]] - head_m[edge_index_pl2m_refine[1]])
        r_pl2m = torch.stack(
            [torch.norm(rel_pos_pl2m[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_pl2m[1]], nbr_vector=rel_pos_pl2m[:, :2]),
             rel_orient_pl2m], dim=-1)
        r_pl2m_refine = torch.stack(
            [torch.norm(rel_pos_pl2m_refine[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_pl2m_refine[1]], nbr_vector=rel_pos_pl2m_refine[:, :2]),
             rel_orient_pl2m_refine], dim=-1)
        r_pl2m = self.r_pl2m_emb(continuous_inputs=r_pl2m, categorical_embs=None)
        r_pl2m_refine = self.r_pl2m_emb(continuous_inputs=r_pl2m_refine, categorical_embs=None)
        edge_index_pl2m = torch.cat([edge_index_pl2m + i * edge_index_pl2m.new_tensor(
            [[data['map_polygon']['num_nodes']], [data['agent']['num_nodes']]]) for i in range(self.num_modes)], dim=1)
        edge_index_pl2m_refine = torch.cat([edge_index_pl2m_refine + i * edge_index_pl2m_refine.new_tensor(
            [[data['map_polygon']['num_nodes']], [data['agent']['num_nodes']]]) for i in range(6)], dim=1)
        r_pl2m = r_pl2m.repeat(self.num_modes, 1)
        r_pl2m_refine = r_pl2m_refine.repeat(6, 1)

        edge_index_a2m = radius_graph(
            x=pos_m[:, :2],
            r=self.a2m_radius,
            batch=data['agent']['batch'] if isinstance(data, Batch) else None,
            loop=False,
            max_num_neighbors=300)
        edge_index_a2m = edge_index_a2m[:, mask_src[:, -1][edge_index_a2m[0]] & mask_dst[edge_index_a2m[1], 0]]
        edge_index_a2m_refine = edge_index_a2m[:, mask_src[:, -1][edge_index_a2m[0]] & mask_dst_refine[edge_index_a2m[1], 0]]
        rel_pos_a2m = pos_m[edge_index_a2m[0]] - pos_m[edge_index_a2m[1]]
        rel_pos_a2m_refine = pos_m[edge_index_a2m_refine[0]] - pos_m[edge_index_a2m_refine[1]]
        rel_head_a2m = wrap_angle(head_m[edge_index_a2m[0]] - head_m[edge_index_a2m[1]])
        rel_head_a2m_refine = wrap_angle(head_m[edge_index_a2m_refine[0]] - head_m[edge_index_a2m_refine[1]])
        r_a2m = torch.stack(
            [torch.norm(rel_pos_a2m[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_a2m[1]], nbr_vector=rel_pos_a2m[:, :2]),
             rel_head_a2m], dim=-1)
        r_a2m_refine = torch.stack(
            [torch.norm(rel_pos_a2m_refine[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_a2m_refine[1]], nbr_vector=rel_pos_a2m_refine[:, :2]),
             rel_head_a2m_refine], dim=-1)
        r_a2m = self.r_a2m_emb(continuous_inputs=r_a2m, categorical_embs=None)
        r_a2m_refine = self.r_a2m_emb(continuous_inputs=r_a2m_refine, categorical_embs=None)
        edge_index_a2m = torch.cat(
            [edge_index_a2m + i * edge_index_a2m.new_tensor([data['agent']['num_nodes']]) for i in
             range(self.num_modes)], dim=1)
        r_a2m = r_a2m.repeat(self.num_modes, 1)
        edge_index_a2m_refine = torch.cat(
            [edge_index_a2m_refine + i * edge_index_a2m_refine.new_tensor([data['agent']['num_nodes']]) for i in
             range(6)], dim=1)
        r_a2m_refine = r_a2m_refine.repeat(6, 1)

        # edge_index_m2m = dense_to_sparse(mask_dst.unsqueeze(2) & mask_dst.unsqueeze(1))[0]
        edge_index_m2m_refine = dense_to_sparse(mask_dst_refine.unsqueeze(2) & mask_dst_refine.unsqueeze(1))[0]

        locs_propose_pos: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        scales_propose_pos: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        locs_propose_head: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        concs_propose_head: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        for t in range(self.num_recurrent_steps): # 循环融合proposal轨迹；
            for i in range(self.num_layers):
                m = m.reshape(-1, self.hidden_dim)
                m = self.t2m_propose_attn_layers[i]((x_t, m), r_t2m, edge_index_t2m)
                m = m.reshape(-1, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
                m = self.pl2m_propose_attn_layers[i]((x_pl, m), r_pl2m, edge_index_pl2m)
                m = self.a2m_propose_attn_layers[i]((x_a, m), r_a2m, edge_index_a2m)
                m = m.reshape(self.num_modes, -1, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
            # m = self.m2m_propose_attn_layer(m, None, edge_index_m2m)
            m = m.reshape(-1, self.num_modes, self.hidden_dim)
            locs_propose_pos[t] = self.to_loc_propose_pos(m) # 没进行累加的坐标，只是保留相邻两点间的距离大小；可以在这里用6个mlp表示6个region；
            scales_propose_pos[t] = self.to_scale_propose_pos(m) # 预测GT概率分布的方差也就对应这不确定度；
            if self.output_head:
                locs_propose_head[t] = self.to_loc_propose_head(m)
                concs_propose_head[t] = self.to_conc_propose_head(m)
        m1 = m.max(dim=1)[0]  # mode维度最大池化操作；
        pred_region = self.region(m1)  # 可以尝试可学习和不可学习两种形式；
        pred_region = torch.argmax(pred_region, dim=-1)  # shape(agents)；但是在真正预测的时候需要考虑轨迹类型是否为2和3，在轨迹预测里是的single prediciton验证和测试任务是预测3，而不是2和3；训练时可以用任何的轨迹，在QCNet里面用的是类型为1，2，3的轨迹；
        loc_propose_pos = torch.cumsum(
            torch.cat(locs_propose_pos, dim=-1).view(-1, self.num_modes, self.num_future_steps, self.output_dim),
            dim=-2) # cumsum是在时间步维度上做累加和，从而在轨迹坐标上呈现出一个递增的趋势；
        scale_propose_pos = torch.cumsum(
            F.elu_(
                torch.cat(scales_propose_pos, dim=-1).view(-1, self.num_modes, self.num_future_steps, self.output_dim),
                alpha=1.0) +
            1.0,
            dim=-2) + 0.1

        # 聚类 --> shape(agents, mode, 1) : 1表示region_label；
        # 区别tensor.detach_(), clone(), clone().detach(), new_zeros()：
        # https://blog.csdn.net/qq_27825451/article/details/95498211：当我们再训练网络的时候可能希望保持一部分的网络参数不变，只对其中一部分的参数进行调整；或者值训练部分分支网络，并不让其梯度对主网络的梯度造成影响，这时候我们就需要使用detach()函数来切断一些分支的反向传播
        # 转换为numpy之前要先把数据放到cpu，这样可以通过clone建立新空间再放到cpu中来就好啦；
        endpoints = loc_propose_pos.clone().detach()[:, :, -1, :2].view(-1, 2).to('cpu').numpy() # shape(agents*modes, 2)；使用detach返回的tensor和原始的tensor共同一个内存，即一个修改另一个也会跟着改变，但是detach得到的tensor不再有梯度啦；修改detach之后的tensor再进行反向传播会
        region_label = self.kmeans.predict(endpoints) # shape(agents*modes)
        cluster_centers = self.kmeans.cluster_centers_ # 各区域的中心坐标；shape(6, 2)
        cluster_centers = torch.tensor(cluster_centers).cuda()
        region_cluster = cluster_centers.gather(dim=0, index=pred_region.view(-1, 1).repeat(1, 2)) # shape(agents, 2)各agents预测区域的中心；
        region_label = torch.tensor(region_label).view(data['agent']['num_nodes'], -1).cuda() # shape(agents, modes)
        pred_region = pred_region.view(region_label.size(0), 1).repeat(1, self.num_modes) # shape(agents, modes) # 将预测的区域在最后一个维度重复modes次；
        equal = (pred_region == region_label) # 一个bool类型的矩阵，若对应位置元素相等则为True；需要让两个矩阵的dtype相同；shape(agents, modes)；
        num_equal = equal.sum(dim=-1) # 对其sum之后就能得到有多少个endpoints在预测区域内；shape(agents)
        # 如果多模态轨迹的endpoints在目标区域数量大于6的话就选择可能性前6的轨迹进行优化，如果不大于6，可以在已有轨迹的基础上选择最终点距离目标区域的簇中心最近的那条轨迹；
        # 选出预测区域内多于6条轨迹的endpoints的那些agents的多模态建议轨迹进行计算；
        # 需要考虑在划为两部分agents的时候，如何按照原来的顺序合并回去；
        # more_agents_mask = (num_equal >= 6).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, self.num_modes, self.num_future_steps, loc_propose_pos.size(-1)) # shape(agents, modes, t, output_dims) 其中more_agents的行全为True，less_agents的行全为False；用来乘上来mask less的agents
        more_than_six_agents_proposal_trajectory = loc_propose_pos[(num_equal >= 6)]
        more_than_six_agents_proposal_scale = scale_propose_pos[(num_equal >= 6)]
        new_proposal = torch.zeros(data['agent']['num_nodes'], 6, self.num_future_steps, self.output_dim).cuda()  # 用于存放选择出的区域内6条多模态建议轨迹；
        new_proposal_scale = torch.zeros(data['agent']['num_nodes'], 6, self.num_future_steps, self.output_dim).cuda() # 用于存放选择出的区域内6条多模态建议轨迹的方差；
        if more_than_six_agents_proposal_trajectory.size(0) > 0:
            proposal_pi = self.region_multimdoes_pi(m[(num_equal >= 6)].detach()).squeeze(-1) # shape(more_agents, n_modes)这里的modes不一定是多少个，就是区域内所有的轨迹数；
            top6_modes = torch.topk(proposal_pi, 6, dim=-1).indices.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.num_future_steps, self.output_dim) # shape(more_agents, 6, t, output_dims)用于gather方法中的个index选择dim=1的那个维度的元素；
            # 使用 torch.gather 获取最大的前 6 个元素；shape(more_agents, 6, t, output_dims)
            more_than_six_agents_proposal_trajectory = torch.gather(more_than_six_agents_proposal_trajectory, dim=1, index=top6_modes) # shape(more_agents, 6, 60, 2)
            more_than_six_agents_proposal_scale = torch.gather(more_than_six_agents_proposal_scale, dim=1, index=top6_modes)
            new_proposal[(num_equal >= 6)] = more_than_six_agents_proposal_trajectory
            new_proposal_scale[(num_equal >= 6)] = more_than_six_agents_proposal_scale
        else:
            proposal_pi = None
        # 用同样维度的bool索引来进行切片操作只会选择出True的那些元素，并以一维tensor的形式返回，如果少一维度，就会以二维tensor的形式返回，是相互对应滴；
        # less_agents_mask = ~more_agents_mask  # less_agents的元素全为True，more_agents的行全为False；
        less_than_six_agents_proposal_trajectory = loc_propose_pos[(num_equal < 6)]
        less_than_six_agents_proposal_scale = scale_propose_pos[(num_equal < 6)]
        if less_than_six_agents_proposal_trajectory.size(0) > 0: # 找到less_agents在预测区域外的所有轨迹的endpoints，计算其同预测区域中心的距离；
            less_equal = equal[(num_equal < 6)] # shape(less_agents, 36)
            less_num_equals = num_equal[(num_equal < 6)]  # shape(less_agents)；
            lack_num = 6 - less_num_equals # shape(less_agents)；
            # 通过torch.xxx建立的tensor都是独立的空间，而通过=赋值的tensor都是同音空间的引用变量而已呦；
            endpoints = torch.tensor(endpoints).cuda()

            endpoints_l2_distance = torch.norm(endpoints.view(data['agent']['num_nodes'], self.num_modes, 2)[(num_equal < 6)] - region_cluster[(num_equal < 6)].unsqueeze(1).repeat(1, self.num_modes, 1), dim=-1) # 计算所有模态的endpoints的距离，最后再筛选出那些在region之外的点；shape(less_agents, modes)
            # 先选出目标区域内的距离，将他们加10000；再通过topk选出前6小的模态索引，通过lack_num来设定各agents需要选择前几个，并将他们装入list，然后通过list来为将equal[(num_equal<6)]的对应模态设为True；通过equal_less_agents bool切片索引从而loc_proposal映射出对应的模态轨迹；
            endpoints_l2_distance[less_equal] += 10000
            top6_modes = torch.topk(endpoints_l2_distance, 6, dim=-1, largest=False).indices
            selected_elements = [row[:lack_num[i].to(torch.int)].tolist() for i, row in enumerate(top6_modes)] # less_agents缺少的个数的模态索引；list shape(less_agents, lack_num)，其中这个lack_num每个agents的个数不一定相同；
            for i, li in enumerate(selected_elements):
                for j in li:
                    less_equal[i][j] = True
            less_than_six_agents_proposal_trajectory = less_than_six_agents_proposal_trajectory[less_equal].view(less_num_equals.size(0), 6, self.num_future_steps, self.output_dim) # shape(less_agents, 6, t, output_dims)
            less_than_six_agents_proposal_scale = less_than_six_agents_proposal_scale[less_equal].view(less_num_equals.size(0), 6, self.num_future_steps, self.output_dim)
            new_proposal[(num_equal < 6)] = less_than_six_agents_proposal_trajectory
            new_proposal_scale[(num_equal < 6)] = less_than_six_agents_proposal_scale

        # loc_propose_pos = new_proposal # 如果加了这句，就会造成原来的tensor梯度丢失；

        if self.output_head:
            loc_propose_head = torch.cumsum(torch.tanh(torch.cat(locs_propose_head, dim=-1).unsqueeze(-1)) * math.pi,
                                            dim=-2)
            conc_propose_head = 1.0 / (torch.cumsum(F.elu_(torch.cat(concs_propose_head, dim=-1).unsqueeze(-1)) + 1.0,
                                                    dim=-2) + 0.02)
            # 梯度截断之后就相当于另外一个模型任务，也就是计算另外一个loss；这里就是相当于refine任务；
            m = self.y_emb(torch.cat([new_proposal.detach(),
                                      wrap_angle(loc_propose_head.detach())], dim=-1).view(-1, self.output_dim + 1))
        else:
            loc_propose_head = new_proposal.new_zeros((new_proposal.size(0), 6,
                                                          self.num_future_steps, 1))
            conc_propose_head = scale_propose_pos.new_zeros((scale_propose_pos.size(0), 6,
                                                             self.num_future_steps, 1))
            l_propose_head = new_proposal.new_zeros((new_proposal.size(0), self.num_modes,
                                                   self.num_future_steps, 1))
            c_propose_head = scale_propose_pos.new_zeros((scale_propose_pos.size(0), self.num_modes,
                                                        self.num_future_steps, 1))
            m = self.y_emb(new_proposal.detach().view(-1, self.output_dim))
        m = m.reshape(-1, self.num_future_steps, self.hidden_dim).transpose(0, 1)
        m = self.traj_emb(m, self.traj_emb_h0.unsqueeze(1).repeat(1, m.size(1), 1))[1].squeeze(0)
        for i in range(self.num_layers):
            m = self.t2m_refine_attn_layers[i]((x_t, m), r_t2m_refine, edge_index_t2m_refine) # r_t2m还是36条轨迹呢
            m = m.reshape(-1, 6, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
            m = self.pl2m_refine_attn_layers[i]((x_pl_refine, m), r_pl2m_refine, edge_index_pl2m_refine)
            m = self.a2m_refine_attn_layers[i]((x_a_refine, m), r_a2m_refine, edge_index_a2m_refine)
            m = m.reshape(6, -1, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
        m = self.m2m_refine_attn_layer(m, None, edge_index_m2m_refine)
        m = m.reshape(-1, 6, self.hidden_dim)
        # 对建议轨迹和优化轨迹都要和GT轨迹计算损失，但是在对优化轨迹反向传播的时候会固定建议轨迹模块的参数，只对优化模块这一部分内容进行提高；
        loc_refine_pos = self.to_loc_refine_pos(m).view(-1, 6, self.num_future_steps, self.output_dim)
        loc_refine_pos = loc_refine_pos + new_proposal.detach() # 通过第二轮计算偏置，并且截断梯度，避免优化proposal轨迹之前的网络模型权重参数；
        scale_refine_pos = F.elu_(
            self.to_scale_refine_pos(m).view(-1, 6, self.num_future_steps, self.output_dim),
            alpha=1.0) + 1.0 + 0.1
        if self.output_head:
            loc_refine_head = torch.tanh(self.to_loc_refine_head(m).unsqueeze(-1)) * math.pi
            loc_refine_head = loc_refine_head + loc_propose_head.detach()
            conc_refine_head = 1.0 / (F.elu_(self.to_conc_refine_head(m).unsqueeze(-1)) + 1.0 + 0.02)
        else:
            loc_refine_head = loc_refine_pos.new_zeros((loc_refine_pos.size(0), 6, self.num_future_steps,
                                                        1))
            conc_refine_head = scale_refine_pos.new_zeros((scale_refine_pos.size(0), 6,
                                                           self.num_future_steps, 1))
        pi = self.to_pi(m).squeeze(-1) # shape(agents, modes)

        return {
            'num_equal': num_equal,
            'multi_loc_proposal': loc_propose_pos,  # 36
            'multi_loc_propose_scale': scale_propose_pos, # 36
            'loc_propose_pos': new_proposal, # shape(agents, 6, t, 2)
            'scale_propose_pos': new_proposal_scale,
            'loc_propose_head': loc_propose_head,
            'conc_propose_head': conc_propose_head,
            'loc_refine_pos': loc_refine_pos,
            'scale_refine_pos': scale_refine_pos,
            'loc_refine_head': loc_refine_head,
            'conc_refine_head': conc_refine_head,
            'proposal_pi': proposal_pi,
            'pi': pi, # shape(agents, 6)
        }

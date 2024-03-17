import torch
from predictors import QCNet
import pytorch_lightning as pl
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from layers import MLPLayer
from utils import angle_between_2d_vectors
from utils import bipartite_dense_to_sparse
from utils import wrap_angle
from torch_cluster import radius
from torch_cluster import radius_graph
from torch_geometric.utils import dense_to_sparse
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from datamodules import ArgoverseV2DataModule
import datetime


model = QCNet.load_from_checkpoint(checkpoint_path='/space/renjx/qcnet_region/epoch=63-step=399872.ckpt', map_location='cpu') # load模型时需要map_location='cpu'或是全部的gpu上面，否则会全部加载到gpu 0 中来，多了几个额外的进程；原因见：https://blog.csdn.net/cax1165/article/details/125862915#
class region_predict(pl.LightningModule):
    def __init__(self, input_dim = 128, hidden_dim = 128, output_dim = 6): # 输出的区域数
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_historical_steps = 50
        self.input_dim = 2
        self.num_modes = 36
        self.num_layers = 2
        self.num_heads = 8
        self.num_recurrent_steps = 3
        self.num_t2m_steps = 30
        self.redi_pl2m = 150
        self.redi_a2m = 150
        self.model_encoder = model.encoder
        self.embedding = nn.Embedding(self.num_modes, hidden_dim)# 这里需要用新的embedding；
        self.r_t2m = model.decoder.r_t2m_emb
        self.r_pl2m = model.decoder.r_pl2m_emb
        self.r_a2m = model.decoder.r_a2m_emb
        self.t2m = model.decoder.t2m_propose_attn_layers
        self.pl2m = model.decoder.pl2m_propose_attn_layers
        self.a2m = model.decoder.a2m_propose_attn_layers
        self.region = MLPLayer(input_dim, hidden_dim, output_dim)
        self.region_loss_function = nn.CrossEntropyLoss( )

    def forward(self, data: HeteroData):
        scene_enc = self.model_encoder(data)
        pos_m = data['agent']['position'][:, self.num_historical_steps - 1, :self.input_dim]
        head_m = data['agent']['heading'][:, self.num_historical_steps - 1]
        head_vector_m = torch.stack([head_m.cos(), head_m.sin()], dim=-1)
        x_t = scene_enc['x_a'].reshape(-1, self.hidden_dim)  # shape(agents x history_time_steps_, hidden_dims)
        x_pl = scene_enc['x_pl'][:, self.num_historical_steps - 1].repeat(self.num_modes, 1)
        x_a = scene_enc['x_a'][:, -1].repeat(self.num_modes, 1)
        m = self.embedding.weight.repeat(scene_enc['x_a'].size(0), 1)  # 输入到decoder的最初多模态轨迹；
        mask_src = data['agent']['valid_mask'][:, :self.num_historical_steps].contiguous()
        mask_src[:, :self.num_historical_steps - self.num_t2m_steps] = False
        mask_dst = data['agent']['predict_mask'].any(dim=-1, keepdim=True).repeat(1, self.num_modes)
        pos_t = data['agent']['position'][:, :self.num_historical_steps, :self.input_dim].reshape(-1, self.input_dim)
        head_t = data['agent']['heading'][:, :self.num_historical_steps].reshape(-1)
        edge_index_t2m = bipartite_dense_to_sparse(mask_src.unsqueeze(2) & mask_dst[:, -1:].unsqueeze(1))
        rel_pos_t2m = pos_t[edge_index_t2m[0]] - pos_m[edge_index_t2m[1]]
        rel_head_t2m = wrap_angle(head_t[edge_index_t2m[0]] - head_m[edge_index_t2m[1]])
        r_t2m = torch.stack(
            [torch.norm(rel_pos_t2m[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_t2m[1]], nbr_vector=rel_pos_t2m[:, :2]),
             rel_head_t2m,
             (edge_index_t2m[0] % self.num_historical_steps) - self.num_historical_steps + 1], dim=-1)
        r_t2m = self.r_t2m(continuous_inputs=r_t2m, categorical_embs=None)
        edge_index_t2m = bipartite_dense_to_sparse(mask_src.unsqueeze(2) & mask_dst.unsqueeze(1))
        r_t2m = r_t2m.repeat_interleave(repeats=self.num_modes, dim=0)
        pos_pl = data['map_polygon']['position'][:, :self.input_dim]
        orient_pl = data['map_polygon']['orientation']
        edge_index_pl2m = radius(
            x=pos_m[:, :2],
            y=pos_pl[:, :2],
            r=self.redi_pl2m,
            batch_x=data['agent']['batch'] if isinstance(data, Batch) else None,
            batch_y=data['map_polygon']['batch'] if isinstance(data, Batch) else None,
            max_num_neighbors=300)
        edge_index_pl2m = edge_index_pl2m[:, mask_dst[edge_index_pl2m[1], 0]]
        rel_pos_pl2m = pos_pl[edge_index_pl2m[0]] - pos_m[edge_index_pl2m[1]]
        rel_orient_pl2m = wrap_angle(orient_pl[edge_index_pl2m[0]] - head_m[edge_index_pl2m[1]])
        r_pl2m = torch.stack(
            [torch.norm(rel_pos_pl2m[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_pl2m[1]], nbr_vector=rel_pos_pl2m[:, :2]),
             rel_orient_pl2m], dim=-1)
        r_pl2m = self.r_pl2m(continuous_inputs=r_pl2m, categorical_embs=None)
        edge_index_pl2m = torch.cat([edge_index_pl2m + i * edge_index_pl2m.new_tensor(
            [[data['map_polygon']['num_nodes']], [data['agent']['num_nodes']]]) for i in range(self.num_modes)], dim=1)
        r_pl2m = r_pl2m.repeat(self.num_modes, 1)
        edge_index_a2m = radius_graph(
            x=pos_m[:, :2],
            r=self.redi_a2m,
            batch=data['agent']['batch'] if isinstance(data, Batch) else None,
            loop=False,
            max_num_neighbors=300)
        edge_index_a2m = edge_index_a2m[:, mask_src[:, -1][edge_index_a2m[0]] & mask_dst[edge_index_a2m[1], 0]]
        rel_pos_a2m = pos_m[edge_index_a2m[0]] - pos_m[edge_index_a2m[1]]
        rel_head_a2m = wrap_angle(head_m[edge_index_a2m[0]] - head_m[edge_index_a2m[1]])
        r_a2m = torch.stack(
            [torch.norm(rel_pos_a2m[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_a2m[1]], nbr_vector=rel_pos_a2m[:, :2]),
             rel_head_a2m], dim=-1)
        r_a2m = self.r_a2m(continuous_inputs=r_a2m, categorical_embs=None)
        edge_index_a2m = torch.cat(
            [edge_index_a2m + i * edge_index_a2m.new_tensor([data['agent']['num_nodes']]) for i in
             range(self.num_modes)], dim=1)
        r_a2m = r_a2m.repeat(self.num_modes, 1)
        # edge_index_m2m = dense_to_sparse(mask_dst.unsqueeze(2) & mask_dst.unsqueeze(1))[0]
        for t in range(self.num_recurrent_steps): # 循环融合proposal轨迹；
            for i in range(self.num_layers):
                m = m.reshape(-1, self.hidden_dim)
                m = self.t2m[i]((x_t, m), r_t2m, edge_index_t2m)
                m = m.reshape(-1, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
                m = self.pl2m[i]((x_pl, m), r_pl2m, edge_index_pl2m)
                m = self.a2m[i]((x_a, m), r_a2m, edge_index_a2m)
                m = m.reshape(self.num_modes, -1, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
            # m = self.m2m(m, None, edge_index_m2m)
            m = m.reshape(-1, self.num_modes, self.hidden_dim)
        # x = scene_enc['x_a'][:, -1]
        m1 = m.max(dim=1)[0] # mode维度最大池化操作；
        pred_region = self.region(m1) # shape(agents, 128)
        return pred_region.to(torch.float)

    # 函数头只能传入batch和batch_idx，其中这里的data就是dataloader传进的batch数据；
    def training_step(self,
                      data,
                      batch_idx):
        # torch.cuda.empty_cache() # 清除gpu缓存空间，防止显存溢出发生OOM；但是频繁使用比如一个step调用一次会降低效率，可以多个step或一个epoch用一次释放显存的操作；
        if isinstance(data, Batch): # Batch中的ptr()方法，返回一个指针列表，ptr 指明了当前batch中每个 图 的节点的起始索引号，也就是第一个节点元素的索引位置；
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        t = (data['agent']['category'] == 2) | (data['agent']['category'] == 3)
        pred = self(data) # shape(agents, 6)
        pred = pred[t] # shape(scored_agents, 6)
        region_loss = self.region_loss_function(pred, data['region'].squeeze(-1).to(torch.long)[t]) # data['region] shape(agents, 1) --> shape(scored_agents, )
        self.log('train_region_loss', region_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        return region_loss

    def on_train_epoch_end(self):
        torch.cuda.empty_cache()

    def validation_step(self,
                      data,
                      batch_idx):
        # torch.cuda.empty_cache() # 清除gpu缓存空间，防止显存溢出发生OOM；但是频繁使用比如一个step调用一次会降低效率，可以多个step或一个epoch用一次释放显存的操作；
        if isinstance(data, Batch): # Batch中的ptr()方法，返回一个指针列表，ptr 指明了当前batch中每个 图 的节点的起始索引号，也就是第一个节点元素的索引位置；
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        t = (data['agent']['category'] == 2) | (data['agent']['category'] == 3)
        pred = self(data) # shape(agents, 6)
        pred = pred[t] # shape(scored_agents, 6)
        region_loss = self.region_loss_function(pred, data['region'].squeeze(-1).to(torch.long)[t]) # data['region] shape(agents, 1) --> shape(scored_agents, )
        self.log('val_region_loss', region_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)

    def on_validation_epoch_end(self):
        torch.cuda.empty_cache()

    def configure_optimizers(self): # 通用的不同层优化器学习率和decay设定；
        decay = set() # 存入需要学利率衰减的层；
        no_decay = set() # 存入不需要学习率衰减的层；
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
             "weight_decay": 1e-4},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=5e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=64, eta_min=0.0)
        return [optimizer], [scheduler]

# 需要将class以外的东西放在main里面，否则其他包在import的时候会执行全部文件；
if __name__ == '__main__':
    region_module = region_predict()
    datamodule = ArgoverseV2DataModule(train_batch_size=2, val_batch_size=2, num_workers=4, pin_memory=False)
    model_checkpoint = ModelCheckpoint(dirpath="/space/renjx/checkpoints/region_predicting", monitor='val_region_loss', save_top_k=5, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(accumulate_grad_batches=2, accelerator='gpu', devices=8,
                         strategy=DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True, timeout=datetime.timedelta(days=2)),
                         callbacks=[model_checkpoint, lr_monitor], max_epochs=64)
    trainer.fit(region_module, datamodule) # 如果传入的第二个参数train_dataloader是pl的dataloader那么其他dataloader参数就不能再传啦；



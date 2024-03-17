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
from argparse import ArgumentParser

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from datasets import ArgoverseV2Dataset
from predictors import QCNet
from transforms import TargetBuilder
# 每一次调用trainer.validata或是trainer.test都需要重新装入dataloader；
if __name__ == '__main__':
    pl.seed_everything(2023, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, default='/space/renjx/qcnet_counter/epoch=63-step=399872.ckpt')
    args = parser.parse_args()

    model = {
        'QCNet': QCNet,
    }[args.model].load_from_checkpoint(checkpoint_path=args.ckpt_path, map_location='cpu') # 加载模型的device映射到cpu或是全部gpu上，否则会在第零张卡执行多个进行加载模型；
    dataset = {
        'argoverse_v2': ArgoverseV2Dataset,
    }['argoverse_v2'](root='/datasets/Argoverse2/motion/', split='train', raw_dir='/datasets/Argoverse2/motion/train/',
                      processed_dir='~/qcnet_processed_datasets/', transform=TargetBuilder(50, 60))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True,
                            persistent_workers=True)  # 相当于ArgoverseV2DataModule(**vars(args))；bs别太大，num_workers也别太大，但是也要至少保证一个gpu一个worker，也就是有多少个线程用来并行装载数据；
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices, strategy='ddp')
    trainer.validate(model, dataloader)

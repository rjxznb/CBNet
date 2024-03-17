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
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
import datetime

from datamodules import ArgoverseV2DataModule
from predictors import QCNet

if __name__ == '__main__':
    pl.seed_everything(2023, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--train_batch_size', type=int, required=True)
    parser.add_argument('--val_batch_size', type=int, required=True)
    parser.add_argument('--test_batch_size', type=int, required=True)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=bool, default=False)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--train_raw_dir', type=str, default=None)
    parser.add_argument('--val_raw_dir', type=str, default=None)
    parser.add_argument('--test_raw_dir', type=str, default=None)
    parser.add_argument('--train_processed_dir', type=str, default=None)
    parser.add_argument('--val_processed_dir', type=str, default=None)
    parser.add_argument('--test_processed_dir', type=str, default=None)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=int, default=8)
    parser.add_argument('--max_epochs', type=int, default=100)
    QCNet.add_model_specific_args(parser) # 加上在qcnet文件中定义的参数，python中的对象全是引用，也就是在同一地址，不用再次赋值CLI对象；
    args = parser.parse_args() # 将CLI参数对象转换为字典的形式；



    model = QCNet(**vars(args))
    model1 =  QCNet.load_from_checkpoint('./epoch=63-step=399872.ckpt', map_location='cpu', strict=False)
    model.encoder = model1.encoder
    # model1 = QCNet.load_from_checkpoint('/space/renjx/checkpoints/qcnet_region_ensemble/epoch=62-step=449820.ckpt', map_location='cpu')
    # model.encoder = model1.encoder
    # model.decoder = model1.decoder
    datamodule = {
        'argoverse_v2': ArgoverseV2DataModule,
    }[args.dataset](**vars(args))
    model_checkpoint = ModelCheckpoint(dirpath="/space/renjx/checkpoints/qcnet_region_ensemble", monitor='val_minFDE', save_top_k=5, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(accumulate_grad_batches=2, accelerator=args.accelerator, devices=args.devices,
                         strategy=DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True, timeout=datetime.timedelta(days=2)),
                         callbacks=[model_checkpoint, lr_monitor], max_epochs=args.max_epochs)
    trainer.fit(model, datamodule)


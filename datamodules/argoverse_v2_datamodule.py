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
from typing import Callable, Optional

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from datasets import ArgoverseV2Dataset
from transforms import TargetBuilder
from torch import nn

# 这个datamodule是继承自pl.LightningDataModule类，主要就是重写几个抽象函数，但是其dataloader和pl无关，而是用torch_geometric的dataloader；
class ArgoverseV2DataModule(pl.LightningDataModule):
    def __init__(self,
                 root: str = '/datasets/Argoverse2/motion/',
                 train_batch_size: int = 4,
                 val_batch_size: int = 4,
                 test_batch_size: int = 4,
                 shuffle: bool = True,
                 num_workers: int = 2,
                 pin_memory: bool = False,
                 persistent_workers: bool = True,
                 train_raw_dir: Optional[str] = '/datasets/Argoverse2/motion/train/',
                 val_raw_dir: Optional[str] = '/datasets/Argoverse2/motion/val/',
                 test_raw_dir: Optional[str] = '/datasets/Argoverse2/motion/test/',
                 train_processed_dir: Optional[str] = '/datasets/qcnet/train/',
                 val_processed_dir: Optional[str] = '/datasets/qcnet/val/',
                 test_processed_dir: Optional[str] = '/datasets/qcnet/test/',
                 train_transform: Optional[Callable] = TargetBuilder(50, 60),
                 val_transform: Optional[Callable] = TargetBuilder(50, 60),
                 test_transform: Optional[Callable] = None,
                 **kwargs) -> None:
        super(ArgoverseV2DataModule, self).__init__()
        self.root = root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.train_raw_dir = train_raw_dir
        self.val_raw_dir = val_raw_dir
        self.test_raw_dir = test_raw_dir
        self.train_processed_dir = train_processed_dir
        self.val_processed_dir = val_processed_dir
        self.test_processed_dir = test_processed_dir
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def prepare_data(self) -> None:
        ArgoverseV2Dataset(self.root, 'train', self.train_raw_dir, self.train_processed_dir, self.train_transform)
        ArgoverseV2Dataset(self.root, 'val', self.val_raw_dir, self.val_processed_dir, self.val_transform)
        ArgoverseV2Dataset(self.root, 'test', self.test_raw_dir, self.test_processed_dir, self.test_transform)

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = ArgoverseV2Dataset(self.root, 'train', self.train_raw_dir, self.train_processed_dir,
                                                self.train_transform)
        self.val_dataset = ArgoverseV2Dataset(self.root, 'val', self.val_raw_dir, self.val_processed_dir,
                                              self.val_transform)
        self.test_dataset = ArgoverseV2Dataset(self.root, 'test', self.test_raw_dir, self.test_processed_dir,
                                               self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)



if __name__ == '__main__':
    train_transform = TargetBuilder(50, 60)
    datasets = ArgoverseV2Dataset('/datasets/Argoverse2/motion/', 'train', '/datasets/Argoverse2/motion/train/', '/datasets/qcnet/train/', train_transform)
    loader = DataLoader(datasets, batch_size=4, shuffle=False,num_workers=1, pin_memory=False, persistent_workers=False)
    mlp = nn.Linear(110, 2)
    for batch in loader:
        print(batch)
        break



"""当batch_size=4时，所有维度都变为4个场景图的数据；
HeteroDataBatch(
  scenario_id=[4],
  city=[4],
  agent={
    num_nodes=193,
    av_index=[4],
    valid_mask=[193, 110],
    predict_mask=[193, 110],
    id=[4],
    type=[193],
    category=[193],
    position=[193, 110, 3],
    heading=[193, 110],
    velocity=[193, 110, 3],
    target=[193, 60, 4],
    batch=[193],
    ptr=[5]
  },
  map_polygon={
    num_nodes=310,
    position=[310, 3],
    orientation=[310],
    height=[310],
    type=[310],
    is_intersection=[310],
    batch=[310],
    ptr=[5]
  },
  map_point={
    num_nodes=6117,
    position=[6117, 3],
    orientation=[6117],
    magnitude=[6117],
    height=[6117],
    type=[6117],
    side=[6117],
    batch=[6117],
    ptr=[5]
  },
  (map_point, to, map_polygon)={ edge_index=[2, 6117] },
  (map_polygon, to, map_polygon)={
    edge_index=[2, 886],
    type=[886]
  }
)
"""
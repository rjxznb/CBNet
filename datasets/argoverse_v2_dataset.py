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
import os
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
from urllib import request

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.data import extract_tar
from tqdm import tqdm

from utils import safe_list_index
from utils import side_to_directed_lineseg

try:
    from av2.geometry.interpolate import compute_midpoint_line
    from av2.map.map_api import ArgoverseStaticMap
    from av2.map.map_primitives import Polyline
    from av2.utils.io import read_json_file
except ImportError:
    compute_midpoint_line = object
    ArgoverseStaticMap = object
    Polyline = object
    read_json_file = object


class ArgoverseV2Dataset(Dataset): # torch_geometric.data.Dataset
    """Dataset class for Argoverse 2 Motion Forecasting Dataset.

    See https://www.argoverse.org/av2.html for more information about the dataset.

    Args:
        root (string): the root folder of the dataset. If you've downloaded the raw .tar file, placing it in the root
            folder will skip downloading automatically.
        split (string): specify the split of the dataset: `"train"` | `"val"` | `"test"`.
        raw_dir (string, optional): optionally specify the directory of the raw data. By default, the raw directory is
            path/to/root/split/raw/. If specified, the path of the raw log is path/to/raw_dir/log_id. If all logs
            exist in the raw directory, file downloading/extraction will be skipped. (default: None)
        processed_dir (string, optional): optionally specify the directory of the processed data. By default, the
            processed directory is path/to/root/split/processed/. If specified, the path of the processed .pkl files is
            path/to/processed_dir/*.pkl. If all .pkl files exist in the processed directory, file downloading/extraction
            and data preprocessing will be skipped. (default: None)
        transform (callable, optional): a function/transform that takes in an :obj:`torch_geometric.data.Data` object
            and returns a transformed version. The data object will be transformed before every access. (default: None)
        dim (int, Optional): 2D or 3D data. (default: 3)
        num_historical_steps (int, Optional): the number of historical time steps. (default: 50)
        num_future_steps (int, Optional): the number of future time steps. (default: 60)
        predict_unseen_agents (boolean, Optional): if False, filter out agents that are unseen during the historical
            time steps. (default: False)
        vector_repr (boolean, Optional): if True, a time step t is valid only when both t and t-1 are valid.
            (default: True)
    """

    def __init__(self,
                 root: str,
                 split: str,
                 raw_dir: Optional[str] = None,
                 processed_dir: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 dim: int = 3,
                 num_historical_steps: int = 50,
                 num_future_steps: int = 60,
                 predict_unseen_agents: bool = False,
                 vector_repr: bool = True) -> None:
        root = os.path.expanduser(os.path.normpath(root))
        if not os.path.isdir(root):
            os.makedirs(root)
        if split not in ('train', 'val', 'test'):
            raise ValueError(f'{split} is not a valid split')
        self.split = split

        if raw_dir is None:
            raw_dir = os.path.join(root, split, 'raw')
            self._raw_dir = raw_dir
            if os.path.isdir(self._raw_dir):
                self._raw_file_names = [name for name in os.listdir(self._raw_dir) if
                                        os.path.isdir(os.path.join(self._raw_dir, name))]
            else:
                self._raw_file_names = []
        else:
            raw_dir = os.path.expanduser(os.path.normpath(raw_dir))
            self._raw_dir = raw_dir
            if os.path.isdir(self._raw_dir):
                self._raw_file_names = [name for name in os.listdir(self._raw_dir) if
                                        os.path.isdir(os.path.join(self._raw_dir, name))]
            else:
                self._raw_file_names = []

        if processed_dir is None:
            processed_dir = os.path.join(root, split, 'processed')
            self._processed_dir = processed_dir
            if os.path.isdir(self._processed_dir):
                self._processed_file_names = [name for name in os.listdir(self._processed_dir) if
                                              os.path.isfile(os.path.join(self._processed_dir, name)) and
                                              name.endswith(('pkl', 'pickle'))]
            else:
                self._processed_file_names = []
        else:
            processed_dir = os.path.expanduser(os.path.normpath(processed_dir))
            self._processed_dir = processed_dir
            if os.path.isdir(self._processed_dir):
                self._processed_file_names = [name for name in os.listdir(self._processed_dir) if
                                              os.path.isfile(os.path.join(self._processed_dir, name)) and
                                              name.endswith(('pkl', 'pickle'))]
            else:
                self._processed_file_names = []

        self.dim = dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_steps = num_historical_steps + num_future_steps
        self.predict_unseen_agents = predict_unseen_agents
        self.vector_repr = vector_repr
        self._url = f'https://s3.amazonaws.com/argoverse/datasets/av2/tars/motion-forecasting/{split}.tar'
        self._num_samples = {
            'train': 199908,
            'val': 24988,
            'test': 24984,
        }[split]
        self._agent_types = ['vehicle', 'pedestrian', 'motorcyclist', 'cyclist', 'bus', 'static', 'background',
                             'construction', 'riderless_bicycle', 'unknown']
        self._agent_categories = ['TRACK_FRAGMENT', 'UNSCORED_TRACK', 'SCORED_TRACK', 'FOCAL_TRACK']
        self._polygon_types = ['VEHICLE', 'BIKE', 'BUS', 'PEDESTRIAN']
        self._polygon_is_intersections = [True, False, None]
        self._point_types = ['DASH_SOLID_YELLOW', 'DASH_SOLID_WHITE', 'DASHED_WHITE', 'DASHED_YELLOW',
                             'DOUBLE_SOLID_YELLOW', 'DOUBLE_SOLID_WHITE', 'DOUBLE_DASH_YELLOW', 'DOUBLE_DASH_WHITE',
                             'SOLID_YELLOW', 'SOLID_WHITE', 'SOLID_DASH_WHITE', 'SOLID_DASH_YELLOW', 'SOLID_BLUE',
                             'NONE', 'UNKNOWN', 'CROSSWALK', 'CENTERLINE']
        self._point_sides = ['LEFT', 'RIGHT', 'CENTER']
        self._polygon_to_polygon_types = ['NONE', 'PRED', 'SUCC', 'LEFT', 'RIGHT']
        super(ArgoverseV2Dataset, self).__init__(root=root, transform=transform, pre_transform=None, pre_filter=None)

    @property
    def raw_dir(self) -> str:
        return self._raw_dir

    @property
    def processed_dir(self) -> str:
        return self._processed_dir

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    def download(self) -> None:
        if not os.path.isfile(os.path.join(self.root, f'{self.split}.tar')):
            print(f'Downloading {self._url}', file=sys.stderr)
            request.urlretrieve(self._url, os.path.join(self.root, f'{self.split}.tar'))
        if os.path.isdir(os.path.join(self.root, self.split)):
            shutil.rmtree(os.path.join(self.root, self.split))
        if os.path.isdir(self.raw_dir):
            shutil.rmtree(self.raw_dir)
        os.makedirs(self.raw_dir)
        extract_tar(path=os.path.join(self.root, f'{self.split}.tar'), folder=self.raw_dir, mode='r')
        self._raw_file_names = [name for name in os.listdir(os.path.join(self.raw_dir, self.split)) if
                                os.path.isdir(os.path.join(self.raw_dir, self.split, name))]
        for raw_file_name in self.raw_file_names:
            shutil.move(os.path.join(self.raw_dir, self.split, raw_file_name), self.raw_dir)
        os.rmdir(os.path.join(self.raw_dir, self.split))

    def process(self) -> None: # parquet文件中以df格式存储agent信息；json文件中以字典的格式存储map信息；转换后以pkl文件存储agent和map信息，读取之后是以字典的形式赋值数据；
        for raw_file_name in tqdm(self.raw_file_names):
            df = pd.read_parquet(os.path.join(self.raw_dir, raw_file_name, f'scenario_{raw_file_name}.parquet')) # 读parquet文件，也就是agent文件；
            map_dir = Path(self.raw_dir) / raw_file_name
            map_path = map_dir / sorted(map_dir.glob('log_map_archive_*.json'))[0]
            map_data = read_json_file(map_path) # 读json文件，也就是map文件
            centerlines = {lane_segment['id']: Polyline.from_json_data(lane_segment['centerline'])
                           for lane_segment in map_data['lane_segments'].values()} # 全部的车道线的中心线的polyline表示，一个polyline表示一大段车道线的一小段车道线节点的表示，由n个waypoints组成，格式如右面展示的样子： {239869250: Polyline(waypoints=[Point(x=2647.98, y=1248.95, z=0.0), Point(x=2646.52, y=1247.6, z=0.0), Point(x=2645.06, y=1246.26, z=0.0), Point(x=2643.6, y=1244.91, z=0.0),...)]
            map_api = ArgoverseStaticMap.from_json(map_path)
            data = dict()
            data['scenario_id'] = self.get_scenario_id(df)
            data['city'] = self.get_city(df)
            data['agent'] = self.get_agent_features(df)
            data.update(self.get_map_features(map_api, centerlines))
            with open(os.path.join(self.processed_dir, f'{raw_file_name}.pkl'), 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def get_scenario_id(df: pd.DataFrame) -> str:
        return df['scenario_id'].values[0]

    @staticmethod
    def get_city(df: pd.DataFrame) -> str:
        return df['city'].values[0]

    def get_agent_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        if not self.predict_unseen_agents:  # filter out agents that are unseen during the historical time steps
            historical_df = df[df['timestep'] < self.num_historical_steps]
            agent_ids = list(historical_df['track_id'].unique())
            df = df[df['track_id'].isin(agent_ids)]
        else:
            agent_ids = list(df['track_id'].unique())

        num_agents = len(agent_ids)
        av_idx = agent_ids.index('AV')

        # initialization
        # # 这里的valid_mask shape为[81, 110]，其中81表示agent的nums，110表示过去50秒和未来60s的长度，其中如果为False，则表示在那个时间步还未出现捏；如果为True，则表示出现啦；
        valid_mask = torch.zeros(num_agents, self.num_steps, dtype=torch.bool)
        current_valid_mask = torch.zeros(num_agents, dtype=torch.bool)
        # 这里的predict_mask shape为[81, 110]，形状同valid_mask但是这里的mask是用于predict时表示每一个agent的哪些时间步是可预测的；每一个行的前50个元素全为False，然后category为0的轨迹全部都为false，因为这些轨迹本身就不存在；category为2和3的后60个时间步都为true，表示需要预测；有的category为1的轨迹后60个时间步部分为true部分为false表示有的需要预测有的不需要预测，在预测的时候会乘这个矩阵，从而消除为false的时间步值；
        predict_mask = torch.zeros(num_agents, self.num_steps, dtype=torch.bool)
        agent_id: List[Optional[str]] = [None] * num_agents
        agent_type = torch.zeros(num_agents, dtype=torch.uint8)
        agent_category = torch.zeros(num_agents, dtype=torch.uint8)
        position = torch.zeros(num_agents, self.num_steps, self.dim, dtype=torch.float)
        heading = torch.zeros(num_agents, self.num_steps, dtype=torch.float)
        velocity = torch.zeros(num_agents, self.num_steps, self.dim, dtype=torch.float)

        for track_id, track_df in df.groupby('track_id'):
            agent_idx = agent_ids.index(track_id)
            agent_steps = track_df['timestep'].values

            valid_mask[agent_idx, agent_steps] = True
            current_valid_mask[agent_idx] = valid_mask[agent_idx, self.num_historical_steps - 1]
            predict_mask[agent_idx, agent_steps] = True
            if self.vector_repr:  # 是否用向量表示历史的轨迹； a time step t is valid only when both t and t-1 are valid
                valid_mask[agent_idx, 1: self.num_historical_steps] = (
                        valid_mask[agent_idx, :self.num_historical_steps - 1] &
                        valid_mask[agent_idx, 1: self.num_historical_steps])
                valid_mask[agent_idx, 0] = False
            predict_mask[agent_idx, :self.num_historical_steps] = False
            if not current_valid_mask[agent_idx]:
                predict_mask[agent_idx, self.num_historical_steps:] = False

            agent_id[agent_idx] = track_id
            agent_type[agent_idx] = self._agent_types.index(track_df['object_type'].values[0])
            agent_category[agent_idx] = track_df['object_category'].values[0]
            position[agent_idx, agent_steps, :2] = torch.from_numpy(np.stack([track_df['position_x'].values,
                                                                              track_df['position_y'].values],
                                                                             axis=-1)).float()
            heading[agent_idx, agent_steps] = torch.from_numpy(track_df['heading'].values).float()
            velocity[agent_idx, agent_steps, :2] = torch.from_numpy(np.stack([track_df['velocity_x'].values,
                                                                              track_df['velocity_y'].values],
                                                                             axis=-1)).float()

        if self.split == 'test':
            predict_mask[current_valid_mask
                         | (agent_category == 2)
                         | (agent_category == 3), self.num_historical_steps:] = True

        return {
            'num_nodes': num_agents,
            'av_index': av_idx,
            'valid_mask': valid_mask,
            'predict_mask': predict_mask,
            'id': agent_id,
            'type': agent_type,
            'category': agent_category,
            'position': position, # shape([81, 110, 3]) 81表示agent的数量，110表示时间步长度，3表示三维坐标,还未转换为agent_centric，只是原始数据；
            'heading': heading,
            'velocity': velocity,
        }

    def get_map_features(self,
                         map_api: ArgoverseStaticMap,
                         centerlines: Mapping[str, Polyline]) -> Dict[Union[str, Tuple[str, str, str]], Any]:
        lane_segment_ids = map_api.get_scenario_lane_segment_ids()
        cross_walk_ids = list(map_api.vector_pedestrian_crossings.keys())
        polygon_ids = lane_segment_ids + cross_walk_ids
        num_polygons = len(lane_segment_ids) + len(cross_walk_ids) * 2

        # initialization
        polygon_position = torch.zeros(num_polygons, self.dim, dtype=torch.float)
        polygon_orientation = torch.zeros(num_polygons, dtype=torch.float)
        polygon_height = torch.zeros(num_polygons, dtype=torch.float)
        polygon_type = torch.zeros(num_polygons, dtype=torch.uint8)
        polygon_is_intersection = torch.zeros(num_polygons, dtype=torch.uint8)
        point_position: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_orientation: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_magnitude: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_height: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_type: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_side: List[Optional[torch.Tensor]] = [None] * num_polygons

        for lane_segment in map_api.get_scenario_lane_segments():
            lane_segment_idx = polygon_ids.index(lane_segment.id)
            centerline = torch.from_numpy(centerlines[lane_segment.id].xyz).float()
            polygon_position[lane_segment_idx] = centerline[0, :self.dim]
            polygon_orientation[lane_segment_idx] = torch.atan2(centerline[1, 1] - centerline[0, 1],
                                                                centerline[1, 0] - centerline[0, 0])
            polygon_height[lane_segment_idx] = centerline[1, 2] - centerline[0, 2]
            polygon_type[lane_segment_idx] = self._polygon_types.index(lane_segment.lane_type.value)
            polygon_is_intersection[lane_segment_idx] = self._polygon_is_intersections.index(
                lane_segment.is_intersection)

            left_boundary = torch.from_numpy(lane_segment.left_lane_boundary.xyz).float()
            right_boundary = torch.from_numpy(lane_segment.right_lane_boundary.xyz).float()
            point_position[lane_segment_idx] = torch.cat([left_boundary[:-1, :self.dim],
                                                          right_boundary[:-1, :self.dim],
                                                          centerline[:-1, :self.dim]], dim=0)
            left_vectors = left_boundary[1:] - left_boundary[:-1]
            right_vectors = right_boundary[1:] - right_boundary[:-1]
            center_vectors = centerline[1:] - centerline[:-1]
            point_orientation[lane_segment_idx] = torch.cat([torch.atan2(left_vectors[:, 1], left_vectors[:, 0]),
                                                             torch.atan2(right_vectors[:, 1], right_vectors[:, 0]),
                                                             torch.atan2(center_vectors[:, 1], center_vectors[:, 0])],
                                                            dim=0)
            point_magnitude[lane_segment_idx] = torch.norm(torch.cat([left_vectors[:, :2],
                                                                      right_vectors[:, :2],
                                                                      center_vectors[:, :2]], dim=0), p=2, dim=-1)
            point_height[lane_segment_idx] = torch.cat([left_vectors[:, 2], right_vectors[:, 2], center_vectors[:, 2]],
                                                       dim=0)
            left_type = self._point_types.index(lane_segment.left_mark_type.value)
            right_type = self._point_types.index(lane_segment.right_mark_type.value)
            center_type = self._point_types.index('CENTERLINE')
            point_type[lane_segment_idx] = torch.cat(
                [torch.full((len(left_vectors),), left_type, dtype=torch.uint8),
                 torch.full((len(right_vectors),), right_type, dtype=torch.uint8),
                 torch.full((len(center_vectors),), center_type, dtype=torch.uint8)], dim=0)
            point_side[lane_segment_idx] = torch.cat(
                [torch.full((len(left_vectors),), self._point_sides.index('LEFT'), dtype=torch.uint8),
                 torch.full((len(right_vectors),), self._point_sides.index('RIGHT'), dtype=torch.uint8),
                 torch.full((len(center_vectors),), self._point_sides.index('CENTER'), dtype=torch.uint8)], dim=0)

        for crosswalk in map_api.get_scenario_ped_crossings():
            crosswalk_idx = polygon_ids.index(crosswalk.id)
            edge1 = torch.from_numpy(crosswalk.edge1.xyz).float()
            edge2 = torch.from_numpy(crosswalk.edge2.xyz).float()
            start_position = (edge1[0] + edge2[0]) / 2
            end_position = (edge1[-1] + edge2[-1]) / 2
            polygon_position[crosswalk_idx] = start_position[:self.dim]
            polygon_position[crosswalk_idx + len(cross_walk_ids)] = end_position[:self.dim]
            polygon_orientation[crosswalk_idx] = torch.atan2((end_position - start_position)[1],
                                                             (end_position - start_position)[0])
            polygon_orientation[crosswalk_idx + len(cross_walk_ids)] = torch.atan2((start_position - end_position)[1],
                                                                                   (start_position - end_position)[0])
            polygon_height[crosswalk_idx] = end_position[2] - start_position[2]
            polygon_height[crosswalk_idx + len(cross_walk_ids)] = start_position[2] - end_position[2]
            polygon_type[crosswalk_idx] = self._polygon_types.index('PEDESTRIAN')
            polygon_type[crosswalk_idx + len(cross_walk_ids)] = self._polygon_types.index('PEDESTRIAN')
            polygon_is_intersection[crosswalk_idx] = self._polygon_is_intersections.index(None)
            polygon_is_intersection[crosswalk_idx + len(cross_walk_ids)] = self._polygon_is_intersections.index(None)

            if side_to_directed_lineseg((edge1[0] + edge1[-1]) / 2, start_position, end_position) == 'LEFT':
                left_boundary = edge1
                right_boundary = edge2
            else:
                left_boundary = edge2
                right_boundary = edge1
            num_centerline_points = math.ceil(torch.norm(end_position - start_position, p=2, dim=-1).item() / 2.0) + 1
            centerline = torch.from_numpy(
                compute_midpoint_line(left_ln_boundary=left_boundary.numpy(),
                                      right_ln_boundary=right_boundary.numpy(),
                                      num_interp_pts=int(num_centerline_points))[0]).float()

            point_position[crosswalk_idx] = torch.cat([left_boundary[:-1, :self.dim],
                                                       right_boundary[:-1, :self.dim],
                                                       centerline[:-1, :self.dim]], dim=0)
            point_position[crosswalk_idx + len(cross_walk_ids)] = torch.cat(
                [right_boundary.flip(dims=[0])[:-1, :self.dim],
                 left_boundary.flip(dims=[0])[:-1, :self.dim],
                 centerline.flip(dims=[0])[:-1, :self.dim]], dim=0)
            left_vectors = left_boundary[1:] - left_boundary[:-1]
            right_vectors = right_boundary[1:] - right_boundary[:-1]
            center_vectors = centerline[1:] - centerline[:-1]
            point_orientation[crosswalk_idx] = torch.cat(
                [torch.atan2(left_vectors[:, 1], left_vectors[:, 0]),
                 torch.atan2(right_vectors[:, 1], right_vectors[:, 0]),
                 torch.atan2(center_vectors[:, 1], center_vectors[:, 0])], dim=0)
            point_orientation[crosswalk_idx + len(cross_walk_ids)] = torch.cat(
                [torch.atan2(-right_vectors.flip(dims=[0])[:, 1], -right_vectors.flip(dims=[0])[:, 0]),
                 torch.atan2(-left_vectors.flip(dims=[0])[:, 1], -left_vectors.flip(dims=[0])[:, 0]),
                 torch.atan2(-center_vectors.flip(dims=[0])[:, 1], -center_vectors.flip(dims=[0])[:, 0])], dim=0)
            point_magnitude[crosswalk_idx] = torch.norm(torch.cat([left_vectors[:, :2],
                                                                   right_vectors[:, :2],
                                                                   center_vectors[:, :2]], dim=0), p=2, dim=-1)
            point_magnitude[crosswalk_idx + len(cross_walk_ids)] = torch.norm(
                torch.cat([-right_vectors.flip(dims=[0])[:, :2],
                           -left_vectors.flip(dims=[0])[:, :2],
                           -center_vectors.flip(dims=[0])[:, :2]], dim=0), p=2, dim=-1)
            point_height[crosswalk_idx] = torch.cat([left_vectors[:, 2], right_vectors[:, 2], center_vectors[:, 2]],
                                                    dim=0)
            point_height[crosswalk_idx + len(cross_walk_ids)] = torch.cat(
                [-right_vectors.flip(dims=[0])[:, 2],
                 -left_vectors.flip(dims=[0])[:, 2],
                 -center_vectors.flip(dims=[0])[:, 2]], dim=0)
            crosswalk_type = self._point_types.index('CROSSWALK')
            center_type = self._point_types.index('CENTERLINE')
            point_type[crosswalk_idx] = torch.cat([
                torch.full((len(left_vectors),), crosswalk_type, dtype=torch.uint8),
                torch.full((len(right_vectors),), crosswalk_type, dtype=torch.uint8),
                torch.full((len(center_vectors),), center_type, dtype=torch.uint8)], dim=0)
            point_type[crosswalk_idx + len(cross_walk_ids)] = torch.cat(
                [torch.full((len(right_vectors),), crosswalk_type, dtype=torch.uint8),
                 torch.full((len(left_vectors),), crosswalk_type, dtype=torch.uint8),
                 torch.full((len(center_vectors),), center_type, dtype=torch.uint8)], dim=0)
            point_side[crosswalk_idx] = torch.cat(
                [torch.full((len(left_vectors),), self._point_sides.index('LEFT'), dtype=torch.uint8),
                 torch.full((len(right_vectors),), self._point_sides.index('RIGHT'), dtype=torch.uint8),
                 torch.full((len(center_vectors),), self._point_sides.index('CENTER'), dtype=torch.uint8)], dim=0)
            point_side[crosswalk_idx + len(cross_walk_ids)] = torch.cat(
                [torch.full((len(right_vectors),), self._point_sides.index('LEFT'), dtype=torch.uint8),
                 torch.full((len(left_vectors),), self._point_sides.index('RIGHT'), dtype=torch.uint8),
                 torch.full((len(center_vectors),), self._point_sides.index('CENTER'), dtype=torch.uint8)], dim=0)

        num_points = torch.tensor([point.size(0) for point in point_position], dtype=torch.long)
        point_to_polygon_edge_index = torch.stack(
            [torch.arange(num_points.sum(), dtype=torch.long),
             torch.arange(num_polygons, dtype=torch.long).repeat_interleave(num_points)], dim=0)
        polygon_to_polygon_edge_index = []
        polygon_to_polygon_type = []
        for lane_segment in map_api.get_scenario_lane_segments():
            lane_segment_idx = polygon_ids.index(lane_segment.id)
            pred_inds = []
            for pred in lane_segment.predecessors: # 遍历所有车道线的前驱车道线；
                pred_idx = safe_list_index(polygon_ids, pred)
                if pred_idx is not None:
                    pred_inds.append(pred_idx)
            if len(pred_inds) != 0:
                polygon_to_polygon_edge_index.append(
                    torch.stack([torch.tensor(pred_inds, dtype=torch.long),
                                 torch.full((len(pred_inds),), lane_segment_idx, dtype=torch.long)], dim=0))
                polygon_to_polygon_type.append(
                    torch.full((len(pred_inds),), self._polygon_to_polygon_types.index('PRED'), dtype=torch.uint8))
            succ_inds = []
            for succ in lane_segment.successors: # 遍历所有车道线的后继车道线；
                succ_idx = safe_list_index(polygon_ids, succ)
                if succ_idx is not None:
                    succ_inds.append(succ_idx)
            if len(succ_inds) != 0:
                polygon_to_polygon_edge_index.append(
                    torch.stack([torch.tensor(succ_inds, dtype=torch.long),
                                 torch.full((len(succ_inds),), lane_segment_idx, dtype=torch.long)], dim=0))
                polygon_to_polygon_type.append(
                    torch.full((len(succ_inds),), self._polygon_to_polygon_types.index('SUCC'), dtype=torch.uint8))
            if lane_segment.left_neighbor_id is not None:
                left_idx = safe_list_index(polygon_ids, lane_segment.left_neighbor_id)
                if left_idx is not None:
                    polygon_to_polygon_edge_index.append(
                        torch.tensor([[left_idx], [lane_segment_idx]], dtype=torch.long))
                    polygon_to_polygon_type.append(
                        torch.tensor([self._polygon_to_polygon_types.index('LEFT')], dtype=torch.uint8))
            if lane_segment.right_neighbor_id is not None:
                right_idx = safe_list_index(polygon_ids, lane_segment.right_neighbor_id)
                if right_idx is not None:
                    polygon_to_polygon_edge_index.append(
                        torch.tensor([[right_idx], [lane_segment_idx]], dtype=torch.long))
                    polygon_to_polygon_type.append(
                        torch.tensor([self._polygon_to_polygon_types.index('RIGHT')], dtype=torch.uint8))
        if len(polygon_to_polygon_edge_index) != 0:
            polygon_to_polygon_edge_index = torch.cat(polygon_to_polygon_edge_index, dim=1)
            polygon_to_polygon_type = torch.cat(polygon_to_polygon_type, dim=0)
        else:
            polygon_to_polygon_edge_index = torch.tensor([[], []], dtype=torch.long)
            polygon_to_polygon_type = torch.tensor([], dtype=torch.uint8)

        map_data = {
            'map_polygon': {},
            'map_point': {},
            ('map_point', 'to', 'map_polygon'): {},
            ('map_polygon', 'to', 'map_polygon'): {},
        }
        map_data['map_polygon']['num_nodes'] = num_polygons # 场景中用中心线表示的map_polyline级别的车道线数量；
        map_data['map_polygon']['position'] = polygon_position # shape(112, 3) 全部polygon的坐标；
        map_data['map_polygon']['orientation'] = polygon_orientation
        if self.dim == 3:
            map_data['map_polygon']['height'] = polygon_height
        map_data['map_polygon']['type'] = polygon_type
        map_data['map_polygon']['is_intersection'] = polygon_is_intersection
        if len(num_points) == 0:
            map_data['map_point']['num_nodes'] = 0
            map_data['map_point']['position'] = torch.tensor([], dtype=torch.float)
            map_data['map_point']['orientation'] = torch.tensor([], dtype=torch.float)
            map_data['map_point']['magnitude'] = torch.tensor([], dtype=torch.float)
            if self.dim == 3:
                map_data['map_point']['height'] = torch.tensor([], dtype=torch.float)
            map_data['map_point']['type'] = torch.tensor([], dtype=torch.uint8)
            map_data['map_point']['side'] = torch.tensor([], dtype=torch.uint8)
        else: # 这就是waypoint级别的车道线中的所有节点表示；suo
            map_data['map_point']['num_nodes'] = num_points.sum().item()
            map_data['map_point']['position'] = torch.cat(point_position, dim=0)
            map_data['map_point']['orientation'] = torch.cat(point_orientation, dim=0)
            map_data['map_point']['magnitude'] = torch.cat(point_magnitude, dim=0)
            if self.dim == 3:
                map_data['map_point']['height'] = torch.cat(point_height, dim=0)
            map_data['map_point']['type'] = torch.cat(point_type, dim=0)
            map_data['map_point']['side'] = torch.cat(point_side, dim=0)
        map_data['map_point', 'to', 'map_polygon']['edge_index'] = point_to_polygon_edge_index
        map_data['map_polygon', 'to', 'map_polygon']['edge_index'] = polygon_to_polygon_edge_index # 第一个tensor是waypoints，第二个tensor是polygon,一个polygon对应多个waypoints；
        map_data['map_polygon', 'to', 'map_polygon']['type'] = polygon_to_polygon_type

        return map_data

    def len(self) -> int:
        return self._num_samples

    def get(self, idx: int) -> HeteroData: # 这个函数就是dataloader收集dataset的函数，也就是get_item的等价函数；
        with open(self.processed_paths[idx], 'rb') as handle: # 这里是从外存加载到内存而不是直接从内存读，可以先通过一个列表或是啥的把所有processed.pkl文件的data全读进来
            return HeteroData(pickle.load(handle)) # 把data字典作为HeteroData异质图类型的对象,并且后来用的DataLoader类也是torch_geometric.data包中的类函数；

    def _download(self) -> None:
        # if complete raw/processed files exist, skip downloading
        if ((os.path.isdir(self.raw_dir) and len(self.raw_file_names) == len(self)) or
                (os.path.isdir(self.processed_dir) and len(self.processed_file_names) == len(self))):
            return
        self._processed_file_names = []
        self.download()

    def _process(self) -> None: # 在这里将文件读入内存呢呦；
        # if complete processed files exist, skip processing
        if os.path.isdir(self.processed_dir) and len(self.processed_file_names) == len(self):
            return
        print('Processing...', file=sys.stderr)
        if os.path.isdir(self.processed_dir):
            for name in os.listdir(self.processed_dir):
                if name.endswith(('pkl', 'pickle')):
                    os.remove(os.path.join(self.processed_dir, name))
        else:
            os.makedirs(self.processed_dir)
        self._processed_file_names = [f'{raw_file_name}.pkl' for raw_file_name in self.raw_file_names]
        self.process()
        print('Done!', file=sys.stderr)

'''
map_polygon to map_polygon的edge_index以如下形式存储：其中第二个tensor是按照从前到后的顺序给出id的，一个polygon和其他的polygon最多有四种关系，就是前驱后继左邻居右邻居这四种，如果有的不到四个那么就证明其不存在那种关系的polygon，而且有的polygon可能会一个关系都没有；
tensor([46, 50,  4, 91, 56,  3,  9,  7, 86,  1, 25, 47, 37,  0, 20,  8, 10, 32,
        28, 33, 17, 13,  3, 56,  5, 16, 42, 53, 15, 14,  2, 19, 36, 15, 17,  5,
        28, 24, 15, 42, 37, 31, 43, 49, 25, 30, 31, 21, 53, 39,  7, 40, 32, 18,
        29,  8, 49, 27, 11, 10,  8, 55,  8, 25, 49, 18, 10, 41, 20,  6, 14, 46,
        16,  9, 32, 20, 36,  5, 17, 19, 24, 13, 23, 40, 25, 31, 40, 35, 21, 87,
        11, 21, 26, 48, 52, 35, 16, 22, 52,  4, 12, 46, 24, 55, 40, 15, 53, 55,
         6, 10, 14, 35, 34, 39, 51, 12, 12, 13, 22, 19, 14,  5,  6, 41, 44, 30,
        37, 23, 29, 42, 43, 76, 24, 20, 41,  9, 38, 12,  4, 34, 37, 47, 44, 30,
        13, 56, 22, 23, 27, 45, 54, 13, 17, 36, 33,  8, 35, 11, 12, 35, 52, 34,
        38, 40, 55, 54, 48, 18,  0, 25,  4, 38, 50, 24, 46, 12, 15, 16, 51,  0,
        47, 30, 55, 49, 24, 25, 43,  8, 13, 27, 40, 46, 51, 26, 45, 28, 15,  1,
        40,  7, 59, 60, 74, 64, 69, 65, 57, 61, 71, 66, 65, 57, 72, 64, 59, 69,
        68, 69, 73, 66, 71, 58, 67, 74, 60, 70, 60, 58, 68, 63, 74, 59, 64, 72,
        70, 66, 61, 61, 62, 58, 75, 70, 92, 65, 68, 73, 69, 59, 63, 60, 67, 70,
        62, 64, 66, 57, 69, 79, 92, 35, 78, 89, 87, 84, 89, 92, 79, 76, 85, 75,
        82, 88, 77, 81, 87, 80, 84, 97, 79, 87, 89, 86, 96, 94, 81, 77, 88, 97,
        78,  3, 83, 91, 82, 80, 24, 76, 79, 85, 76, 77, 82, 95, 93, 94,  1, 86,
        77, 70, 75, 96, 90, 95, 91, 83, 90, 94, 96, 83, 93, 95, 81, 85])
tensor([ 0,  0,  0,  1,  1,  1,  2,  3,  3,  3,  4,  4,  4,  4,  5,  5,  5,  5,
         6,  6,  6,  7,  7,  7,  8,  8,  8,  8,  8,  8,  9,  9,  9, 10, 10, 10,
        10, 11, 11, 11, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 14, 14,
        14, 14, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18,
        18, 19, 19, 19, 20, 20, 20, 20, 21, 21, 21, 22, 22, 22, 23, 23, 23, 24,
        24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 26, 26, 27, 27, 27, 28,
        28, 28, 29, 29, 30, 30, 30, 30, 31, 31, 31, 32, 32, 32, 33, 33, 34, 34,
        34, 35, 35, 35, 35, 35, 35, 36, 36, 36, 37, 37, 37, 37, 38, 38, 38, 39,
        39, 40, 40, 40, 40, 40, 40, 40, 41, 41, 41, 42, 42, 42, 43, 43, 43, 44,
        44, 45, 45, 46, 46, 46, 46, 46, 47, 47, 47, 48, 48, 49, 49, 49, 49, 50,
        50, 51, 51, 51, 52, 52, 52, 53, 53, 53, 54, 54, 55, 55, 55, 55, 55, 56,
        56, 56, 57, 57, 57, 58, 58, 58, 59, 59, 59, 59, 60, 60, 60, 60, 61, 61,
        61, 62, 62, 63, 63, 64, 64, 64, 64, 65, 65, 65, 66, 66, 66, 66, 67, 67,
        68, 68, 68, 69, 69, 69, 69, 69, 70, 70, 70, 70, 70, 71, 71, 72, 72, 73,
        73, 74, 74, 74, 75, 75, 75, 76, 76, 76, 76, 77, 77, 77, 77, 78, 78, 79,
        79, 79, 79, 80, 80, 81, 81, 81, 82, 82, 82, 83, 83, 83, 84, 84, 85, 85,
        85, 86, 86, 86, 87, 87, 87, 87, 88, 88, 89, 89, 89, 90, 90, 91, 91, 91,
        92, 92, 92, 93, 93, 94, 94, 94, 95, 95, 95, 96, 96, 96, 97, 97])
map_polygon to map_polygon的type以如下形式存储：self._polygon_to_polygon_types = ['NONE', 'PRED', 'SUCC', 'LEFT', 'RIGHT'] 关系都按照枚举值给出滴呦；
tensor([1, 2, 3, 1, 2, 3, 2, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3,
        1, 2, 2, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 2, 2, 3, 4, 1, 1,
        1, 1, 2, 3, 1, 2, 2, 3, 1, 1, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2,
        3, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 2, 2, 2, 2, 3,
        1, 1, 1, 2, 3, 4, 1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 1, 2, 2, 3, 1, 2, 3, 1,
        2, 3, 1, 3, 1, 2, 3, 1, 1, 1, 1, 2, 3, 2, 3, 4, 1, 2, 3, 4, 2, 3, 4, 1,
        2, 1, 2, 2, 2, 2, 2, 3, 1, 3, 4, 1, 2, 3, 1, 2, 3, 2, 3, 1, 2, 1, 1, 1,
        2, 3, 1, 3, 4, 1, 2, 1, 2, 3, 4, 1, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2,
        1, 1, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 2, 2, 3, 1, 1, 1, 3, 1, 2,
        3, 2, 3, 2, 3, 2, 2, 2, 3, 1, 2, 3, 1, 1, 1, 3, 1, 3, 1, 2, 3, 1, 1, 1,
        2, 3, 1, 2, 2, 2, 3, 1, 3, 2, 3, 1, 3, 1, 2, 3, 1, 2, 3, 1, 2, 2, 3, 1,
        1, 2, 3, 1, 2, 1, 2, 2, 3, 1, 2, 2, 2, 2, 1, 2, 3, 1, 2, 3, 1, 2, 1, 1,
        1, 1, 2, 3, 1, 1, 2, 3, 1, 2, 1, 2, 3, 2, 3, 1, 2, 3, 1, 2, 3, 1, 3, 1,
        2, 3, 1, 2, 3, 1, 2, 3, 1, 2], dtype=torch.uint8)
'''
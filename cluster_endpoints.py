import torch
import pickle
from tqdm import tqdm
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import joblib

num_historical_steps = 50
num_future_steps = 60
# 区域的数量：
k = 6
# 这里是处理val数据集，如果是train数据集还需要改回来；
process_file_list = [name for name in os.listdir('/datasets/qcnet/train/')] # get所有的预处理之后的文件；
all_files_endpoints = []

for process_file_name in tqdm(process_file_list):
    data = dict()
    # 打开文件并读取对象
    process_file_name = os.path.join('/datasets/qcnet/train/', process_file_name)
    with open(process_file_name, 'rb') as file:
        # 使用pickle.load()方法加载对象
        data = pickle.load(file)
    origin = data['agent']['position'][:, num_historical_steps - 1]
    theta = data['agent']['heading'][:, num_historical_steps - 1]
    cos, sin = theta.cos(), theta.sin()
    rot_mat = theta.new_zeros(data['agent']['num_nodes'], 2, 2)
    rot_mat[:, 0, 0] = cos
    rot_mat[:, 0, 1] = -sin
    rot_mat[:, 1, 0] = sin
    rot_mat[:, 1, 1] = cos
    gt_trajectory = origin.new_zeros(data['agent']['num_nodes'], num_future_steps, 4)
    gt_trajectory[..., :2] = torch.bmm(data['agent']['position'][:, num_historical_steps:, :2] -
                                                 origin[:, :2].unsqueeze(1), rot_mat)
    # 筛选有效轨迹，也就最后还在场景内的轨迹；在通过切片来过滤的时候，只能通过切片内放入在筛选的维度等大小的bool类型数据来筛选，过滤出为true的那些值来；
    gt_trajectory = gt_trajectory[(data['agent']['category'] == 2) | (data['agent']['category'] == 3)]
    one_file_endpoints = gt_trajectory[:, -1, :2]  # 只用来存储当前文件能够预测的也就是track类型为'scored'或'focal'的agent最后一个时间步的坐标；
    all_files_endpoints.append(one_file_endpoints) # 是一个list，每一个元素都是一个tensor -> shape(agents, endpoints:2)

all_files_endpoints = torch.cat(all_files_endpoints)
data_array = all_files_endpoints.numpy()
# kmeans_model = KMeans(n_clusters=k, random_state=2024)
# kmeans_model.fit(data_array)
kmeans_model = joblib.load('/space/renjx/qcnet_counter/kmeans_model.pkl')
labels = kmeans_model.predict(data_array)

# 当前场景最终点在列表中的索引位置；
end_points_index = 0
# 存入所有agent的endpoints的region_label，如果轨迹类型为0或1，那么就定义其最后点所在区域为-1；
for process_file_name in tqdm(process_file_list):
    data = dict()
    new_processed_file_name = process_file_name
    # 打开文件并读取对象
    process_file_name = os.path.join('/datasets/qcnet/train/', process_file_name)
    with open(process_file_name, 'rb') as file:
        # 使用pickle.load()方法加载对象
        data = pickle.load(file)
    region = torch.full((data['agent']['num_nodes'], 1), -1, dtype=torch.int32) # 这里应该变为torch.long类型；
    scored_track = region[(data['agent']['category'] == 2) | (data['agent']['category'] == 3)]
    t = torch.tensor(labels[end_points_index:end_points_index+scored_track.size(0)], dtype=torch.int32).view(-1, 1) # 这里应该变为torch.long类型；
    region[(data['agent']['category'] == 2) | (data['agent']['category'] == 3)] = t
    data['region'] = region.to(torch.long) # shape(agents, 1)
    end_points_index += scored_track.size(0)
    new_processed_file_name = os.path.join('/datasets/qcnet/train', new_processed_file_name)
    with open(new_processed_file_name, 'wb') as file:
        pickle.dump(data, file)


plt.scatter(data_array[:, 1], data_array[:, 0], c=labels, cmap='viridis', alpha=0.7)
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.savefig('clu_val.png')

# 保存KMeans模型
# joblib.dump(kmeans_model, './kmeans_model.pkl')
# 加载KMeans模型
# loaded_kmeans_model = joblib.load('kmeans_model.pkl')

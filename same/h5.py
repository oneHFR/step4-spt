
# import h5py
# import torch
# import matplotlib.pyplot as plt
# import numpy as np

# # 加载 .h5 文件
# h5_file = "v0_scene0011_00.h5"  # 替换为你的 .h5 文件路径
# with h5py.File(h5_file, "r") as f:
#     # 提取 partition_0 的数据
#     partition_0 = f["partition_0"]
#     super_index_0 = torch.tensor(partition_0["super_index"][:])  # 第 0 层的 super_index
#     sub_pointers_0 = torch.tensor(partition_0["_cluster_"]["sub"]["pointers"][:])  # 第 0 层的 sub 指针
#     sub_points_0 = torch.tensor(partition_0["_cluster_"]["sub"]["points"][:])  # 第 0 层的 sub 点
#     pos_0 = torch.tensor(partition_0["pos"][:])  # 第 0 层的点坐标

#     # 提取 partition_1 的数据
#     partition_1 = f["partition_1"]
#     super_index_1 = torch.tensor(partition_1["super_index"][:])  # 第 1 层的 super_index
#     sub_pointers_1 = torch.tensor(partition_1["_cluster_"]["sub"]["pointers"][:])  # 第 1 层的 sub 指针
#     sub_points_1 = torch.tensor(partition_1["_cluster_"]["sub"]["points"][:])  # 第 1 层的 sub 点

# # 初始化聚类标签
# # 第 0 层的 super_index 是每个点的初始聚类标签
# cluster_labels = super_index_0.clone().long()  # 将 cluster_labels 转换为 long 类型（torch.int64）

# # 从最高层（partition_1）向下逐层推导
# # 第 1 层的 super_index 映射到第 0 层
# cluster_labels = super_index_1[cluster_labels]

# # 最终的 cluster_labels 是一个形状为 (N,) 的张量
# # 其中 N 是点云的点数，每个元素表示该点所属的聚类标签
# print(cluster_labels.shape)  # 应该输出 (N,)

# # 保存结果到 .pt 文件
# output_file = "cluster_labels_with_pos.pt"  # 输出文件名
# output_data = {
#     "cluster_labels": cluster_labels,  # 聚类标签
#     "pos": pos_0  # 点坐标
# }
# torch.save(output_data, output_file)
# print(f"聚类标签和点坐标已保存到 {output_file}")



# # 加载 .pt 文件
# output_file = "cluster_labels_with_pos.pt"
# data = torch.load(output_file)

# # 获取聚类标签和点坐标
# cluster_labels = data["cluster_labels"]
# pos = data["pos"]

# print("聚类标签:", cluster_labels)
# print("点坐标:", pos)


# # 获取唯一的聚类标签
# unique_labels = torch.unique(cluster_labels)

# # 设置一个色彩映射，用不同颜色表示不同的聚类标签
# colors = plt.cm.get_cmap('tab20', len(unique_labels))  # 选择一个合适的颜色映射

# # 创建一个绘图
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111)

# # 遍历每一个聚类标签，绘制对应颜色的点
# for i, label in enumerate(unique_labels):
#     # 找到属于当前聚类标签的点
#     indices = cluster_labels == label
#     cluster_points = pos[indices].numpy()  # 获取属于该聚类的点
#     ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
#                color=colors(i), label=f"Cluster {label.item()}", s=10)  # 使用该聚类的颜色

# # 设置图表标题和标签
# ax.set_title("Point Cloud with Cluster Labels", fontsize=16)
# ax.set_xlabel("X Coordinate", fontsize=14)
# ax.set_ylabel("Y Coordinate", fontsize=14)

# # 显示图例
# ax.legend(loc='best', fontsize=10)

# # 保存图片
# output_image = "cluster_labels_plot.png"
# plt.savefig(output_image, dpi=300)  # 保存为高分辨率图片
# print(f"图形已保存为 {output_image}")

# # 关闭绘图以释放内存
# plt.close()


import h5py
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 加载 .h5 文件
h5_file = "v0_scene0011_00.h5"  # 替换为你的 .h5 文件路径
with h5py.File(h5_file, "r") as f:
    # 提取 partition_0 的数据
    partition_0 = f["partition_0"]
    super_index_0 = torch.tensor(partition_0["super_index"][:])  # 第 0 层的 super_index
    sub_pointers_0 = torch.tensor(partition_0["_cluster_"]["sub"]["pointers"][:])  # 第 0 层的 sub 指针
    sub_points_0 = torch.tensor(partition_0["_cluster_"]["sub"]["points"][:])  # 第 0 层的 sub 点
    pos_0 = torch.tensor(partition_0["pos"][:])  # 第 0 层的点坐标

    # 提取 partition_1 的数据
    partition_1 = f["partition_1"]
    super_index_1 = torch.tensor(partition_1["super_index"][:])  # 第 1 层的 super_index
    sub_pointers_1 = torch.tensor(partition_1["_cluster_"]["sub"]["pointers"][:])  # 第 1 层的 sub 指针
    sub_points_1 = torch.tensor(partition_1["_cluster_"]["sub"]["points"][:])  # 第 1 层的 sub 点

# 初始化聚类标签
# 第 0 层的 super_index 是每个点的初始聚类标签
cluster_labels = super_index_0.clone().long()  # 将 cluster_labels 转换为 long 类型（torch.int64）

# 从最高层（partition_1）向下逐层推导
# 第 1 层的 super_index 映射到第 0 层
cluster_labels = super_index_1[cluster_labels]

# 最终的 cluster_labels 是一个形状为 (N,) 的张量
# 其中 N 是点云的点数，每个元素表示该点所属的聚类标签
print(cluster_labels.shape)  # 应该输出 (N,)

# 保存结果到 .pt 文件
output_file = "cluster_labels_with_pos.pt"  # 输出文件名
output_data = {
    "cluster_labels": cluster_labels,  # 聚类标签
    "pos": pos_0  # 点坐标
}
torch.save(output_data, output_file)
print(f"聚类标签和点坐标已保存到 {output_file}")
























# # 加载 .pt 文件
# output_file = "cluster_labels_with_pos.pt"
# data = torch.load(output_file)

# # 获取聚类标签和点坐标
# cluster_labels = data["cluster_labels"]
# pos = data["pos"]

# # 获取唯一的聚类标签
# unique_labels = torch.unique(cluster_labels)

# # 设置一个色彩映射，用不同颜色表示不同的聚类标签
# colors = plt.cm.get_cmap('tab20', len(unique_labels))  # 选择一个合适的颜色映射

# # 创建一个绘图
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111)

# # 遍历每一个聚类标签，绘制对应颜色的点
# for i, label in enumerate(unique_labels):
#     # 找到属于当前聚类标签的点
#     indices = cluster_labels == label
#     cluster_points = pos[indices].numpy()  # 获取属于该聚类的点
#     ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
#                color=colors(i), label=f"Cluster {label.item()}", s=10)  # 使用该聚类的颜色

# # 设置图表标题和标签
# ax.set_title("Point Cloud with Cluster Labels", fontsize=16)
# ax.set_xlabel("X Coordinate", fontsize=14)
# ax.set_ylabel("Y Coordinate", fontsize=14)

# # 显示图例
# ax.legend(loc='best', fontsize=10)

# # 保存从不同角度的截图
# for angle in range(0, 360, 45):  # 旋转角度，创建不同的视角
#     ax.view_init(azim=angle)  # 设置不同的 azimuthal 角度
#     output_image = f"cluster_labels_plot_{angle}.png"  # 根据角度生成不同的文件名
#     plt.savefig(output_image, dpi=300)  # 保存为高分辨率图片
#     print(f"图形已保存为 {output_image}")

# # 关闭绘图以释放内存
# plt.close()

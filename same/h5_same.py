import h5py
import torch
import numpy as np
from plyfile import PlyData, PlyElement  # 用于读写 .ply 文件
import random

import os
import debugpy
try:
    # 设置调试监听端口为 9501
    # 放在import之后
    debugpy.listen(("localhost", 9502))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()  # 等待调试器连接
except Exception as e:
    pass

# 加载原始 .ply 文件
def load_ply_file(ply_path):
    ply_data = PlyData.read(ply_path)
    vertices = ply_data['vertex'].data  # 获取顶点数据
    faces = ply_data['face'].data if 'face' in ply_data else None  # 获取面数据（如果有）
    return vertices, faces

# 加载 .h5 文件并计算聚类标签
def load_h5_and_compute_cluster_labels(h5_path):
    with h5py.File(h5_path, "r") as f:
        # 提取 partition_0 的数据
        partition_0 = f["partition_0"]
        super_index_0 = torch.tensor(partition_0["super_index"][:])  # 第 0 层的 super_index
        pos_0 = torch.tensor(partition_0["pos"][:])  # 第 0 层的点坐标（体素中心点）

        # 提取 partition_1 的数据
        partition_1 = f["partition_1"]
        super_index_1 = torch.tensor(partition_1["super_index"][:])  # 第 1 层的 super_index

    # 初始化聚类标签
    cluster_labels = super_index_0.clone().long()

    # 从最高层（partition_1）向下逐层推导
    cluster_labels = super_index_1[cluster_labels]

    return cluster_labels.numpy()  # 转换为 numpy 数组

# 为每个聚类分配随机颜色
def generate_cluster_colors(num_clusters):
    colors = []
    for _ in range(num_clusters):
        color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]  # 随机 RGB 颜色
        colors.append(color)
    return np.array(colors, dtype=np.uint8)

# 将体素化后的聚类标签映射到原始点
def map_cluster_labels_to_original_points(cluster_labels, voxel_to_points, num_original_points):
    original_cluster_labels = np.zeros(num_original_points, dtype=np.int64)
    for voxel_idx, point_indices in voxel_to_points.items():
        original_cluster_labels[point_indices] = cluster_labels[voxel_idx]
    return original_cluster_labels

# 修改原始点的颜色
def modify_point_colors_by_cluster(vertices, cluster_labels, cluster_colors):
    # 检查是否有颜色字段
    if 'red' in vertices.dtype.names and 'green' in vertices.dtype.names and 'blue' in vertices.dtype.names:
        for i in range(len(cluster_labels)):
            cluster_id = cluster_labels[i]
            vertices['red'][i] = cluster_colors[cluster_id][0]
            vertices['green'][i] = cluster_colors[cluster_id][1]
            vertices['blue'][i] = cluster_colors[cluster_id][2]
    else:
        # 如果没有颜色字段，添加颜色字段
        colors = np.zeros(len(vertices), dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        for i in range(len(cluster_labels)):
            cluster_id = cluster_labels[i]
            colors['red'][i] = cluster_colors[cluster_id][0]
            colors['green'][i] = cluster_colors[cluster_id][1]
            colors['blue'][i] = cluster_colors[cluster_id][2]
        vertices = np.lib.recfunctions.append_fields(vertices, names=['red', 'green', 'blue'], data=[colors['red'], colors['green'], colors['blue']], usemask=False)
    return vertices

# 保存修改后的 .ply 文件
def save_ply_file(output_path, vertices, faces):
    # 创建顶点 PlyElement
    vertex_element = PlyElement.describe(vertices, 'vertex')
    
    # 创建面 PlyElement（如果有）
    face_elements = []
    if faces is not None:
        face_element = PlyElement.describe(faces, 'face')
        face_elements.append(face_element)
    
    # 保存 .ply 文件
    PlyData([vertex_element] + face_elements, text=False).write(output_path)

# 主函数
def main():
    # 文件路径
    data = torch.load('superpoint.pt')
    ply_path = "scene0011_00_vh_clean_2.ply"  # 原始点云的 .ply 文件路径
    h5_path = "v0_scene0011_00.h5"  # 体素化后的 .h5 文件路径
    voxel_to_points_data = torch.load("scene0011_00_same.pt")  # 加载体素到原始点的映射关系

    # 加载原始点云
    output_ply_path = 'spt-' + ply_path # 修改后的 .ply 文件保存路径

    # 加载原始点云和网格数据
    vertices, faces = load_ply_file(ply_path)
    num_original_points = len(vertices)

    # 加载 .h5 文件并计算聚类标签
    cluster_labels = load_h5_and_compute_cluster_labels(h5_path)
    voxel_to_points = voxel_to_points_data["voxel_to_points"]

    # 将体素化后的聚类标签映射到原始点
    original_cluster_labels = map_cluster_labels_to_original_points(cluster_labels, voxel_to_points, num_original_points)

    # 为每个聚类分配随机颜色
    num_clusters = int(original_cluster_labels.max()) + 1  # 聚类数量
    cluster_colors = generate_cluster_colors(num_clusters)

    # 修改原始点的颜色
    modified_vertices = modify_point_colors_by_cluster(vertices, original_cluster_labels, cluster_colors)

    # 保存修改后的 .ply 文件
    save_ply_file(output_ply_path, modified_vertices, faces)
    print(f"修改后的点云和网格已保存到 {output_ply_path}")

if __name__ == "__main__":
    main()
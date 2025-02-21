import open3d as o3d
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

def modify_ply_colors(point_cloud_path, spp_path):
    """
    直接修改 `.ply` 文件的点颜色，而不改变结构或创建新文件
    """
    # 读取点云
    pcd = o3d.io.read_point_cloud(point_cloud_path)

    # 获取点的坐标
    points = np.asarray(pcd.points)

    # 读取 `superpoint.pt` 数据
    spp = torch.load(spp_path)
    
    # 确保 `superpoint.pt` 的数据在 CPU 上
    spp['coordinates'] = spp['coordinates'].cpu().numpy()
    spp['partition_labels'] = spp['partition_labels'].cpu().numpy()

    # 如果 `.ply` 文件没有颜色信息，初始化为白色
    if not pcd.has_colors():
        colors = np.ones((len(points), 3))  # 全部初始化为白色 (1,1,1)
    else:
        colors = np.asarray(pcd.colors)  # 读取现有颜色

    # 如果 `.ply` 文件的点数与 `superpoint.pt` 不同，使用最近邻匹配
    if len(spp['coordinates']) != len(points):
        print("Point counts mismatch, performing coordinate-based matching...")

        # 使用 `sklearn` 进行最近邻匹配
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(spp['coordinates'])
        distances, indices = nbrs.kneighbors(points)

        # 获取 `superpoint.pt` 对应的 `partition_labels`
        matched_labels = spp['partition_labels'][indices.flatten()]
    else:
        matched_labels = spp['partition_labels']

    # 生成颜色映射
    unique_labels = np.unique(matched_labels)
    color_map = np.random.rand(len(unique_labels), 3)  # 生成 RGB 颜色

    # 只修改匹配到的点的颜色
    for i, label in enumerate(unique_labels):
        mask = matched_labels == label
        colors[mask] = color_map[i]

    # 更新点云颜色
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # **直接覆盖原 `.ply` 文件**
    o3d.io.write_point_cloud(point_cloud_path, pcd)

    print(f"Updated colors in {point_cloud_path} successfully!")

# **测试**
scene_id = 'scene0011_00'
point_cloud_path = f'scene0011_00_vh_clean_2_labels.ply'  # 原始 .ply 文件
spp_path = f'superpoint.pt'  # superpoint 结果

modify_ply_colors(point_cloud_path, spp_path)

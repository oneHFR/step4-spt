import torch
import numpy as np
import plotly.graph_objects as go
from plyfile import PlyData

# 1. 加载 .ply 文件，提取坐标
def load_ply(ply_file):
    ply_data = PlyData.read(ply_file)
    vertices = ply_data['vertex'].data
    points = np.array([vertices['x'], vertices['y'], vertices['z']]).T
    return points

# 2. 加载分类结果的掩码 (pt 文件)，它包含每个点的类别
def load_mask(mask_file):
    mask = torch.load(mask_file)  # 假设 mask 是保存为 .pt 格式的张量
    return mask.numpy()  # 转换为 numpy 数组

def load_mask2(mask_file):
    mask_data = torch.load(mask_file)  # 加载保存为 .pt 格式的字典
    mask = mask_data['mask']  # 提取字典中的 'mask' 部分
    return mask.numpy()  # 转换为 numpy 数组

# 3. 可视化点云
def visualize_point_cloud(points, mask1, mask2, output_file='comp_2005.html'):
    # 创建 Plotly 3D 散点图
    fig = go.Figure()
    point_size = 3
    # 找到只在 mask1 中的点（不与 mask2 重叠）
    mask1_unique = np.logical_and(mask1 == 1, mask2 != 1)  # 只在 mask1 中的点
    points_of_mask1_unique = points[mask1_unique]  # 获取这些点
    color_mask1 = 'rgb(255, 0, 0)'  # 例如红色
    fig.add_trace(go.Scatter3d(
        x=points_of_mask1_unique[:, 0],
        y=points_of_mask1_unique[:, 1],
        z=points_of_mask1_unique[:, 2],
        mode='markers',
        marker=dict(size=point_size, color=color_mask1, opacity=0.7),
        name=f'Mask 1 (Unique)'  # 标记 mask1 中唯一的点
    ))

    # 找到只在 mask2 中的点（不与 mask1 重叠）
    mask2_unique = np.logical_and(mask2 == 1, mask1 != 1)  # 只在 mask2 中的点
    points_of_mask2_unique = points[mask2_unique]  # 获取这些点
    color_mask2 = 'rgb(0, 0, 255)'  # 例如蓝色
    fig.add_trace(go.Scatter3d(
        x=points_of_mask2_unique[:, 0],
        y=points_of_mask2_unique[:, 1],
        z=points_of_mask2_unique[:, 2],
        mode='markers',
        marker=dict(size=point_size, color=color_mask2, opacity=0.7),
        name=f'Mask 2 (Unique)'  # 标记 mask2 中唯一的点
    ))


    # 设置图表布局
    fig.update_layout(
        title='3D Point Cloud Visualization by Object',
        scene=dict(
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate',
            zaxis_title='Z Coordinate'
        ),
        showlegend=True,
        paper_bgcolor='rgb(20, 24, 35)',  # 设置背景色
        font=dict(color='white'),  # 设置字体颜色
        margin=dict(l=0, r=0, t=30, b=0)  # 调整边距
    )

    # 保存为 HTML 文件
    fig.write_html(output_file)
    print(f"可视化已保存为 {output_file}")

# 示例使用
ply_file = 'scene0011_00/scene0011_00_vh_clean_2.ply'  # 替换为你的 .ply 文件路径
mask_file1 = 'output_masks/3.pt'  # 替换为你的第一个 .pt 掩码文件路径
mask_file2 = 'gt_output/2005_chair_0.pt'  # 替换为你的第二个 .pt 掩码文件路径

# 加载数据
points = load_ply(ply_file)
mask1 = load_mask(mask_file1)
mask2 = load_mask2(mask_file2)

# 可视化点云
visualize_point_cloud(points, mask1, mask2)


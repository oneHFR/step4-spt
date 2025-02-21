import torch
import open3d as o3d
import numpy as np

scene_id = 'scene0011_00'
point_cloud_path = f'{scene_id}_vh_clean_2.ply'
spp_path = f'{scene_id}.pth'
# spp_path = f'{scene_id}_superpoints.pt'
output_path = f'{scene_id}_b.ply'

# def convert_spp_to_labeled_ply(point_cloud_path, spp_path, output_path):
#     pcd = o3d.io.read_point_cloud(point_cloud_path)
#     points = np.asarray(pcd.points)
    
#     # 读取superpoint标签
#     spp = torch.load(spp_path)
#     if isinstance(spp, torch.Tensor):
#         spp = spp.cpu().numpy()
    
#     # 生成颜色映射
#     unique_labels = np.unique(spp)
#     color_map = np.random.rand(len(unique_labels), 3) * 255
    
#     # 生成颜色
#     colors = np.zeros((len(points), 3), dtype=np.uint8)
#     for i, label in enumerate(unique_labels):
#         mask = spp == label
#         colors[mask] = color_map[i].astype(np.uint8)
    
#     # 计算法线
#     pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
#     pcd.orient_normals_consistent_tangent_plane(k=30)
    
#     # 重新创建带法线的点云
#     labeled_pcd = o3d.geometry.PointCloud()
#     labeled_pcd.points = o3d.utility.Vector3dVector(points.astype(np.float32))
#     labeled_pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
#     labeled_pcd.normals = pcd.normals
    
#     # 生成网格
#     radius = np.mean(labeled_pcd.compute_nearest_neighbor_distance()) * 2
#     mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#         labeled_pcd, o3d.utility.DoubleVector([radius, radius * 2]))
    
#     # 保存为PLY
#     o3d.io.write_triangle_mesh(output_path, mesh, write_vertex_colors=True)


import torch
import open3d as o3d
import numpy as np

def convert_spp_to_labeled_ply(point_cloud_path, spp_path, output_path):
    # 读取点云
    pcd = o3d.io.read_point_cloud(point_cloud_path)
    points = np.asarray(pcd.points)
    
    # 读取superpoint标签
    spp = torch.load(spp_path)
    if isinstance(spp, torch.Tensor):
        spp = spp.cpu().numpy()
    
    # 确保点云数量和标签数量一致
    assert len(points) == len(spp), f"Points count ({len(points)}) doesn't match labels count ({len(spp)})"
    
    # 生成颜色映射
    unique_labels = np.unique(spp)
    color_map = np.random.rand(len(unique_labels), 3) * 255
    
    # 生成点云颜色
    colors = np.zeros((len(points), 3), dtype=np.uint8)
    for i, label in enumerate(unique_labels):
        mask = spp == label
        colors[mask] = color_map[i].astype(np.uint8)
    
    # 计算法线
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=30)
    
    # 创建带颜色和法线的点云
    labeled_pcd = o3d.geometry.PointCloud()
    labeled_pcd.points = o3d.utility.Vector3dVector(points.astype(np.float32))
    labeled_pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    labeled_pcd.normals = pcd.normals
    
    # 使用Poisson重建来生成更密集的网格
    print("正在进行Poisson重建...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        labeled_pcd, 
        depth=8,  # 可以调整这个参数来控制细节程度
        width=0,
        scale=1.1,
        linear_fit=False
    )
    
    # 基于密度过滤掉一些低质量的面片
    vertices_to_remove = densities < np.quantile(densities, 0.1)  # 移除密度最低的10%
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    # 对网格进行颜色插值
    print("正在进行颜色插值...")
    pcd_tree = o3d.geometry.KDTreeFlann(labeled_pcd)
    mesh_vertices = np.asarray(mesh.vertices)
    mesh_colors = np.zeros((len(mesh_vertices), 3))
    
    # 使用K近邻插值来获得更平滑的颜色过渡
    K = 5  # 使用5个最近邻点
    for i, vertex in enumerate(mesh_vertices):
        k, idx, _ = pcd_tree.search_knn_vector_3d(vertex, K)
        # 使用距离加权平均来计算颜色
        neighbor_colors = np.asarray(labeled_pcd.colors)[idx]
        mesh_colors[i] = np.mean(neighbor_colors, axis=0)
    
    # 设置网格顶点颜色
    mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)
    
    # 网格优化
    print("正在优化网格...")
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    
    # 可选：简化网格以减少面片数量，同时保持形状
    print("正在简化网格...")
    target_number = len(points) // 2  # 可以调整这个比例
    mesh = mesh.simplify_quadric_decimation(target_number)
    
    # 打印一些统计信息
    print(f"原始点云数量: {len(points)}")
    print(f"网格顶点数量: {len(mesh.vertices)}")
    print(f"网格面片数量: {len(mesh.triangles)}")
    
    # 保存为PLY
    print("正在保存网格...")
    o3d.io.write_triangle_mesh(output_path, mesh, write_vertex_colors=True)
    print(f"已保存到: {output_path}")

convert_spp_to_labeled_ply(point_cloud_path, spp_path, output_path)
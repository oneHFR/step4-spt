import torch
import open3d as o3d
import numpy as np
from tqdm import tqdm
import time

def get_reference_mesh_faces(mesh_path):
    """获取参考网格的面片数量"""
    ref_mesh = o3d.io.read_triangle_mesh(mesh_path)
    return len(ref_mesh.triangles)

def convert_spp_to_labeled_ply(point_cloud_path, spp_path, output_path):
    start_time = time.time()
    
    # 获取目标面片数量
    target_faces = get_reference_mesh_faces(point_cloud_path)
    print(f"参考模型面片数量: {target_faces}")
    
    # 读取点云和标签
    print("正在读取点云和标签...")
    pcd = o3d.io.read_point_cloud(point_cloud_path)
    points = np.asarray(pcd.points)
    
    spp = torch.load(spp_path)
    if isinstance(spp, torch.Tensor):
        spp = spp.cpu().numpy()
    
    assert len(points) == len(spp), f"Points count ({len(points)}) doesn't match labels count ({len(spp)})"
    
    # 生成颜色映射
    print("生成颜色映射...")
    unique_labels = np.unique(spp)
    color_map = np.random.rand(len(unique_labels), 3) * 255
    
    colors = np.zeros((len(points), 3), dtype=np.uint8)
    for i, label in tqdm(enumerate(unique_labels), desc="处理颜色标签", total=len(unique_labels)):
        mask = spp == label
        colors[mask] = color_map[i].astype(np.uint8)
    
    # 创建带颜色的点云
    labeled_pcd = o3d.geometry.PointCloud()
    labeled_pcd.points = o3d.utility.Vector3dVector(points.astype(np.float32))
    labeled_pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    
    # 计算法线
    print("计算法线中...")
    labeled_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50)
    )
    labeled_pcd.orient_normals_consistent_tangent_plane(k=50)
    print("法线计算完成")
    
    # 点云滤波和下采样
    print("正在进行点云优化...")
    labeled_pcd = labeled_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)[0]
    print("点云优化完成")
    
    # 使用Ball Pivoting算法重建
    print("使用Ball Pivoting算法重建网格...")
    radii = [0.005, 0.01, 0.02, 0.04]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        labeled_pcd, o3d.utility.DoubleVector(radii)
    )
    print("网格重建完成")
    
    # 网格修复和优化
    print("开始网格优化...")
    with tqdm(total=4, desc="网格修复") as pbar:
        mesh.remove_degenerate_triangles()
        pbar.update(1)
        mesh.remove_duplicated_triangles()
        pbar.update(1)
        mesh.remove_duplicated_vertices()
        pbar.update(1)
        mesh.remove_non_manifold_edges()
        pbar.update(1)
    
    # 填充孔洞
    print("检查并填充孔洞...")
    holes = mesh.get_non_manifold_edges()
    if len(holes) > 0:
        mesh = mesh.fill_holes()
        print(f"已填充 {len(holes)} 个孔洞")
    
    # 平滑处理
    print("进行平滑处理...")
    for i in tqdm(range(50), desc="平滑迭代"):
        mesh = mesh.filter_smooth_taubin(number_of_iterations=1)
    
    # 调整面片数量
    current_faces = len(mesh.triangles)
    print(f"当前面片数量: {current_faces}")
    
    if current_faces < target_faces:
        print("正在细分网格...")
        print("检查并修复网格流形性...")
        
        # 增强网格清理
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        
        # 检查是否存在非流形边
        non_manifold_edges = mesh.get_non_manifold_edges()
        if len(non_manifold_edges) > 0:
            print(f"发现 {len(non_manifold_edges)} 个非流形边，尝试修复...")
            # 移除问题区域并重新封闭网格
            mesh = mesh.remove_vertices_by_mask(non_manifold_edges)
            mesh.fill_holes()
        
        # 使用更保守的细分方式
        with tqdm(total=target_faces - current_faces, desc="增加面片") as pbar:
            while len(mesh.triangles) < target_faces:
                prev_faces = len(mesh.triangles)
                # 每次只细分一小部分面片
                if (target_faces - prev_faces) > 1000:
                    target_this_iteration = prev_faces + 1000
                else:
                    target_this_iteration = target_faces
                    
                mesh = mesh.subdivide_midpoint()  # 使用midpoint细分替代loop细分
                
                # 如果细分后面片数量过多，进行适当简化
                if len(mesh.triangles) > target_this_iteration:
                    mesh = mesh.simplify_quadric_decimation(target_this_iteration)
                
                new_faces = len(mesh.triangles)
                pbar.update(new_faces - prev_faces)
                if new_faces >= target_faces:
                    break
                
                # 每次迭代后检查并修复可能的问题
                mesh.remove_degenerate_triangles()
                mesh.remove_duplicated_triangles()
    elif current_faces > target_faces:
        print("正在简化网格...")
        mesh = mesh.simplify_quadric_decimation(target_faces)
    
    # 颜色插值优化
    print("优化顶点颜色...")
    pcd_tree = o3d.geometry.KDTreeFlann(labeled_pcd)
    mesh_vertices = np.asarray(mesh.vertices)
    mesh_colors = np.zeros((len(mesh_vertices), 3))
    
    for i in tqdm(range(len(mesh_vertices)), desc="颜色插值"):
        # 增加搜索邻居数量以确保足够的样本
        k, idx, distances = pcd_tree.search_knn_vector_3d(mesh_vertices[i], 16)
        
        if k > 0:
            neighbor_colors = np.asarray(labeled_pcd.colors)[idx]
            
            # 使用反距离加权
            weights = 1.0 / (distances + 1e-6)  # 添加小值避免除零
            
            # 确保权重和不为零
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
                mesh_colors[i] = np.average(neighbor_colors, weights=weights, axis=0)
            else:
                # 如果权重和为零，使用简单平均
                mesh_colors[i] = np.mean(neighbor_colors, axis=0)
        else:
            # 如果没有找到邻居，使用默认颜色
            mesh_colors[i] = np.array([0.5, 0.5, 0.5])
    
    mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)
    
    # 最终平滑
    print("进行最终平滑...")
    for i in tqdm(range(20), desc="最终平滑"):
        mesh = mesh.filter_smooth_taubin(number_of_iterations=1)
    
    # 打印统计信息和总耗时
    end_time = time.time()
    total_time = end_time - start_time
    print("\n处理完成！")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"原始点云数量: {len(points)}")
    print(f"网格顶点数量: {len(mesh.vertices)}")
    print(f"网格面片数量: {len(mesh.triangles)}")
    print(f"目标面片数量: {target_faces}")
    
    # 保存网格
    print(f"正在保存到: {output_path}")
    o3d.io.write_triangle_mesh(output_path, mesh, write_vertex_colors=True)
    print("保存完成！")

# 使用示例
if __name__ == "__main__":
    scene_id = 'scene0011_00'
    point_cloud_path = f'{scene_id}_vh_clean_2.ply'
    spp_path = f'{scene_id}.pth'
    output_path = f'{scene_id}_b2.ply'
    
    convert_spp_to_labeled_ply(point_cloud_path, spp_path, output_path)
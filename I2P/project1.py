import numpy as np
import cv2
from plyfile import PlyData
from scipy.spatial import cKDTree
from pathlib import Path
from tqdm import tqdm
from plyfile import PlyData, PlyElement
import plotly.graph_objects as go
import pandas as pd
import torch
import torch
from collections import Counter

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

class PointCloudMatcher:
    def __init__(self, ply_path, data_dict):
        ply_data = PlyData.read(ply_path)
        vertices = ply_data['vertex'].data
        self.points_ply = np.array([vertices['x'], vertices['y'], vertices['z']]).T
        self.kdtree = cKDTree(self.points_ply)
        self.dict = data_dict

    def depth_to_point_cloud(self, depth_image, intrinsic_matrix, pose_matrix, mask=None, depth_scale=1000.0, intrinsic_resolution=[968, 1296]):
        fx, fy, cx, cy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1], intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
        
        # 强制调整深度图和内参矩阵为标准分辨率
        height, width = intrinsic_resolution
        depth_image_resized = cv2.resize(depth_image, (width, height), interpolation=cv2.INTER_NEAREST)
        
        # 调整内参矩阵
        intrinsic_matrix_resized = intrinsic_matrix.copy()
        intrinsic_matrix_resized[0, 0] = intrinsic_matrix[0, 0] * (width / depth_image.shape[1])
        intrinsic_matrix_resized[1, 1] = intrinsic_matrix[1, 1] * (height / depth_image.shape[0])
        intrinsic_matrix_resized[0, 2] = intrinsic_matrix[0, 2] * (width / depth_image.shape[1])
        intrinsic_matrix_resized[1, 2] = intrinsic_matrix[1, 2] * (height / depth_image.shape[0])

        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        
        if mask is not None:
            # Ensure mask matches the resolution of depth image
            if mask.shape != (height, width):
                mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
            valid_mask = (mask > 0) & (depth_image_resized > 0)
        else:
            valid_mask = depth_image_resized > 0
        
        # 调试：输出 valid_mask 的有效点数
        print(f"有效点数: {np.sum(valid_mask)}")
        
        x, y, depth = x[valid_mask], y[valid_mask], depth_image_resized[valid_mask] / depth_scale
        points_camera = np.stack([(x - cx) * depth / fx, (y - cy) * depth / fy, depth, np.ones_like(depth)]).T
        points_world = (pose_matrix @ points_camera.T).T
        return points_world[:, :3]

    def analyze_matching_ratio(self, matched_points, total_points):
        num_matched = len(matched_points)
        num_total = len(total_points)
        ratio = num_matched / num_total * 100

        print(f"\n匹配统计:")
        print(f"总点数: {num_total}")
        print(f"匹配点数: {num_matched}")
        print(f"匹配比例: {ratio:.2f}%")
        
        return ratio

    def find_matching_points(self, query_points, distance_threshold=1.0):
        distances, indices = self.kdtree.query(query_points, k=1)
        mask = distances < distance_threshold
        
        matched_indices = indices[mask]
        matched_ply_points = self.points_ply[matched_indices]
        matched_query_points = query_points[mask]
        
        matching_mask = np.zeros(len(self.points_ply), dtype=bool)
        matching_mask[matched_indices] = True
        
        print(f"\n点云匹配统计:")
        print(f"查询点云总数: {len(query_points)}")
        print(f"匹配点数: {np.sum(matching_mask)}")
        print(f"匹配比例1(sam): {np.sum(matching_mask)/len(self.points_ply)*100:.2f}%")
        print(f"匹配比例2(m3): {np.sum(matching_mask)/len(self.dict)*100:.2f}%")
        print(f"匹配比例3(gt): {np.sum(matching_mask)/len(self.dict)*100:.2f}%")
        
        return matched_ply_points, matched_query_points, distances[mask], matching_mask

    def random_sample_check(self, matched_distances, sample_size=200):
        # 随机检查200个匹配点的距离
        sample_indices = np.random.choice(len(matched_distances), sample_size, replace=False)
        print("\n随机采样点对比:")
        for idx in sample_indices:
            print(f"距离: {matched_distances[idx]:.4f}")

    def modify_matched_points_color(self, ply_path, matched_indices, output_path):
        ply_data = PlyData.read(ply_path)
        vertices = ply_data['vertex'].data
        
        vertices['red'][matched_indices] = 0
        vertices['green'][matched_indices] = 0
        vertices['blue'][matched_indices] = 0

        vertex_element = PlyElement.describe(vertices, 'vertex')
        face_element = PlyElement.describe(ply_data['face'].data, 'face') if 'face' in ply_data else None
        elements = [vertex_element, face_element] if face_element else [vertex_element]
        
        PlyData(elements, text=True).write(output_path)

def visualize_matched_points(original_points, matched_points, output_path):
    df_original = pd.DataFrame(original_points, columns=['x', 'y', 'z'])
    df_original['type'] = 'Unmatched'
    
    df_matched = pd.DataFrame(matched_points, columns=['x', 'y', 'z'])
    df_matched['type'] = 'Matched'
    
    df = pd.concat([df_original, df_matched])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter3d(
        x=df_matched['x'], y=df_matched['y'], z=df_matched['z'],
        mode='markers',
        marker=dict(size=2, color='white', opacity=1),
        name='Matched Points'
    ))
    
    fig.update_layout(
        scene=dict(
            bgcolor='rgb(20, 24, 35)',
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False)
        ),
        paper_bgcolor='rgb(20, 24, 35)',
        font=dict(color='white'),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    fig.write_html(output_path)

def update_sam_for_scene(ply_path, intrinsic_matrix_path, mask_dir, pose_dir, depth_dir, data_dict):
    intrinsic_matrix = np.loadtxt(intrinsic_matrix_path)
    matcher = PointCloudMatcher(ply_path, data_dict)

    for subfolder in os.listdir(mask_dir):
        subfolder_path = os.path.join(mask_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        
        m3_id = int(subfolder.split('_')[1])
        for mask_file in os.listdir(subfolder_path):
            if mask_file.endswith(".npy"):
                depth_index = int(mask_file.split('_')[1])*10
                depth_path = f"{depth_dir}{depth_index}.png"
                pose_path = f"{pose_dir}{depth_index}.txt"
                mask_path = os.path.join(subfolder_path, mask_file)

                if not Path(depth_path).exists():
                    print(f"深度图文件 {depth_path} 不存在，跳过...")
                    continue
                if not Path(pose_path).exists():
                    print(f"位姿矩阵文件 {pose_path} 不存在，跳过...")
                    continue
                if not Path(mask_path).exists():
                    print(f"掩码文件 {mask_path} 不存在，跳过...")
                    continue

                mask = np.load(mask_path)
                depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                pose_matrix = np.loadtxt(pose_path)
                
                query_points = matcher.depth_to_point_cloud(depth_image, intrinsic_matrix, pose_matrix, mask, intrinsic_resolution=[968, 1296])
                matched_ply_points, matched_query_points, distances, matched_indices = matcher.find_matching_points(query_points)
                
                # 更新字典中的sam值
                for idx in matched_indices:
                    if data_dict.get(idx):
                        point_info = data_dict[idx]
                        if point_info['m3_id'] == m3_id:
                            point_info['sam'] = 1
                        else:
                            point_info['sam'] = -1
            
    # # 保存更新后的字典
    # updated_dict_path = f"scene{scene_id}/updated_0011_00_dict.pt"
    # torch.save(data_dict, updated_dict_path)
    # print(f"更新后的字典已保存至 {updated_dict_path}.")

def process_scene(scene_id):
    ply_path = f"scene{scene_id}/scene{scene_id}_vh_clean_2.labels.ply"
    intrinsic_matrix_path = f"scene{scene_id}/data/intrinsic/intrinsic_color.txt"
    pose_matrix_path = f"scene{scene_id}/data/pose/"
    depth_path = f"scene{scene_id}/data_compressed/depth/"
    mask_path = f"scene{scene_id}/sam_masks"
    dict_path = f"scene{scene_id}/{scene_id}_dict.pt"
    if not Path(ply_path).exists():
        print(f"PLY文件 {ply_path} 不存在！")
        return
    if not Path(intrinsic_matrix_path).exists():
        print(f"内参矩阵文件 {intrinsic_matrix_path} 不存在！")
        return
    if not Path(pose_matrix_path).exists():
        print(f"位姿矩阵文件夹 {pose_matrix_path} 不存在！")
        return
    if not Path(depth_path).exists():
        print(f"深度图文件夹 {depth_path} 不存在！")
        return
    if not Path(mask_path).exists():
        print(f"掩码文件夹 {mask_path} 不存在！")
        return
    if not Path(dict_path).exists():
        print(f"字典文件 {dict_path} 不存在！")
        return
    data_dict = torch.load(dict_path)
    for point_id, point_info in data_dict.items():
        point_info['sam'] = 0  # 初始化'sam'为0

    update_sam_for_scene(ply_path, intrinsic_matrix_path, mask_path, pose_matrix_path, depth_path, data_dict)

    updated_dict_path = f"scene{scene_id}/updated_{scene_id}_dict.pt"
    torch.save(data_dict, updated_dict_path)
    print(f"更新后的字典已保存至 {updated_dict_path}.")

    print(f"开始处理场景 {scene_id}...")

def batch_process_scenes(scene_ids):
    for scene_id in scene_ids:
        process_scene(scene_id)

if __name__ == "__main__":
    scene_ids = ['0011_00']
    scene_id = scene_ids[0]
    # batch_process_scenes(scene_ids)
    updated_dict_path = f"scene{scene_id}/updated_{scene_id}_dict.pt"
    updated_dict_path2 = f"scene{scene_id}/2_updated_{scene_id}_dict.pt"
    update_dict = torch.load(updated_dict_path)
    update_dict2 = torch.load(updated_dict_path2)

    # 统计sam值的分布
    sam_values = [info['sam'] for info in update_dict2.values()]
    value_counts = Counter(sam_values)

    # 计算总数
    total_count = len(sam_values)

    # 打印每个值的数量和百分比
    print("sam值分布情况:")
    print("-" * 40)
    print("sam值  |  数量  |  百分比")
    print("-" * 40)

    for sam_value in sorted(value_counts.keys()):
        count = value_counts[sam_value]
        percentage = (count / total_count) * 100
        print(f"{sam_value:4d}  | {count:6d} | {percentage:6.2f}%")

    print("-" * 40)
    print(f"总记录数: {total_count}")



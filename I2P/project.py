import os
from pathlib import Path
import numpy as np
import cv2
import torch
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

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
        height, width = intrinsic_resolution
        depth_image_resized = cv2.resize(depth_image, (width, height), interpolation=cv2.INTER_NEAREST)
        
        intrinsic_matrix_resized = intrinsic_matrix.copy()
        intrinsic_matrix_resized[0, 0] = intrinsic_matrix[0, 0] * (width / depth_image.shape[1])
        intrinsic_matrix_resized[1, 1] = intrinsic_matrix[1, 1] * (height / depth_image.shape[0])
        intrinsic_matrix_resized[0, 2] = intrinsic_matrix[0, 2] * (width / depth_image.shape[1])
        intrinsic_matrix_resized[1, 2] = intrinsic_matrix[1, 2] * (height / depth_image.shape[0])

        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        
        if mask is not None:
            if mask.shape != (height, width):
                mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
            valid_mask = (mask > 0) & (depth_image_resized > 0)
        else:
            valid_mask = depth_image_resized > 0
        
        # print(f"有效点数: {np.sum(valid_mask)}")
        
        x, y, depth = x[valid_mask], y[valid_mask], depth_image_resized[valid_mask] / depth_scale
        points_camera = np.stack([(x - cx) * depth / fx, (y - cy) * depth / fy, depth, np.ones_like(depth)]).T
        points_world = (pose_matrix @ points_camera.T).T
        return points_world[:, :3]

    def find_matching_points(self, query_points, m3_id, distance_threshold=1.0):
        distances, indices = self.kdtree.query(query_points, k=1)
        mask = distances < distance_threshold
        
        matched_indices = indices[mask]
        matched_ply_points = self.points_ply[matched_indices]
        matched_query_points = query_points[mask]
        
        matching_mask = np.zeros(len(self.points_ply), dtype=bool)
        matching_mask[matched_indices] = True

        # 同步更新 self.dict 中的 sam 值
        matched_keys = [idx for idx in matched_indices if idx in self.dict]
        for key in matched_keys:
            point_info = self.dict[key]
            # 检查 m3_id 是否是列表
            if isinstance(point_info['m3_id'], (list, tuple)):
                m3_id_list = point_info['m3_id']
                sam_list = point_info['sam']
                # 如果 m3_id 在 m3_id_list 中，更新对应位置为 1（保持不变如果已为 1）
                if m3_id in m3_id_list:
                    for i, m3_val in enumerate(m3_id_list):
                        if m3_val == m3_id and sam_list[i] != 1:  # 只在非 1 时更新为 1
                            sam_list[i] = 1
                else:
                    # 如果 m3_id 不在列表中，所有位置设为 -1
                    point_info['sam'] = [-1] * len(m3_id_list)
            else:
                # 如果 m3_id 不是列表，按单值处理（兼容旧逻辑）
                point_info['sam'] = [1] if point_info['m3_id'] == m3_id else [-1]

        # print(f"查询点云总数: {len(query_points)}")
        # print(f"匹配点数: {np.sum(matching_mask)}")
        
        return matched_ply_points, matched_query_points, distances[mask], matching_mask

def process_subfolder(subfolder_path, m3_id, matcher, intrinsic_matrix, depth_dir, pose_dir):
    start_time = time.time()
    processed_files = 0
    total_files = sum(1 for f in os.listdir(subfolder_path) if f.endswith(".npy"))
    
    for mask_file in os.listdir(subfolder_path):
        if mask_file.endswith(".npy"):
            depth_index = int(mask_file.split('_')[1]) * 10
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
            matcher.find_matching_points(query_points, m3_id)
            
            processed_files += 1
            elapsed_time = time.time() - start_time
            avg_time_per_file = elapsed_time / processed_files
            estimated_remaining = avg_time_per_file * (total_files - processed_files)
            
            # print(f"已处理: {processed_files}/{total_files} 文件 | 平均每文件耗时: {avg_time_per_file:.2f}秒 | 预计剩余时间: {estimated_remaining:.2f}秒")

def update_sam_for_scene(ply_path, intrinsic_matrix_path, mask_dir, pose_dir, depth_dir, data_dict):
    intrinsic_matrix = np.loadtxt(intrinsic_matrix_path)
    matcher = PointCloudMatcher(ply_path, data_dict)
    
    total_start_time = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for subfolder in os.listdir(mask_dir):
            subfolder_path = os.path.join(mask_dir, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            
            m3_id = int(subfolder.split('_')[1])
            futures.append(executor.submit(process_subfolder, subfolder_path, m3_id, matcher, intrinsic_matrix, depth_dir, pose_dir))
        
        for future in tqdm(as_completed(futures), desc="Processing Subfolders", total=len(futures)):
            future.result()  # 等待所有任务完成

    total_time = time.time() - total_start_time
    print(f"\n总处理时间: {total_time:.2f}秒")

def process_scene(scene_id):
    start_time = time.time()

    ply_path = f"scene{scene_id}/scene{scene_id}_vh_clean_2.labels.ply"
    intrinsic_matrix_path = f"scene{scene_id}/data/intrinsic/intrinsic_color.txt"
    mask_path = f"scene{scene_id}/sam_masks"
    pose_matrix_path = f"scene{scene_id}/data/pose/"
    depth_path = f"scene{scene_id}/data_compressed/depth/"
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
    # 修改初始化：sam 与 m3_id 长度相同，全为 0 的列表
    for point_id, point_info in data_dict.items():
        m3_id_len = len(point_info['m3_id']) if isinstance(point_info['m3_id'], (list, tuple)) else 1
        point_info['sam'] = [0] * m3_id_len  # 初始化为长度匹配的全 0 列表

    update_sam_for_scene(ply_path, intrinsic_matrix_path, mask_path, pose_matrix_path, depth_path, data_dict)

    updated_dict_path = f"scene{scene_id}/updated_{scene_id}_dict.pt"
    torch.save(data_dict, updated_dict_path)
    print(f"更新后的字典已保存至 {updated_dict_path}.")

def batch_process_scenes(scene_ids):
    batch_start_time = time.time()
    
    for scene_id in scene_ids:
        process_scene(scene_id)
    
    batch_total_time = time.time() - batch_start_time
    print(f"\n批处理总时间: {batch_total_time:.2f}秒")

if __name__ == "__main__":
    scene_ids = ['0011_00']
    batch_process_scenes(scene_ids)

    scene_id = scene_ids[0]
    updated_dict_path2 = f"scene{scene_id}/updated_{scene_id}_dict.pt"
    update_dict2 = torch.load(updated_dict_path2)
    a=1
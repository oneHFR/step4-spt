"""
v1
"""
# import torch
# import numpy as np
# import plotly.graph_objects as go
# from plyfile import PlyData

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

# # 1. 加载 .ply 文件，提取坐标
# def load_ply(ply_file):
#     ply_data = PlyData.read(ply_file)
#     vertices = ply_data['vertex'].data
#     points = np.array([vertices['x'], vertices['y'], vertices['z']]).T
#     return points

# # 2. 修改 load_pt 函数，对 m3_ids 进行特殊处理
# def load_pt(pt_file, is_m3=False):
#     data = torch.load(pt_file)
#     if is_m3:  # 如果是加载 m3_ids，保持其原有结构
#         return data
#     elif isinstance(data, torch.Tensor):
#         return data.numpy()
#     elif isinstance(data, list):
#         data_flat = [item if isinstance(item, (int, float)) else item[0] for item in data]
#         return np.array(data_flat)
#     return data

# # 3. 修改合并数据的函数
# def merge_data(ply_file, spt_file, gt_file, m3_file, output_file):
#     points = load_ply(ply_file)
#     spt_ids = load_pt(spt_file).flatten()
#     gt_ids = load_pt(gt_file).flatten()
#     m3_ids = load_pt(m3_file, is_m3=True)  # 不进行 flatten

#     spt_ids = np.array(spt_ids).flatten()
#     gt_ids = np.array(gt_ids).flatten()

#     assert len(points) == len(spt_ids) == len(gt_ids) == len(m3_ids), "The lengths of points and labels must match."
    
#     # 修改计数方式
#     m3_counts = {}
#     for m3_id_list in m3_ids:
#         for m3_id in m3_id_list:
#             if m3_id not in m3_counts:
#                 m3_counts[m3_id] = 0
#             m3_counts[m3_id] += 1

#     gt_counts = {gt_id: np.sum(gt_ids == gt_id) for gt_id in np.unique(gt_ids)}
#     spt_counts = {spt_id: np.sum(spt_ids == spt_id) for spt_id in np.unique(spt_ids)}

#     data = {}
#     for i in range(len(points)):
#         point_info = {
#             'spt_id': spt_ids[i],
#             'spt_num': spt_counts[spt_ids[i]],
#             'gt_id': gt_ids[i],
#             'gt_num': gt_counts[gt_ids[i]],
#             'm3_id': m3_ids[i],
#             'm3_num': sum(m3_counts[m3_id] for m3_id in m3_ids[i]),
#             'coordinates': points[i]
#         }
#         data[i] = point_info

#     torch.save(data, output_file)
#     output_file2 = f'scene{scene_id}/{output_file}'
#     torch.save(data, output_file2)
#     print(f"数据已保存为 {output_file}")

# scene_id = '0011_00'
# ply_file = f'scene{scene_id}/scene{scene_id}_vh_clean_2.ply'
# spt_file = 'output_masks/cluster.pt'
# gt_file = 'output_masks/gt.pt'
# m3_file = f'{scene_id}_m3_ids.pt'
# gt_file = f'{scene_id}_gt_ids.pt'
# spt_file = f'{scene_id}_spt_ids.pt'
# output_file = f'{scene_id}_dict.pt'

# merge_data(ply_file, spt_file, gt_file, m3_file, output_file)

# # dict_file = output_file
# # id_dict = load_pt(dict_file)
# # a=1
# # 合并数据并保存


"""
v2
"""
# import time
# from tqdm import tqdm
# import numpy as np
# import torch
# from plyfile import PlyData
# import os
# from typing import Dict, Any, Union

# from collections import Counter
# from typing import Dict, List, Tuple

# def analyze_m3_to_gt_relationship(data_dict: Dict) -> Tuple[List[Tuple[int, int]], Dict[int, Dict[int, float]]]:
#     """
#     分析 m3_id 与 gt_id 的对应关系，并计算每个 m3_id 下 gt_id 的占比。
    
#     Args:
#         data_dict: 包含点信息的字典，键为点索引，值为点信息字典。
    
#     Returns:
#         m3_to_gt_mapping: 列表，每个元素为 (m3_id, dominant_gt_id)，表示 m3_id 与主要 gt_id 的对应关系。
#         m3_to_gt_ratios: 字典，每个 m3_id 映射到其 gt_id 的占比分布。
#     """
#     # Step 1: 收集每个 m3_id 对应的 gt_id 分布
#     m3_to_gt_counts: Dict[int, Dict[int, int]] = {}
    
#     for point_id, point_info in data_dict.items():
#         gt_id = point_info['gt_id']
#         m3_ids = point_info['m3_id'] if isinstance(point_info['m3_id'], (list, tuple)) else [point_info['m3_id']]
        
#         for m3_id in m3_ids:
#             if m3_id not in m3_to_gt_counts:
#                 m3_to_gt_counts[m3_id] = {}
#             m3_to_gt_counts[m3_id][gt_id] = m3_to_gt_counts[m3_id].get(gt_id, 0) + 1
    
#     # Step 2: 计算每个 m3_id 的 gt_id 占比并确定主导 gt_id
#     m3_to_gt_mapping = []
#     m3_to_gt_ratios = {}
    
#     for m3_id, gt_counts in m3_to_gt_counts.items():
#         total_points = sum(gt_counts.values())
#         gt_ratios = {gt_id: count / total_points for gt_id, count in gt_counts.items()}
#         m3_to_gt_ratios[m3_id] = gt_ratios
        
#         # 找到占比最高的 gt_id
#         dominant_gt_id = max(gt_counts.items(), key=lambda x: x[1])[0]
#         dominant_ratio = gt_ratios[dominant_gt_id]
        
#         # 如果占比超过一定阈值（例如 50%），认为有对应关系
#         if dominant_ratio >= 0.5:  # 可调整阈值
#             m3_to_gt_mapping.append((m3_id, dominant_gt_id))
    
#     return m3_to_gt_mapping, m3_to_gt_ratios

# # 时间记录装饰器
# def time_logger(func):
#     def wrapper(*args, **kwargs):
#         start_time = time.time()
#         print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始执行 {func.__name__}")
#         result = func(*args, **kwargs)
#         end_time = time.time()
#         print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {func.__name__} 执行完成，耗时: {end_time - start_time:.2f}秒")
#         return result
#     return wrapper

# # 1. 优化加载 .ply 文件，使用缓存
# @time_logger
# def load_ply(ply_file: str, cache_dir: str = "./cache") -> np.ndarray:
#     cache_file = os.path.join(cache_dir, f"{os.path.basename(ply_file)}.npy")
#     if os.path.exists(cache_file):
#         return np.load(cache_file)
    
#     ply_data = PlyData.read(ply_file)
#     vertices = ply_data['vertex'].data
#     points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
#     os.makedirs(cache_dir, exist_ok=True)
#     np.save(cache_file, points)
#     return points

# # 2. 优化 load_pt 函数，使用类型提示
# @time_logger
# def load_pt(pt_file: str, is_m3: bool = False) -> Union[np.ndarray, Any]:
#     data = torch.load(pt_file, map_location='cpu')  # 使用CPU加载减少内存压力
#     if is_m3:
#         return data
#     if isinstance(data, torch.Tensor):
#         return data.numpy()
#     if isinstance(data, list):
#         return np.array([item if isinstance(item, (int, float)) else item[0] for item in data])
#     return data

# # 3. 通用的数据处理类
# class SceneDataProcessor:
#     def __init__(self, scene_id: str, output_dir: str = "."):
#         self.scene_id = scene_id
#         self.output_dir = output_dir
#         self.timestamps = {}

#     @time_logger
#     def process_data(self, config: Dict[str, str]) -> Dict[int, Dict[str, Any]]:
#         # 加载数据
#         points = load_ply(config['ply_file'])
#         spt_ids = load_pt(config['spt_file']).flatten()
#         gt_ids = load_pt(config['gt_file']).flatten()
#         m3_ids = load_pt(config['m3_file'], is_m3=True)

#         # 数据长度校验
#         assert len(points) == len(spt_ids) == len(gt_ids) == len(m3_ids), \
#             "The lengths of points and labels must match."

#         # 使用 numpy 的向量化操作替代循环
#         unique_spt, spt_counts = np.unique(spt_ids, return_counts=True)
#         unique_gt, gt_counts = np.unique(gt_ids, return_counts=True)
#         spt_count_dict = dict(zip(unique_spt, spt_counts))
#         gt_count_dict = dict(zip(unique_gt, gt_counts))

#         # 处理 m3_ids 的计数
#         m3_counts = {}
#         for m3_id_list in tqdm(m3_ids, desc="Processing m3_ids"):
#             for m3_id in m3_id_list:
#                 m3_counts[m3_id] = m3_counts.get(m3_id, 0) + 1

#         # 构建结果字典
#         data = {}
#         for i in tqdm(range(len(points)), desc="Merging data"):
#             point_info = {
#                 'spt_id': spt_ids[i],
#                 'spt_num': spt_count_dict[spt_ids[i]],
#                 'gt_id': gt_ids[i],
#                 'gt_num': gt_count_dict[gt_ids[i]],
#                 'm3_id': m3_ids[i],
#                 'm3_num': sum(m3_counts[m3_id] for m3_id in m3_ids[i]),
#                 'coordinates': points[i]
#             }
#             data[i] = point_info

#         return data

#     @time_logger
#     def save_data(self, data: Dict[int, Dict[str, Any]], output_file: str):
#         output_path = os.path.join(self.output_dir, output_file)
#         scene_output_path = os.path.join(self.output_dir, f"scene{self.scene_id}", output_file)
#         os.makedirs(os.path.dirname(scene_output_path), exist_ok=True)
        
#         torch.save(data, output_path)
#         torch.save(data, scene_output_path)
#         print(f"数据已保存为 {output_path} 和 {scene_output_path}")

# # 4. 抽象的顶层调用函数
# def process_scene(scene_id: str, config: Dict[str, str], output_dir: str = "."):
#     processor = SceneDataProcessor(scene_id, output_dir)
#     data = processor.process_data(config)
#     processor.save_data(data, config['output_file'])

# # 使用示例
# if __name__ == "__main__":
#     scene_id = '0011_00'
#     config = {
#         'ply_file': f'scene{scene_id}/scene{scene_id}_vh_clean_2.ply',
#         'spt_file': f'{scene_id}_spt_ids.pt',
#         'gt_file': f'{scene_id}_gt_ids.pt',
#         'm3_file': f'{scene_id}_m3_ids.pt',
#         'output_file': f'{scene_id}_dict.pt'
#     }
    
#     process_scene(scene_id, config)



"""
v3
"""
import time
from tqdm import tqdm
import numpy as np
import torch
from plyfile import PlyData
import os
from typing import Dict, Any, Union, List, Tuple

# 时间记录装饰器
def time_logger(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始执行 {func.__name__}")
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {func.__name__} 执行完成，耗时: {end_time - start_time:.2f}秒")
        return result
    return wrapper

# 1. 优化加载 .ply 文件，使用缓存
@time_logger
def load_ply(ply_file: str, cache_dir: str = "./cache") -> np.ndarray:
    cache_file = os.path.join(cache_dir, f"{os.path.basename(ply_file)}.npy")
    if os.path.exists(cache_file):
        return np.load(cache_file)
    
    ply_data = PlyData.read(ply_file)
    vertices = ply_data['vertex'].data
    points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    os.makedirs(cache_dir, exist_ok=True)
    np.save(cache_file, points)
    return points

# 2. 优化 load_pt 函数，使用类型提示
@time_logger
def load_pt(pt_file: str, is_m3: bool = False) -> Union[np.ndarray, Any]:
    data = torch.load(pt_file, map_location='cpu')  # 使用CPU加载减少内存压力
    if is_m3:
        return data
    if isinstance(data, torch.Tensor):
        return data.numpy()
    if isinstance(data, list):
        return np.array([item if isinstance(item, (int, float)) else item[0] for item in data])
    return data

# 3. 通用的数据处理类
class SceneDataProcessor:
    def __init__(self, scene_id: str, output_dir: str = "."):
        self.scene_id = scene_id
        self.output_dir = output_dir
        self.timestamps = {}

    @time_logger
    def process_data(self, config: Dict[str, str]) -> Dict[int, Dict[str, Any]]:
        # 加载数据
        points = load_ply(config['ply_file'])
        spt_ids = load_pt(config['spt_file']).flatten()
        gt_ids = load_pt(config['gt_file']).flatten()
        m3_ids = load_pt(config['m3_file'], is_m3=True)

        # 数据长度校验
        assert len(points) == len(spt_ids) == len(gt_ids) == len(m3_ids), \
            "The lengths of points and labels must match."

        # 使用 numpy 的向量化操作替代循环
        unique_spt, spt_counts = np.unique(spt_ids, return_counts=True)
        unique_gt, gt_counts = np.unique(gt_ids, return_counts=True)
        spt_count_dict = dict(zip(unique_spt, spt_counts))
        gt_count_dict = dict(zip(unique_gt, gt_counts))

        # 处理 m3_ids 的计数
        m3_counts = {}
        for m3_id_list in tqdm(m3_ids, desc="Processing m3_ids"):
            for m3_id in m3_id_list:
                m3_counts[m3_id] = m3_counts.get(m3_id, 0) + 1

        # 构建结果字典
        data = {}
        for i in tqdm(range(len(points)), desc="Merging data"):
            point_info = {
                'spt_id': spt_ids[i],
                'spt_num': spt_count_dict[spt_ids[i]],
                'gt_id': gt_ids[i],
                'm3_id': m3_ids[i],
                'coordinates': points[i]
            }
            data[i] = point_info

        # 分析 m3_id 与 gt_id 的关系并保存结果
        m3_to_gt_mapping, gt_to_m3_mapping = self.analyze_id_relationships(data)
        self.save_relationships(m3_to_gt_mapping, gt_to_m3_mapping, m3_counts, gt_count_dict)

        return data

    @time_logger
    def analyze_id_relationships(self, data_dict: Dict) -> Tuple[List[Tuple[int, int, float]], List[Tuple[int, int, float]]]:
        """
        分析 m3_id 到 gt_id 和 gt_id 到 m3_id 的对应关系，包含置信度。
        
        Args:
            data_dict: 包含点信息的字典，键为点索引，值为点信息字典。
        
        Returns:
            m3_to_gt_mapping: 列表，每个元素为 (m3_id, dominant_gt_id, confidence)。
            gt_to_m3_mapping: 列表，每个元素为 (gt_id, dominant_m3_id, confidence)。
        """
        # Step 1: 统计 m3_id 到 gt_id 和 gt_id 到 m3_id 的分布
        m3_to_gt_counts: Dict[int, Dict[int, int]] = {}
        gt_to_m3_counts: Dict[int, Dict[int, int]] = {}
        all_gt_ids = set()
        
        for point_id, point_info in data_dict.items():
            gt_id = point_info['gt_id']
            m3_ids = point_info['m3_id'] if isinstance(point_info['m3_id'], (list, tuple)) else [point_info['m3_id']]
            all_gt_ids.add(gt_id)
            
            # m3_id -> gt_id
            for m3_id in m3_ids:
                if m3_id not in m3_to_gt_counts:
                    m3_to_gt_counts[m3_id] = {}
                m3_to_gt_counts[m3_id][gt_id] = m3_to_gt_counts[m3_id].get(gt_id, 0) + 1
            
            # gt_id -> m3_id
            if gt_id not in gt_to_m3_counts:
                gt_to_m3_counts[gt_id] = {}
            for m3_id in m3_ids:
                gt_to_m3_counts[gt_id][m3_id] = gt_to_m3_counts[gt_id].get(m3_id, 0) + 1
        
        # Step 2: 计算 m3_id 到 gt_id 的主导关系
        m3_to_gt_mapping = []
        for m3_id, gt_counts in m3_to_gt_counts.items():
            total_points = sum(gt_counts.values())
            dominant_gt_id = max(gt_counts.items(), key=lambda x: x[1])[0]
            confidence = round(gt_counts[dominant_gt_id] / total_points, 2)  # 修改此行
            m3_to_gt_mapping.append((m3_id, dominant_gt_id, confidence))
        
        # Step 3: 计算 gt_id 到 m3_id 的主导关系，确保所有 gt_id 都被包含
        gt_to_m3_mapping = []
        for gt_id in all_gt_ids:
            if gt_id not in gt_to_m3_counts:
                gt_to_m3_mapping.append((gt_id, -1, 0.0))
                continue
            
            m3_counts = gt_to_m3_counts[gt_id]
            total_points = sum(m3_counts.values())
            
            # 排除 -1 后的计数
            valid_m3_counts = {m3_id: count for m3_id, count in m3_counts.items() if m3_id != -1}
            if valid_m3_counts:
                dominant_m3_id = max(valid_m3_counts.items(), key=lambda x: x[1])[0]
                confidence = round(valid_m3_counts[dominant_m3_id] / total_points, 2)  # 修改此行
                if confidence < 0.5:
                    dominant_m3_id = -1
                    confidence = 0.0
            else:
                dominant_m3_id = -1
                confidence = round(m3_counts.get(-1, 0) / total_points, 2) if -1 in m3_counts else 0.0  # 修改此行
            
            gt_to_m3_mapping.append((gt_id, dominant_m3_id, confidence))
        
        return m3_to_gt_mapping, gt_to_m3_mapping

    @time_logger
    def save_relationships(self, m3_to_gt_mapping: List[Tuple[int, int, float]], gt_to_m3_mapping: List[Tuple[int, int, float]], m3_counts: Dict[int, int], gt_count_dict: Dict[int, int]):
        """保存 m3_to_gt 和 gt_to_m3 的对应关系到文件，包含置信度、m3_counts 和 gt_count_dict，并按 ID 升序排序"""
        m3_to_gt_path = os.path.join(self.output_dir, f"scene{self.scene_id}", f"{self.scene_id}_m3_to_gt.pt")
        gt_to_m3_path = os.path.join(self.output_dir, f"scene{self.scene_id}", f"{self.scene_id}_gt_to_m3.pt")
        os.makedirs(os.path.dirname(m3_to_gt_path), exist_ok=True)
        
        # 按第一个元素（m3_id 或 gt_id）升序排序 mapping
        m3_to_gt_mapping_sorted = sorted(m3_to_gt_mapping, key=lambda x: x[0])
        gt_to_m3_mapping_sorted = sorted(gt_to_m3_mapping, key=lambda x: x[0])
        
        # 将 m3_counts 和 gt_count_dict 转换为按键升序的列表
        m3_counts_sorted = sorted(m3_counts.items(), key=lambda x: x[0])  # [(m3_id, count), ...]
        gt_count_dict_sorted = sorted(gt_count_dict.items(), key=lambda x: x[0])  # [(gt_id, count), ...]
        
        # 保存包含映射和计数信息
        torch.save({
            'mapping': m3_to_gt_mapping_sorted,
            'm3_counts': m3_counts_sorted
        }, m3_to_gt_path)
        torch.save({
            'mapping': gt_to_m3_mapping_sorted,
            'gt_count_dict': gt_count_dict_sorted
        }, gt_to_m3_path)
        
        print(f"m3_to_gt 关系及 m3_counts 已保存至 {m3_to_gt_path}")
        print(f"gt_to_m3 关系及 gt_count_dict 已保存至 {gt_to_m3_path}")

    @time_logger
    def save_data(self, data: Dict[int, Dict[str, Any]], output_file: str):
        output_path = os.path.join(self.output_dir, output_file)
        scene_output_path = os.path.join(self.output_dir, f"scene{self.scene_id}", output_file)
        os.makedirs(os.path.dirname(scene_output_path), exist_ok=True)
        
        torch.save(data, output_path)
        torch.save(data, scene_output_path)
        print(f"数据已保存为 {output_path} 和 {scene_output_path}")

# 4. 抽象的顶层调用函数
def process_scene(scene_id: str, config: Dict[str, str], output_dir: str = "."):
    processor = SceneDataProcessor(scene_id, output_dir)
    data = processor.process_data(config)
    processor.save_data(data, config['output_file'])

# 使用示例
if __name__ == "__main__":
    scene_id = '0011_00'
    config = {
        'ply_file': f'scene{scene_id}/scene{scene_id}_vh_clean_2.ply',
        'spt_file': f'{scene_id}_spt_ids.pt',
        'gt_file': f'{scene_id}_gt_ids.pt',
        'm3_file': f'{scene_id}_m3_ids.pt',
        'output_file': f'{scene_id}_dict.pt'
    }
    
    process_scene(scene_id, config)
    
    m3_to_gt = torch.load('./scene0011_00/0011_00_m3_to_gt.pt')
    gt_to_m3 = torch.load('./scene0011_00/0011_00_gt_to_m3.pt')
    a=1
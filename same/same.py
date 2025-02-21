import torch
import numpy as np
import open3d as o3d

def voxelize_to_target_size(ply_path, target_size):
    """
    将点云体素化到目标点数
    """
    # 读取点云
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    
    # 计算体素大小
    # 通过二分查找找到合适的体素大小，使得体素化后的点数接近target_size
    min_voxel_size = 0.01
    max_voxel_size = 1.0
    
    while min_voxel_size < max_voxel_size:
        voxel_size = (min_voxel_size + max_voxel_size) / 2
        downsampled_pcd = pcd.voxel_down_sample(voxel_size)
        current_size = len(np.asarray(downsampled_pcd.points))
        
        if abs(current_size - target_size) < target_size * 0.01:  # 允许1%的误差
            return downsampled_pcd
        elif current_size > target_size:
            min_voxel_size = voxel_size
        else:
            max_voxel_size = voxel_size
    
    return pcd.voxel_down_sample(voxel_size)

def upsample_labels(labels, original_points, downsampled_points):
    """
    将标签从下采样点扩展到原始点
    使用最近邻插值
    """
    from sklearn.neighbors import NearestNeighbors
    
    # 构建最近邻搜索
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(downsampled_points)
    distances, indices = nbrs.kneighbors(original_points)
    
    # 根据最近邻分配标签
    upsampled_labels = labels[indices.flatten()]
    return upsampled_labels

# 主函数
def process_point_cloud(ply_path, pt_path, mode='voxelize'):
    """
    mode: 'voxelize' 或 'upsample'
    voxelize: 将ply点云体素化到pt文件的点数
    upsample: 将pt文件的标签扩展到ply点云的点数
    """
    # 加载.pt文件
    labels = torch.load(pt_path)
    if labels.is_cuda:
        labels = labels.cpu()
    labels = labels.numpy()
    
    # 读取点云
    pcd = o3d.io.read_point_cloud(ply_path)
    original_points = np.asarray(pcd.points)
    
    if mode == 'voxelize':
        # 体素化到目标点数
        target_size = len(labels)
        downsampled_pcd = voxelize_to_target_size(ply_path, target_size)
        downsampled_points = np.asarray(downsampled_pcd.points)
        print(f"Original points: {len(original_points)}")
        print(f"Downsampled points: {len(downsampled_points)}")
        
        # 保存体素化后的点云
        o3d.io.write_point_cloud("downsampled.ply", downsampled_pcd)
        return downsampled_pcd
        
    elif mode == 'upsample':
        # 先获取对应于标签的点云
        original_labeled_pcd = o3d.io.read_point_cloud(ply_path)
        labeled_points = np.asarray(original_labeled_pcd.points)
        
        # 扩展标签
        upsampled_labels = upsample_labels(labels, original_points, labeled_points)
        
        # 保存扩展后的标签
        torch.save(torch.from_numpy(upsampled_labels), "upsampled_labels.pt")
        print(f"Original labels: {len(labels)}")
        print(f"Upsampled labels: {len(upsampled_labels)}")
        return upsampled_labels

# 使用示例
if __name__ == "__main__":
    ply_file = "scene0011_00_vh_clean_2.ply"  # 替换为你的.ply文件路径
    pt_file = "superpoint.pt"         # 替换为你的.pt文件路径
    
    # 选择模式：'voxelize' 或 'upsample'
    result = process_point_cloud(ply_file, pt_file, mode='voxelize')
import open3d as o3d
import numpy as np

def analyze_ply_header(ply_path):
    with open(ply_path, 'rb') as f:
        # 读取头部信息直到 'end_header'
        header = []
        while True:
            line = f.readline().decode('ascii').strip()
            header.append(line)
            if line == 'end_header':
                break
    return header

def compare_plys(ply1_path, ply2_path):
    print("=== PLY 1 Header ===")
    print('\n'.join(analyze_ply_header(ply1_path)))
    print("\n=== PLY 2 Header ===")
    print('\n'.join(analyze_ply_header(ply2_path)))
    
    # 读取点云数据
    pcd1 = o3d.io.read_point_cloud(ply1_path)
    pcd2 = o3d.io.read_point_cloud(ply2_path)
    
    print("\n=== 点云比较 ===")
    print(f"PLY 1 点数: {len(pcd1.points)}")
    print(f"PLY 2 点数: {len(pcd2.points)}")
    
    # 分析颜色分布
    colors1 = np.asarray(pcd1.colors)
    colors2 = np.asarray(pcd2.colors)
    
    print(f"\nPLY 1 唯一颜色数: {len(np.unique(colors1, axis=0))}")
    print(f"PLY 2 唯一颜色数: {len(np.unique(colors2, axis=0))}")

scene_id = 'scene0011_00'
compare_plys(f'{scene_id}_b.ply', f'{scene_id}_vh_clean_2.labels.ply')
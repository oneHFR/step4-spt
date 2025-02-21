# import open3d as o3d
# import numpy as np
# from xvfbwrapper import Xvfb

# # 创建虚拟显示
# vdisplay = Xvfb()
# vdisplay.start()

# try:
#     # 您的可视化代码
#     pcd = o3d.io.read_point_cloud("scene0011_00_spt.ply")
#     o3d.visualization.draw_geometries([pcd])
# finally:
#     vdisplay.stop()

# import open3d as o3d
# import numpy as np

# def visualize_ply_offscreen(ply_path, output_image="point_cloud.png"):
#     """
#     使用离屏渲染方式可视化点云并保存为图片
    
#     参数:
#     ply_path: str, PLY文件的路径
#     output_image: str, 输出图片的路径
#     """
#     # 读取点云
#     pcd = o3d.io.read_point_cloud(ply_path)
    
#     # 创建可视化器
#     vis = o3d.visualization.Visualizer()
#     vis.create_window(visible=False)  # 创建不可见的窗口
    
#     # 添加几何体
#     vis.add_geometry(pcd)
    
#     # 设置视角
#     ctr = vis.get_view_control()
#     ctr.set_zoom(0.8)
#     ctr.set_front([0, 0, -1])
#     ctr.set_lookat([0, 0, 0])
#     ctr.set_up([0, -1, 0])
    
#     # 渲染
#     vis.poll_events()
#     vis.update_renderer()
    
#     # 捕获图像
#     vis.capture_screen_image(output_image)
    
#     # 销毁窗口
#     vis.destroy_window()
    
#     print(f"Point cloud visualization has been saved to {output_image}")

# if __name__ == "__main__":
#     ply_file = "scene0011_00_spt.ply"  # 替换为您的PLY文件路径
#     visualize_ply_offscreen(ply_file)

import pyvista as pv
import numpy as np
import open3d as o3d

def visualize_ply_pyvista(ply_path, output_image="point_cloud.png"):
    """
    使用PyVista进行点云可视化
    
    参数:
    ply_path: str, PLY文件的路径
    output_image: str, 输出图片的路径
    """
    # 设置离屏渲染
    pv.OFF_SCREEN = True
    
    # 读取点云
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    
    # 创建PyVista点云对象
    cloud = pv.PolyData(points)
    if colors is not None:
        cloud.point_data['RGB'] = colors * 255
    
    # 创建plotter
    plotter = pv.Plotter(off_screen=True)
    
    # 添加点云
    if colors is not None:
        plotter.add_mesh(cloud, scalars='RGB', rgb=True, point_size=2)
    else:
        plotter.add_mesh(cloud, color='white', point_size=2)
    
    # 添加坐标轴
    plotter.add_axes()
    
    # 设置相机位置
    plotter.camera_position = 'iso'
    
    # 保存图像
    plotter.screenshot(output_image)
    
    # 关闭plotter
    plotter.close()
    
    print(f"Point cloud visualization has been saved to {output_image}")
    
    # 返回统计信息
    return {
        'num_points': len(points),
        'has_colors': pcd.has_colors(),
        'bounds': cloud.bounds,
        'center': cloud.center
    }

if __name__ == "__main__":
    # 使用示例
    ply_file = "scene0011_00_spt.ply"  # 替换为您的PLY文件路径
    stats = visualize_ply_pyvista(ply_file)
    
    # 打印点云统计信息
    print("\nPoint Cloud Statistics:")
    print(f"Number of points: {stats['num_points']}")
    print(f"Has colors: {stats['has_colors']}")
    print(f"Bounding box: {stats['bounds']}")
    print(f"Center: {stats['center']}")
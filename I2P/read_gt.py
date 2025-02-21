import torch
import numpy as np
import util
import util_3d
import os
import torch
import numpy as np
import plotly.graph_objects as go
from plyfile import PlyData
from typing import Dict, Any, Union, List, Tuple

dataset = "scannet200"
global DATASET_NAME
global CLASS_LABELS
global VALID_CLASS_IDS
global ID_TO_LABEL
global LABEL_TO_ID
global opt
global HEAD_CATS_SCANNET_200
global COMMON_CATS_SCANNET_200
global TAIL_CATS_SCANNET_200

if dataset == "scannet200":
    DATASET_NAME = "scannet200"
    CLASS_LABELS = (
        'chair', 'table', 'door', 'couch', 'cabinet', 'shelf', 'desk', 'office chair', 'bed', 'pillow', 'sink',
        'picture', 'window', 'toilet', 'bookshelf', 'monitor', 'curtain', 'book', 'armchair', 'coffee table', 'box',
        'refrigerator', 'lamp', 'kitchen cabinet', 'towel', 'clothes', 'tv', 'nightstand', 'counter', 'dresser',
        'stool', 'cushion', 'plant', 'ceiling', 'bathtub', 'end table', 'dining table', 'keyboard', 'bag',
        'backpack', 'toilet paper',
        'printer', 'tv stand', 'whiteboard', 'blanket', 'shower curtain', 'trash can', 'closet', 'stairs',
        'microwave', 'stove', 'shoe', 'computer tower', 'bottle', 'bin', 'ottoman', 'bench', 'board',
        'washing machine', 'mirror', 'copier',
        'basket', 'sofa chair', 'file cabinet', 'fan', 'laptop', 'shower', 'paper', 'person',
        'paper towel dispenser', 'oven', 'blinds', 'rack', 'plate', 'blackboard', 'piano', 'suitcase', 'rail',
        'radiator', 'recycling bin', 'container',
        'wardrobe', 'soap dispenser', 'telephone', 'bucket', 'clock', 'stand', 'light', 'laundry basket', 'pipe',
        'clothes dryer', 'guitar', 'toilet paper holder', 'seat', 'speaker', 'column', 'bicycle', 'ladder',
        'bathroom stall', 'shower wall',
        'cup', 'jacket', 'storage bin', 'coffee maker', 'dishwasher', 'paper towel roll', 'machine', 'mat',
        'windowsill', 'bar', 'toaster', 'bulletin board', 'ironing board', 'fireplace', 'soap dish',
        'kitchen counter', 'doorframe',
        'toilet paper dispenser', 'mini fridge', 'fire extinguisher', 'ball', 'hat', 'shower curtain rod',
        'water cooler', 'paper cutter', 'tray', 'shower door', 'pillar', 'ledge', 'toaster oven', 'mouse',
        'toilet seat cover dispenser',
        'furniture', 'cart', 'storage container', 'scale', 'tissue box', 'light switch', 'crate', 'power outlet',
        'decoration', 'sign', 'projector', 'closet door', 'vacuum cleaner', 'candle', 'plunger', 'stuffed animal',
        'headphones', 'dish rack',
        'broom', 'guitar case', 'range hood', 'dustpan', 'hair dryer', 'water bottle', 'handicap bar', 'purse',
        'vent', 'shower floor', 'water pitcher', 'mailbox', 'bowl', 'paper bag', 'alarm clock', 'music stand',
        'projector screen', 'divider',
        'laundry detergent', 'bathroom counter', 'object', 'bathroom vanity', 'closet wall', 'laundry hamper',
        'bathroom stall door', 'ceiling light', 'trash bin', 'dumbbell', 'stair rail', 'tube', 'bathroom cabinet',
        'cd case', 'closet rod',
        'coffee kettle', 'structure', 'shower head', 'keyboard piano', 'case of water bottles', 'coat rack',
        'storage organizer', 'folded chair', 'fire alarm', 'power strip', 'calendar', 'poster', 'potted plant',
        'luggage', 'mattress')

    VALID_CLASS_IDS = np.array((2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28,
                                29, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52,
                                54, 55, 56, 57, 58, 59, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
                                72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 84, 86, 87, 88, 89, 90, 93, 95, 96, 97, 98,
                                99, 100, 101, 102, 103, 104, 105, 106, 107, 110, 112, 115, 116, 118, 120, 121, 122,
                                125, 128, 130, 131, 132, 134, 136, 138, 139, 140, 141, 145, 148, 154,
                                155, 156, 157, 159, 161, 163, 165, 166, 168, 169, 170, 177, 180, 185, 188, 191, 193,
                                195, 202, 208, 213, 214, 221, 229, 230, 232, 233, 242, 250, 261, 264, 276, 283, 286,
                                300, 304, 312, 323, 325, 331, 342, 356, 370, 392, 395, 399, 408, 417,
                                488, 540, 562, 570, 572, 581, 609, 748, 776, 1156, 1163, 1164, 1165, 1166, 1167,
                                1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1178, 1179, 1180, 1181, 1182,
                                1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191))

    ID_TO_LABEL = {}
    LABEL_TO_ID = {}
    for i in range(len(VALID_CLASS_IDS)):
        LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
        ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]

    
    HEAD_CATS_SCANNET_200 = set(['tv stand', 'curtain', 'blinds', 'shower curtain', 'bookshelf', 'tv', 'kitchen cabinet', 
                            'pillow', 'lamp', 'dresser', 'monitor', 'object', 'ceiling', 'board', 'stove', 
                            'closet wall', 'couch', 'office chair', 'kitchen counter', 'shower', 'closet', 
                            'doorframe', 'sofa chair', 'mailbox', 'nightstand', 'washing machine', 'picture', 
                            'book', 'sink', 'recycling bin', 'table', 'backpack', 'shower wall', 'toilet', 
                            'copier', 'counter', 'stool', 'refrigerator', 'window', 'file cabinet', 'chair', 
                            'wall', 'plant', 'coffee table', 'stairs', 'armchair', 'cabinet', 'bathroom vanity', 
                            'bathroom stall', 'mirror', 'blackboard', 'trash can', 'stair rail', 'box', 'towel', 
                            'door', 'clothes', 'whiteboard', 'bed', 'floor', 'bathtub', 'desk', 'wardrobe', 
                            'clothes dryer', 'radiator', 'shelf'])

    COMMON_CATS_SCANNET_200 = set(["cushion", "end table", "dining table", "keyboard", "bag", "toilet paper", "printer", 
                            "blanket", "microwave", "shoe", "computer tower", "bottle", "bin", "ottoman", "bench", 
                            "basket", "fan", "laptop", "person", "paper towel dispenser", "oven", "rack", "piano", 
                            "suitcase", "rail", "container", "telephone", "stand", "light", "laundry basket", 
                            "pipe", "seat", "column", "bicycle", "ladder", "jacket", "storage bin", "coffee maker", 
                            "dishwasher", "machine", "mat", "windowsill", "bulletin board", "fireplace", "mini fridge", 
                            "water cooler", "shower door", "pillar", "ledge", "furniture", "cart", "decoration", 
                            "closet door", "vacuum cleaner", "dish rack", "range hood", "projector screen", "divider", 
                            "bathroom counter", "laundry hamper", "bathroom stall door", "ceiling light", "trash bin", 
                            "bathroom cabinet", "structure", "storage organizer", "potted plant", "mattress"])
                            
    TAIL_CATS_SCANNET_200 = set(["paper", "plate", "soap dispenser", "bucket", "clock", "guitar", "toilet paper holder", 
                            "speaker", "cup", "paper towel roll", "bar", "toaster", "ironing board", "soap dish", 
                            "toilet paper dispenser", "fire extinguisher", "ball", "hat", "shower curtain rod", 
                            "paper cutter", "tray", "toaster oven", "mouse", "toilet seat cover dispenser", 
                            "storage container", "scale", "tissue box", "light switch", "crate", "power outlet", 
                            "sign", "projector", "candle", "plunger", "stuffed animal", "headphones", "broom", 
                            "guitar case", "dustpan", "hair dryer", "water bottle", "handicap bar", "purse", "vent", 
                            "shower floor", "water pitcher", "bowl", "paper bag", "alarm clock", "music stand", 
                            "laundry detergent", "dumbbell", "tube", "cd case", "closet rod", "coffee kettle", 
                            "shower head", "keyboard piano", "case of water bottles", "coat rack", "folded chair", 
                            "fire alarm", "power strip", "calendar", "poster", "luggage"])
    
NUM_CLASSES = len(VALID_CLASS_IDS)

def analyze_instance_filtering(gt_ids, save_dir):
    # 1. 获取所有唯一的实例ID
    all_instance_ids = np.unique(gt_ids)
    info_lines = [f"总共唯一实例ID数量: {len(all_instance_ids)}"]
    
    # 2. 排除0后的实例ID
    non_zero_instances = all_instance_ids[all_instance_ids != 0]
    info_lines.append(f"排除0后的实例ID数量: {len(non_zero_instances)}")
    
    # 3. 检查每个实例ID对应的label_id
    invalid_instances = []
    for inst_id in non_zero_instances:
        label_id = inst_id // 1000
        if label_id not in VALID_CLASS_IDS:
            info_lines.append(f"实例ID {inst_id} 对应的标签ID {label_id} 不在有效类别列表中")
            invalid_instances.append(inst_id)
            
    # 4. 统计有效的实例
    valid_instances = [inst_id for inst_id in non_zero_instances 
                      if (inst_id // 1000) in VALID_CLASS_IDS]
    info_lines.append(f"最终有效实例数量: {len(valid_instances)}")
    
    # 输出到终端
    for line in info_lines:
        print(line)
    
    # 保存到 info.txt
    os.makedirs(save_dir, exist_ok=True)
    info_file = os.path.join(save_dir, "info.txt")
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(info_lines))
    print(f"分析信息已保存至 {info_file}")
    
    return valid_instances, invalid_instances

def save_instance_info_with_masks_as_pt(gt_file, save_dir, class_ids, class_labels, id_to_label):
    try:
        # 加载ground truth数据中的ID
        gt_ids = util_3d.load_ids(gt_file)
    except Exception as e:
        util.print_error('unable to load ' + gt_file + ': ' + str(e))

    # 分析有效和无效实例
    valid_instances, invalid_instances = analyze_instance_filtering(gt_ids, save_dir)
    invalid_instances_set = set(invalid_instances)
    all_instance_ids = np.unique(gt_ids)  # 获取所有实例ID，包括 0

    # 获取ground truth中的实例信息（仅用于有效实例）
    gt_instances = util_3d.get_instances(gt_ids, class_ids, class_labels, id_to_label)
    
    # 保存有效实例
    saved_instances = set()
    for class_label, instances in gt_instances.items():
        for idx, instance in enumerate(instances):
            instance_id = instance['instance_id']
            instance_mask = np.equal(gt_ids, instance_id).astype(np.uint8)

            instance_info = {
                'instance_id': instance_id,
                'label_id': instance['label_id'],
                'vert_count': instance['vert_count'],
                'med_dist': instance.get('med_dist', -1),
                'dist_conf': instance.get('dist_conf', 0.0),
                'mask': torch.tensor(instance_mask)
            }

            suffix = "_invalid" if instance_id in invalid_instances_set else ""
            save_path = os.path.join(save_dir, f"{instance_id}_{class_label}_{idx}{suffix}.pt")
            torch.save(instance_info, save_path)
            print(f"Saved instance {instance_id} info and mask to {save_path}")
            saved_instances.add(instance_id)

    # 保存所有实例（包括无效实例和 0）
    for instance_id in all_instance_ids:
        if instance_id not in saved_instances:
            label_id = instance_id // 1000 if instance_id != 0 else 0  # 实例 ID 为 0 时 label_id 设为 0
            instance_mask = np.equal(gt_ids, instance_id).astype(np.uint8)
            vert_count = np.sum(instance_mask)

            instance_info = {
                'instance_id': instance_id,
                'label_id': label_id,
                'vert_count': vert_count,
                'med_dist': -1,  # 无 med_dist 信息
                'dist_conf': 0.0,  # 无 dist_conf 信息
                'mask': torch.tensor(instance_mask)
            }

            suffix = "_invalid" if instance_id in invalid_instances_set else ""
            save_path = os.path.join(save_dir, f"{instance_id}_unknown_0{suffix}.pt")
            torch.save(instance_info, save_path)
            print(f"Saved instance {instance_id} info and mask to {save_path}")


def save_instance_ids_as_pt(gt_file, save_dir):
    try:
        # 加载ground truth数据中的ID
        gt_ids = util_3d.load_ids(gt_file)
    except Exception as e:
        util.print_error('unable to load ' + gt_file + ': ' + str(e))

    # 创建保存路径
    save_path = os.path.join(save_dir, "0011_00_instance_ids.pt")
    
    # 保存每个点的instance_id为.pt文件
    torch.save(torch.tensor(gt_ids), save_path)

    print(f"Saved instance IDs to {save_path}")


def analyze_instance_ids(file_path):
    # 加载.pt文件中的数据
    instance_ids = torch.load(file_path)
    
    # 确保数据是一个一维张量
    assert instance_ids.ndimension() == 1, "The tensor should have shape (N,)"
    
    # 计算每个数字的重复次数和占比
    unique_values, counts = np.unique(instance_ids.numpy(), return_counts=True)
    
    # 计算每个数字的占比
    proportions = counts / len(instance_ids) * 100
    
    # 输出结果
    print("Value Distribution:")
    for value, count, proportion in zip(unique_values, counts, proportions):
        print(f"Value: {value}, Count: {count}, Proportion: {proportion:.2f}%")
       
def process_scene(scene_id: str, config: Dict[str, str], output_dir: str = "."):
    gt_file = config['gt_file']
    gt_masks_dir = os.path.join(output_dir, f"scene{scene_id}", "gt_masks")
    
    # 加载 gt_ids 并分析实例过滤情况
    gt_ids = util_3d.load_ids(gt_file)
    valid_instances, _ = analyze_instance_filtering(gt_ids, gt_masks_dir)
    
    # 保存实例信息和掩码
    save_instance_info_with_masks_as_pt(gt_file, gt_masks_dir, VALID_CLASS_IDS, CLASS_LABELS, ID_TO_LABEL)
    save_instance_ids_as_pt(gt_file, gt_masks_dir)

def batch_process_scenes(scene_ids: List[str], base_config: Dict[str, str], output_dir: str = "."):
    for scene_id in scene_ids:
        config = {
            'gt_file': f"scene{scene_id}/gt_scene{scene_id}.txt",
            # 可根据需要扩展其他配置项
        }
        print(f"Processing scene {scene_id}...")
        process_scene(scene_id, config, output_dir)

# 使用示例
if __name__ == "__main__":
    scene_ids = ['0011_00']  # 示例多个 scene_id
    base_config = {}  # 如果有其他通用配置可在此定义
    batch_process_scenes(scene_ids, base_config)

# gt_file = "scene0011_00/gt_scene0011_00.txt"
# save_dir = "scene0011_00/gt_masks"
# gt_masks_dir = "scene0011_00/gt_masks"  # 用于保存 info.txt

# # 在保存实例信息之前，先分析一下实例ID的过滤情况
# gt_ids = util_3d.load_ids(gt_file)
# valid_instances, _ = analyze_instance_filtering(gt_ids, gt_masks_dir)  # 修改调用，传入 gt_masks_dir
# save_instance_info_with_masks_as_pt(gt_file, gt_masks_dir, VALID_CLASS_IDS, CLASS_LABELS, ID_TO_LABEL)  # 修改保存路径为 gt_masks_dir
# save_instance_ids_as_pt(gt_file, save_dir)
# # instance_file = "scene0011_00/gt_masks/0011_00_instance_ids.pt"









# 加载并打印实例信息
# instance_info = torch.load(instance_file)
# print(instance_info['mask'].shape)
# print(min(instance_info))

# analyze_instance_ids(instance_file)




















# # 1. 加载 .ply 文件，提取坐标
# def load_ply(ply_file):
#     ply_data = PlyData.read(ply_file)
#     vertices = ply_data['vertex'].data
#     points = np.array([vertices['x'], vertices['y'], vertices['z']]).T
#     return points

# # 2. 加载分类结果的掩码 (pt 文件)，它包含每个点的类别
# def load_mask(mask_file):
#     mask = torch.load(mask_file)  # 假设 mask 是保存为 .pt 格式的张量
#     return mask.numpy()  # 转换为 numpy 数组

# # 3. 可视化点云
# def visualize_point_cloud(points, mask, output_file='gt_0011_00_instance_id.html'):
#     # 使用不同颜色标识不同的物体类别
#     unique_labels = np.unique(mask)
    
#     # 生成每个类别的颜色 (你可以自定义颜色)
#     color_map = {label: f'rgb({np.random.randint(0, 255)}, {np.random.randint(0, 255)}, {np.random.randint(0, 255)})' for label in unique_labels}
    
#     # 创建一个 Plotly 3D 散点图
#     fig = go.Figure()
    
#     # 按类别分配不同颜色
#     for label in unique_labels:
#         points_of_label = points[mask == label]
#         fig.add_trace(go.Scatter3d(
#             x=points_of_label[:, 0],
#             y=points_of_label[:, 1],
#             z=points_of_label[:, 2],
#             mode='markers',
#             marker=dict(size=3, color=color_map[label], opacity=0.7),
#             name=f'Object {label}'  # 标记每个类别
#         ))
    
#     # 设置图表布局
#     fig.update_layout(
#         title='3D Point Cloud Visualization by Object',
#         scene=dict(
#             xaxis_title='X Coordinate',
#             yaxis_title='Y Coordinate',
#             zaxis_title='Z Coordinate'
#         ),
#         showlegend=True,
#         paper_bgcolor='rgb(20, 24, 35)',  # 设置背景色
#         font=dict(color='white'),  # 设置字体颜色
#         margin=dict(l=0, r=0, t=30, b=0)  # 调整边距
#     )

#     # 保存为 HTML 文件
#     fig.write_html(output_file)
#     print(f"可视化已保存为 {output_file}")

# # 示例使用
# ply_file = 'scene0011_00/scene0011_00_vh_clean_2.ply'  # 替换为你的 .ply 文件路径
# mask_file = 'output_masks/3.pt'  # 替换为你的 .pt 掩码文件路径
# instance_file = "gt_output/0011_00_instance_ids.pt"

# # 加载数据
# points = load_ply(ply_file)
# mask = load_mask(instance_file)

# # 可视化点云
# # visualize_point_cloud(points, mask)




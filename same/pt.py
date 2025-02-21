

import torch
from collections import Counter


# 这里假设 model_state_dict 是包含 (165209,) 大小的张量
model_state_dict = torch.load("cluster_labels.pt")  # 替换为你的文件路径

# 假设你正在查看的 tensor 是某个层的参数（例如第一层的权重或偏置）
param_tensor = model_state_dict  # 替换为你实际的参数名

# 检查 param_tensor 的类型，如果是稀疏张量，则转换为密集张量
if param_tensor.is_sparse:
    param_tensor = param_tensor.to_dense()

# 直接统计重复的元素
unique_values, counts = torch.unique(param_tensor, return_counts=True)

# 将重复的元素过滤出来
duplicates = {unique_values[i].item(): counts[i].item() for i in range(len(unique_values)) if counts[i] > 1}

# 打印重复的元素及其出现次数
print("重复的元素及其出现次数：")
for value, count in duplicates.items():
    print(f"值: {value}, 重复次数: {count}")

# 计算重复数据的占比
total_elements = param_tensor.numel()  # 获取所有元素的总数
duplicate_elements = sum(count for count in duplicates.values())  # 计算重复元素的总数

duplicate_ratio = duplicate_elements / total_elements  # 重复元素的占比

print(f"\n重复元素的占比: {duplicate_ratio:.4f} ({duplicate_elements}/{total_elements})")

import torch
import numpy as np
from collections import Counter

# 加载文件
# labels = torch.load('superpoint.pt')
# labels = torch.load('superpoint_L1.pt')
labels = torch.load('superpoint_L0.pt')

# 基本信息
print("Labels shape:", labels.shape)
print("Data type:", labels.dtype)
print("Unique values count:", len(torch.unique(labels)))

# 值的范围
print("\nValue range:")
print("Min value:", labels.min().item())
print("Max value:", labels.max().item())

# 将张量移到CPU并转换为numpy
labels_cpu = labels.cpu()
values_count = Counter(labels_cpu.numpy().flatten())
top_10_common = values_count.most_common(10)

# 显示最常见的10个值
print("\nTop 10 most common values and their counts:")
for value, count in top_10_common:
    print(f"Value {value}: {count} occurrences ({count/len(labels)*100:.2f}%)")

# 基本统计量
print("\nBasic statistics:")
print("Mean:", labels.float().mean().item())
print("Median:", torch.median(labels.float()).item())
print("Std:", labels.float().std().item())
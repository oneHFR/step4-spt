import torch
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

import torch

# 1. 加载输入的 .pt 文件
input_file = 'scene0011_00/0011_00_m3_masks.pt'
masks = torch.load(input_file)  # 假设文件包含一个大小为 (237360, 165) 的二进制数组

labels = []
for mask in masks:
    # 找到值为1的索引
    indices = (mask == 1).nonzero()  # 获取非零的索引
    
    # 检查返回的是否为空
    if len(indices[0]) > 0:  # 如果索引非空
        indices_list = indices[0].tolist()  # 提取索引，并转换为list
    else:
        indices_list = [-1]  # 如果没有1，标签设为-1
    
    labels.append(indices_list)

# 3. 保存结果为 .pt 文件
output_file = '0011_00_m3_ids.pt'
torch.save(labels, output_file)

# 4. 加载保存的文件并打印结果
loaded_labels = torch.load(output_file)
print(loaded_labels)

import torch

# 加载 .pth 文件
file_path = 'scene0011_00.pth'

# 如果文件保存的是模型的 state_dict
model_state_dict = torch.load(file_path)

# 如果文件保存的是整个模型
# model = torch.load(file_path)

# 打印出加载的内容（通常是字典）
print(model_state_dict.shape)

# 如果你保存了其他对象，比如训练的参数或其他数据
# 你可以根据保存的内容进行访问

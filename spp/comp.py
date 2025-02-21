import torch

def compare_pt_files(file1_path, file2_path):
    # 加载两个文件
    tensor1 = torch.load(file1_path)
    tensor2 = torch.load(file2_path)

    # 检查是否都在GPU上，如果是则转到CPU
    if tensor1.is_cuda:
        tensor1 = tensor1.cpu()
    if tensor2.is_cuda:
        tensor2 = tensor2.cpu()

    # 比较基本信息
    print(f"Shape comparison: {'Same' if tensor1.shape == tensor2.shape else 'Different'}")
    print(f"File 1 shape: {tensor1.shape}")
    print(f"File 2 shape: {tensor2.shape}")
    
    print(f"\nData type comparison: {'Same' if tensor1.dtype == tensor2.dtype else 'Different'}")
    print(f"File 1 dtype: {tensor1.dtype}")
    print(f"File 2 dtype: {tensor2.dtype}")

    # 检查值是否完全相同
    if tensor1.shape == tensor2.shape:
        is_identical = torch.all(tensor1 == tensor2).item()
        print(f"\nValues comparison: {'Identical' if is_identical else 'Different'}")
        
        if not is_identical:
            # 找出不同值的位置和数量
            differences = torch.where(tensor1 != tensor2)
            num_differences = len(differences[0])
            print(f"Number of different values: {num_differences}")
            
            # 显示前5个不同的位置和对应的值
            if num_differences > 0:
                print("\nFirst 5 differences:")
                for i in range(min(5, num_differences)):
                    idx = tuple(diff[i] for diff in differences)
                    print(f"Position {idx}:")
                    print(f"  File 1 value: {tensor1[idx].item()}")
                    print(f"  File 2 value: {tensor2[idx].item()}")
    else:
        print("\nCannot compare values due to different shapes")

# 使用示例
if __name__ == "__main__":
    file1_path = "superpoint_L0.pt"  # 替换为第二个文件的路径
    file2_path = "superpoint_L1.pt"  # 替换为第二个文件的路径
    
    print("Comparing PT files...")
    print("-" * 50)
    compare_pt_files(file1_path, file2_path)
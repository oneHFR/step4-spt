import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def apply_mask_to_image(image_path, mask_path, output_path=None, alpha=0.3, color=[0, 0, 1]):
    """
    将.npy格式的掩码叠加到图片上并显示/保存结果
    
    参数:
        image_path: 原始图片路径
        mask_path: .npy掩码文件路径
        output_path: 输出图片保存路径，如果为None则只显示不保存
        alpha: 掩码透明度，范围0-1
        color: 掩码颜色 [R, G, B]，范围0-1
    """
    # 加载图片
    image = Image.open(image_path)
    image_np = np.array(image) / 255.0
    
    # 加载掩码
    mask_np = np.load(mask_path)
    
    # 确保掩码是二维的
    if len(mask_np.shape) > 2:
        mask_np = mask_np.squeeze()
    
    # 确保掩码和图片尺寸一致
    if mask_np.shape[:2] != image_np.shape[:2]:
        raise ValueError(f"掩码尺寸 {mask_np.shape[:2]} 与图片尺寸 {image_np.shape[:2]} 不匹配")
    
    # 创建彩色掩码层
    mask_colored = np.zeros_like(image_np)
    for c in range(min(3, mask_colored.shape[2])):
        mask_colored[:, :, c] = mask_np * color[c]
    
    # 叠加掩码到图片上
    blended = image_np * (1 - mask_np[:, :, None] * alpha) + mask_colored * (mask_np[:, :, None] * alpha)
    
    # 显示结果
    plt.figure(figsize=(10, 8))
    plt.imshow(blended)
    plt.axis('off')
    
    if output_path:
        # 保存为图片，确保值在0-255范围内
        result_image = Image.fromarray((blended * 255).astype(np.uint8))
        result_image.save(output_path)
        print(f"结果已保存至 {output_path}")
    
    plt.show()
    
    return blended

apply_mask_to_image('1520.jpg', '3_152_best_mask.npy', '152_mask.png')
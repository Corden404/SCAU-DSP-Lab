import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter

# 1. 准备图片
img_pil = Image.open('exp1/skin-smoothing_demo/test_img.jpg').convert('L')
img = np.array(img_pil).astype(float) / 255.0

# 步骤 1: 频率分离
# Sigma=0.8: 此时低频层痘痘清晰，高频层很干净
split_sigma = 0.8
img_low = gaussian_filter(img, sigma=split_sigma)
img_high = img - img_low

# 步骤 2: 加载手动蒙版
try:
    # 尝试加载手动绘制的蒙版
    mask_pil = Image.open('exp1/skin-smoothing_demo/mask.png').convert('L')
    
    # 强制调整蒙版尺寸以匹配原图。我提供的示例蒙版尺寸是 512x512，理论上与输入图像一致。
    if mask_pil.size != img_pil.size:
        print(f"蒙版尺寸 {mask_pil.size} 与原图 {img_pil.size} 不一致，已自动缩放。")
        mask_pil = mask_pil.resize(img_pil.size, Image.Resampling.LANCZOS)
    
    mask = np.array(mask_pil).astype(float) / 255.0

except FileNotFoundError:
    print("未找到 'exp1/skin-smoothing_demo/mask.png'")
    mask = np.zeros_like(img)

# 对蒙版做一点轻微的高斯模糊，保证黑白交界处过渡自然
mask_soft = gaussian_filter(mask, sigma=2.0)

# 步骤 3: 对低频层进行选择性磨皮
# 1. 生成磨皮层
# 既然有蒙版保护五官，可以放心大胆地用大核磨皮
img_low_retouched_base = median_filter(img_low, size=30) 

# 2. 融合：
# mask=1 (白色/五官) -> 使用原始 img_low (清晰)
# mask=0 (黑色/皮肤) -> 使用 img_low_retouched_base (磨皮)
img_low_final = img_low * mask_soft + img_low_retouched_base * (1 - mask_soft)

# 步骤 4: 合成
img_restored = img_low_final + img_high

# 步骤 5:可视化
plt.figure(figsize=(14, 8))

plt.subplot(2, 3, 1)
plt.imshow(img_low, cmap='gray')
plt.title(f'Original Low Freq (Sigma={split_sigma})')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(img_high, cmap='gray')
plt.title('High Freq (Texture Only)')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(mask_soft, cmap='gray')
plt.title('Manual Mask\n(White=Keep Sharp, Black=Smooth)')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(img_low_retouched_base, cmap='gray')
plt.title('Heavy Blur Low Freq')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(img_low_final, cmap='gray')
plt.title('Masked Low Freq')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(img_restored, cmap='gray')
plt.title('Final Result')
plt.axis('off')

plt.tight_layout()
plt.savefig('exp1/skin-smoothing_demo/skin_smoothing_result.png', dpi=300)
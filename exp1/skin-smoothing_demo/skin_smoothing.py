import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter

def load_image(path, size=None):
    """读取图片并归一化"""
    img = Image.open(path).convert('L')
    if size:
        img = img.resize(size, Image.Resampling.LANCZOS)
    return np.array(img).astype(float) / 255.0

# 1. 准备数据
img_path = 'exp1/skin-smoothing_demo/test_img.jpg'
mask_path = 'exp1/skin-smoothing_demo/mask.png'

# 加载原图
img = load_image(img_path)
h, w = img.shape

# 加载蒙版
try:
    mask_pil = Image.open(mask_path).convert('L').resize((w, h))
    mask = np.array(mask_pil).astype(float) / 255.0
except FileNotFoundError:
    print("未找到蒙版文件，请检查路径。")
    mask = np.ones_like(img) # 如果没蒙版，全白(不处理)

# 蒙版柔化：虽然蒙版是精准的，但在像素级别上，稍微一点点羽化(sigma=1.0)能让修补的边缘完全融合，看不出拼接痕迹
mask_soft = gaussian_filter(mask, sigma=1.5)

# 参数设置
# sigma_texture (小)
sigma_texture = 0.8

# sigma_structure (大)
sigma_structure = 8.0

# 提取纯净的高频纹理
blur_small = gaussian_filter(img, sigma=sigma_texture)
img_high = img - blur_small

# 提取平滑的低频轮廓
img_low = gaussian_filter(img, sigma=sigma_structure)

# 合成Patch = 高频 + 低频
img_patch = img_high + img_low

# 蒙版融合
img_final = img * mask_soft + img_patch * (1 - mask_soft)

# 3. 可视化展示
plt.figure(figsize=(18, 10))

# 1. 原图
plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# 2. 蒙版
plt.subplot(2, 3, 2)
plt.imshow(mask, cmap='gray')
plt.title('Manual Mask\n(Black = Acne Spots)')
plt.axis('off')

# 3. 高频层 (纹理)
plt.subplot(2, 3, 3)
plt.imshow(img_high, cmap='gray')
plt.title(f'High Freq (Texture)\n$\sigma$={sigma_texture}')
plt.axis('off')

# 4. 低频层 (结构)
plt.subplot(2, 3, 4)
plt.imshow(img_low, cmap='gray')
plt.title(f'Low Freq (Structure)\n$\sigma$={sigma_structure}')
plt.axis('off')

# 5. 补丁图
plt.subplot(2, 3, 5)
plt.imshow(img_patch, cmap='gray')
plt.title('Synthesized Patch\n(High + Low)')
plt.axis('off')

# 6. 最终结果
plt.subplot(2, 3, 6)
plt.imshow(img_final, cmap='gray')
plt.title('Final Result\n(Band-Stop Filtered)')
plt.axis('off')

plt.tight_layout()
plt.savefig('exp1/skin-smoothing_demo/skin_smoothing_result.png', dpi=300)
plt.show()
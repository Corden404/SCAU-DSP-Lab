import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

def load_and_prep(img_path, mask_path):
    """读取图片和蒙版，并进行归一化预处理"""
    # 1. 读取原图
    img_pil = Image.open(img_path).convert('L')
    img = np.array(img_pil).astype(float) / 255.0
    
    # 2. 读取蒙版 (强制缩放以匹配原图)
    try:
        mask_pil = Image.open(mask_path).convert('L').resize(img_pil.size)
        mask = np.array(mask_pil).astype(float) / 255.0
    except FileNotFoundError:
        print("未找到蒙版。")
        mask = np.ones_like(img)
        
    return img, mask

# 实验设置
img_path = 'exp1/skin-smoothing_demo/test_img.jpg'
mask_path = 'exp1/skin-smoothing_demo/mask.png'

img, mask = load_and_prep(img_path, mask_path)
h, w = img.shape

# 预处理蒙版：轻微柔化，保证融合边界自然
mask_soft = gaussian_filter(mask, sigma=1.0)

# 方法 1: 简单高斯模糊
img_method_1 = gaussian_filter(img, sigma=3.0)


# 方法 2: 单 Sigma 频率分离 (传统方法)
sigma_single = 2.0

# 1. 分离
low_freq = gaussian_filter(img, sigma=sigma_single)
high_freq = img - low_freq

# 2. 磨皮 (只处理低频)
# 对低频层做中值滤波，模拟常规磨皮
low_freq_blur = median_filter(low_freq, size=20)

# 3. 融合 (利用蒙版只替换低频部分)
low_freq_final = low_freq * mask_soft + low_freq_blur * (1 - mask_soft)

# 4. 重构
img_method_2 = low_freq_final + high_freq

# 方法 3: 双 Sigma 带阻合成

# 参数调优结果
sigma_tex = 0.8   # 小 Sigma: 紧贴毛孔，不包痘痘
sigma_struct = 8.0 # 大 Sigma: 只要大轮廓，不要痘痘阴影

# 1. 提取纯净高频
img_high_pure = img - gaussian_filter(img, sigma=sigma_tex)

# 2. 提取平滑低频
img_low_pure = gaussian_filter(img, sigma=sigma_struct)

# 3. 合成无痘补丁
img_patch = img_high_pure + img_low_pure

# 4. 精准替换
img_method_3 = img * mask_soft + img_patch * (1 - mask_soft)


# 可视化对比展示
plt.figure(figsize=(20, 6))

# 1. 原图
plt.subplot(1, 4, 1)
plt.imshow(img, cmap='gray')
plt.title('1. 原始图像\n(输入信号)', fontsize=12)
plt.axis('off')

# 2. 简单模糊
plt.subplot(1, 4, 2)
plt.imshow(img_method_1, cmap='gray')
plt.title('2. 简单高斯模糊\n(结果: 模糊，细节丢失)', fontsize=12)
plt.axis('off')

# 3. 单 Sigma 分离
plt.subplot(1, 4, 3)
plt.imshow(img_method_2, cmap='gray')
plt.title(f'3. 单 Sigma 频率分离\n(Sigma={sigma_single})\n(结果: 保留细节，但痘印也被部分保留)', fontsize=12)
plt.axis('off')

# 4. 双 Sigma 合成 
plt.subplot(1, 4, 4)
plt.imshow(img_method_3, cmap='gray')
plt.title(f'4. 双 Sigma 带阻合成 \n(Tex $\\sigma$={sigma_tex}, Struct $\\sigma$={sigma_struct}) \n(结果: 效果最好)', fontsize=12, fontweight='bold')
plt.axis('off')

plt.tight_layout()
plt.savefig('exp1/skin-smoothing_demo/comparison_result.png', dpi=300)
plt.show()
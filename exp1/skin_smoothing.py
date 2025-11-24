import numpy as np
import matplotlib.pyplot as plt
import scipy.datasets as datasets # 或者 import scipy.misc as misc
from scipy.ndimage import gaussian_filter
import os

# 1. 准备图像 (转化为二维信号)
# 加载本地图片
img_path = os.path.join(os.path.dirname(__file__), 'test_img.jpg')
img = plt.imread(img_path)

# 如果是彩色图(3维)，转为灰度图(2维)
if img.ndim == 3:
    img = np.dot(img[...,:3], [0.299, 0.587, 0.114])

img = img.astype(float)

# 归一化到 0-1 之间，方便计算 (PS里对应 0-255，原理一样)
if img.max() > 1.0:
    img = img / 255.0

# 2. 提取低频 (Low Frequency) - 对应 PS 的“高斯模糊”
# sigma 控制模糊程度，相当于 PS 里的半径像素
# sigma 越大，保留的细节越少，低频越“低”
sigma = 3.0
img_low = gaussian_filter(img, sigma=sigma)

# 3. 提取高频 (High Frequency) - 对应 PS 的“应用图像(减去)”
# 高频 = 原图 - 低频
# 注意：高频层会有负数（因为原图可能比模糊后的暗），
# 在 PS 里通常会 +128 变成灰色底，这里我们保持数学上的精确值
img_high = img - img_low

# ==========================================
# 4. 模拟“磨皮”操作 (Retouching Simulation)
# ==========================================
# 原理：我们只对“低频层”进行进一步处理（比如更强的模糊以均匀肤色），
# 但保持“高频层”（纹理、毛孔、边缘）不变。
# 这样皮肤看起来光滑，但不会像塑料人，因为纹理还在。

# 对低频层再次模糊，模拟“均匀肤色”
img_low_retouched = gaussian_filter(img_low, sigma=10.0)

# 重组图像：处理过的低频 + 原始高频
img_restored = img_low_retouched + img_high

# ==========================================
# 5. 可视化结果
# ==========================================
plt.figure(figsize=(12, 10))

# 原图
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('1. Original (f(x,y))')
plt.axis('off')

# 低频 (模糊，光影，肤色基调)
plt.subplot(2, 2, 2)
plt.imshow(img_low, cmap='gray')
plt.title(f'2. Low Frequency (Gaussian sigma={sigma})')
plt.axis('off')

# 高频 (纹理，边缘，痘痘)
# 也就是 PS 里的“高反差保留”或灰色层
# 为了显示清楚，我加了 0.5 (相当于 PS 的 +128 灰色)，否则负数显示全黑
plt.subplot(2, 2, 3)
plt.imshow(img_high + 0.5, cmap='gray', vmin=0, vmax=1)
plt.title('3. High Frequency (Texture/Edges)')
plt.axis('off')

# 磨皮后重组
plt.subplot(2, 2, 4)
plt.imshow(img_restored, cmap='gray')
plt.title('4. Retouched (Smoothed Low + Orig High)')
plt.axis('off')

plt.tight_layout()
plt.savefig("skin_smoothing_result.png")  # 保存为图片文件
# plt.show()  # 注释掉这行
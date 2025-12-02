import numpy as np
import os
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False  


# 方波的傅里叶级数展开
def square_wave_fourier(t, n_max):
    f_t = np.zeros_like(t)
    # 循环累加奇数项：1, 3, 5, ..., n_max
    for k in range(1, n_max + 1, 2):
        # 根据图2和图3：bk = 4 / (k * pi), ak = 0
        bk = 4 / (k * np.pi)
        f_t += bk * np.sin(k * t)
    return f_t

# 定义时间轴
t = np.linspace(-np.pi, 3 * np.pi, 1000)

# 生成理想的矩形波
ideal_square_wave = np.sign(np.sin(t))

# 创建画布
fig1, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
fig1.suptitle('Step 3: Fourier Series vs Ideal Square Wave', fontsize=16)

n_values = [10, 20, 30]

for i, n in enumerate(n_values):
    # 1. 先画理想矩形波
    axs[i].plot(t, ideal_square_wave, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Ideal Square Wave')
    # 2. 再画傅里叶拟合波
    y = square_wave_fourier(t, n)
    axs[i].plot(t, y, color='tab:blue', linewidth=1.5, label=f'Fourier Approx (n={n})')
    # 设置图表样式
    axs[i].set_ylabel('Amplitude')
    axs[i].legend(loc='upper right')
    axs[i].grid(True, alpha=0.3)
    axs[i].set_ylim(-1.5, 1.5)

axs[-1].set_xlabel('Time t (radians)')
plt.tight_layout()


# 三角波的傅里叶级数展开

def triangle_wave_fourier(t, n_max):
    f_t = np.zeros_like(t)

    
    # 循环累加奇数项
    for n in range(1, n_max + 1, 2):
        # an = 0 
        # bn = 8 / (n^2 * pi^2) 
        bn = 8 / ((n * np.pi) ** 2)
        
        f_t += bn * np.cos(n * t)
        
    return f_t

# 创建三角波的画布
fig2 = plt.figure(figsize=(10, 4))
plt.title('Step 5: Fourier Series Approximation of Triangle Wave (n=10)', fontsize=16)

# 计算三角波 (取前10项近似)
y_triangle = triangle_wave_fourier(t, 20)

# 绘制
plt.plot(t, y_triangle, color='green', label='Triangle Wave Fit (n=20)')
plt.xlabel('Time t')
plt.ylabel('Amplitude')
plt.grid(True, alpha=0.3)
plt.legend()

# 显示所有图像
output_dir = "exp2"
os.makedirs(output_dir, exist_ok=True)

fig1.savefig(os.path.join(output_dir, "square_wave_vs_ideal_n10_20_30.png"), dpi=300, bbox_inches='tight')

fig2.savefig(os.path.join(output_dir, "triangle_wave_n20.png"), dpi=300, bbox_inches='tight')

plt.close(fig1)
plt.close(fig2)
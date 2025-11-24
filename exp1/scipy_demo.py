from scipy import constants as C
import numpy as np
import scipy.signal as signal
import scipy.linalg as linalg
import random
import matplotlib.pyplot as plt

print(f"圆周率 pi: {C.pi}")
print(f"黄金比例 phi: {C.golden}")
print(f"光速 c: {C.c} m/s")
print(f"普朗克常数 h: {C.h}")
# 单位换算：1英里 = ? 米
print(f"1 Mile in meters: {C.mile}")

# --- 一维中值滤波 ---
x = np.arange(0, 100, 10)
random.shuffle(x) # 打乱顺序模拟噪声
print("原始乱序信号:", x)
# 窗口大小为3：取当前值及其左右各一个值，排序后取中间值
# 这能有效抹平突变的孤立噪点
filt_1d = signal.medfilt(x, 3) 
print("1D中值滤波后:", filt_1d)

# --- 二维中值滤波 (常用于图像) ---
img_noise = np.random.randint(1, 1000, (4, 4))
# 指定核大小为 3x3
filt_2d = signal.medfilt(img_noise, (3, 3))

# 使用 medfilt2d (通常更快，但对数据类型有要求)
img_float = np.float32(img_noise)
filt_2d_fast = signal.medfilt2d(img_float)

# --- 基础一维卷积 ---
x = np.array([1, 2, 3]) # 信号
h = np.array([4, 5, 6]) # 卷积核 (滤波器脉冲响应)
# 离散卷积公式：y[n] = sum(x[k] * h[n-k])
res = signal.convolve(x, h) 
print("卷积结果:", res) # -> [4, 13, 28, 27, 18]

# --- FFT 加速卷积与自相关 ---
# 生成白噪声
sig = np.random.randn(1000)
# 自相关计算：信号与其反转后的自身进行卷积
# mode='full' 表示输出长度为 len(sig1) + len(sig2) - 1
autocorr = signal.fftconvolve(sig, sig[::-1], mode='full')

# 绘图
fig, (ax_orig, ax_mag) = plt.subplots(2, 1)
ax_orig.plot(sig)
ax_orig.set_title('White noise (Random Signal)')

# 计算滞后轴 (Lags)
lags = np.arange(-len(sig)+1, len(sig))
ax_mag.plot(lags, autocorr)
ax_mag.set_title('Autocorrelation (Peak at 0 lag)')
fig.tight_layout()
# plt.show()

# 定义传递函数 H(z) = 1 / (z^2 + 2z + 3)
# dt=0.5 表示采样时间
sys = signal.TransferFunction([1], [1, 2, 3], dt=0.5)

# 计算波德图数据
w, mag, phase = sys.bode()

plt.figure()
# 半对数坐标轴绘制幅频特性
plt.semilogx(w, mag)    
plt.title("Bode Magnitude Plot")
plt.xlabel("Frequency (rad/s)")
plt.ylabel("Magnitude (dB)")

plt.figure()
# 半对数坐标轴绘制相频特性
plt.semilogx(w, phase) 
plt.title("Bode Phase Plot")
plt.show()

rng = np.random.default_rng()
t = np.linspace(-1, 1, 201)

# 1. 生成合成信号：低频正弦 + 高频正弦 + 噪声
x = (np.sin(2*np.pi*0.75*t*(1-t) + 2.1) +
     0.1*np.sin(2*np.pi*1.25*t + 1) +
     0.18*np.cos(2*np.pi*3.85*t))
xn = x + rng.standard_normal(len(t)) * 0.08 # 添加噪声

# 2. 设计滤波器
# 创建一个 3阶 低通巴特沃斯滤波器
# 0.05 是归一化截止频率 (Nyquist 频率的倍数)
b, a = signal.butter(3, 0.05) 

# 3. 应用 lfilter (单向滤波)
# 计算稳态初始条件，减少滤波开始时的瞬态震荡
zi = signal.lfilter_zi(b, a)
# 进行第一次滤波，传入初始状态 zi * xn[0]
z, _ = signal.lfilter(b, a, xn, zi=zi*xn[0])

# 再次滤波 (级联)，这会进一步平滑，但相位延迟会加倍
z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])

# 4. 应用 filtfilt (双向滤波/零相位滤波)
# 它是正向滤一次，反向再滤一次。
# 优点：没有相位延迟 (Zero-phase distortion)，波形不会左右偏移。
# 缺点：不能用于实时系统 (因为需要未来的数据)。
y = signal.filtfilt(b, a, xn)

# 绘图对比
plt.figure(figsize=(10, 6))
plt.plot(t, xn, 'b', alpha=0.5, label='Noisy Signal')
plt.plot(t, z, 'r--', label='lfilter (Once) - Phase Delay')
plt.plot(t, z2, 'r', label='lfilter (Twice) - More Delay')
plt.plot(t, y, 'k', linewidth=2, label='filtfilt - Zero Phase')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# 定义矩阵 A
a = np.array([[1., 2.], [3., 4.]])

# --- 求逆矩阵 ---
a_inv = linalg.inv(a)
print("逆矩阵:\n", a_inv)
# 验证：A * A_inv 应该等于单位矩阵
print("A * A_inv:\n", np.dot(a, a_inv))

# --- 求解线性方程组 ---
# 方程组:
# 3x + 2y + 0z = 2
# 1x - 1y + 0z = 4
# 0x + 5y + 1z = -1
# 写成矩阵形式 Ax = B
A = np.array([[3, 2, 0], [1, -1, 0], [0, 5, 1]])
B = np.array([2, 4, -1])

# 使用 solve 求解 (比求逆再相乘更推荐)
x = linalg.solve(A, B)
print("方程组解 x:", x) # -> [ 2., -2.,  9.]

# 验证解
print("验证 Ax == B:", np.allclose(np.dot(A, x), B))
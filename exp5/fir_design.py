import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# 简单示例：在 4 kHz 采样下构造一个 50 Hz 干扰和 1 kHz 有用信号，
# 设计一个高通 FIR（保留 1 kHz，抑制 50 Hz）。

fs = 4000.0               # 采样率
dt = 1.0 / fs
t = np.arange(0, 0.1, dt)  # 绘图只取 0.1 s

x_noise = np.cos(2 * np.pi * 50 * t)
x_sig = np.cos(2 * np.pi * 1000 * t)
x = x_noise + x_sig

# FIR 参数（经验选择）
numtaps = 25
cutoff = 750.0  # Hz

# 设计高通滤波器
b = signal.firwin(numtaps, cutoff, window='hann', pass_zero=False, fs=fs)

# 频率响应
w, h = signal.freqz(b, worN=512, fs=fs)

# 过滤
y = signal.lfilter(b, 1, x)

# 绘图并保存
fig = plt.figure(figsize=(12, 8))

ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(t, x)
ax1.set_title('原始信号（50 Hz 干扰 + 1 kHz）')
ax1.grid(True)

ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(w, np.abs(h))
ax2.set_title('FIR 高通（Hann 窗）')
ax2.set_xlabel('频率 (Hz)')
ax2.set_ylabel('幅度')
ax2.axvline(cutoff, color='r', linestyle='--', label=f'截止 {cutoff:.0f} Hz')
ax2.legend()
ax2.grid(True)

ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(t, y, color='g')
ax3.set_title('滤波后信号')
ax3.set_xlabel('时间 (s)')
ax3.grid(True)

# 频谱（只看正频率）
N = len(t)
freqs = np.fft.rfftfreq(N, dt)
fft_x = np.abs(np.fft.rfft(x))
fft_y = np.abs(np.fft.rfft(y))

ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(freqs, fft_x, label='输入')
ax4.plot(freqs, fft_y, label='输出', alpha=0.7)
ax4.set_title('频谱')
ax4.set_xlabel('频率 (Hz)')
ax4.legend()
ax4.grid(True)

fig.tight_layout()
out = 'exp5_result.png'
fig.savefig(out, dpi=150)
print('保存：', out)
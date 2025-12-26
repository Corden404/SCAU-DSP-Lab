from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy import signal

OUTPUT_DIR = Path(__file__).resolve().parent / "output_figs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 尽量选择可用的中文字体，避免保存图片时中文变成方块
_cjk_candidates = [
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "WenQuanYi Zen Hei",
    "SimHei",
    "Microsoft YaHei",
]
_available_fonts = {f.name for f in font_manager.fontManager.ttflist}
for _name in _cjk_candidates:
    if _name in _available_fonts:
        plt.rcParams["font.sans-serif"] = [_name]
        break
plt.rcParams["axes.unicode_minus"] = False

# 作业1：陷波器（50Hz 有用信号 + 1000Hz 干扰）
fs = 8000
T = 0.5
t = np.linspace(0, T, int(fs * T), endpoint=False)

f_wanted = 50
f_noise = 1000
original_signal = np.cos(2 * np.pi * f_wanted * t) + np.cos(2 * np.pi * f_noise * t)

Q = 30
b, a = signal.iirnotch(w0=f_noise, Q=Q, fs=fs)
filtered_signal = signal.filtfilt(b, a, original_signal)


def plot_spectrum(sig, fs, label):
    n = len(sig)
    k = np.arange(n)
    T_len = n / fs
    frq = k / T_len  # 频率轴
    frq = frq[range(n // 2)]  # 取一半区间
    
    Y = np.fft.fft(sig) / n  # 归一化 FFT
    Y = Y[range(n // 2)]
    
    plt.plot(frq, abs(Y), label=label)

fig1 = plt.figure(figsize=(12, 8))

# 时域波形对比（放大前 0.1s）
plt.subplot(2, 1, 1)
plt.plot(t, original_signal, label='原始信号 (50Hz + 1000Hz)', alpha=0.5)
plt.plot(t, filtered_signal, label='滤波后信号 (应只有50Hz)', color='red')
plt.xlim(0, 0.1)
plt.title('时域波形对比 (Time Domain)')
plt.legend()
plt.grid(True)

# 频域频谱对比
plt.subplot(2, 1, 2)
plot_spectrum(original_signal, fs, '原始频谱')
plot_spectrum(filtered_signal, fs, '滤波后频谱')
plt.title('频谱对比 (Frequency Domain)')
plt.legend()
plt.grid(True)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')

plt.tight_layout()
fig1.savefig(OUTPUT_DIR / "iir_notch_time_freq.png", dpi=300, bbox_inches="tight")
plt.close(fig1)

# 作业2：音频陷波（模拟 4000Hz 窄带噪声）
fs_audio = 44100
duration = 2.0
t_audio = np.linspace(0, duration, int(fs_audio * duration), endpoint=False)

# 模拟“有用音频”：几个低频分量叠加
clean_audio = (0.5 * np.sin(2 * np.pi * 300 * t_audio) + 
               0.3 * np.sin(2 * np.pi * 500 * t_audio) + 
               0.2 * np.sin(2 * np.pi * 800 * t_audio))

# 叠加窄带干扰：4000Hz
noise_freq = 4000
noise = 0.4 * np.sin(2 * np.pi * noise_freq * t_audio)

noisy_audio = clean_audio + noise

# 设计陷波器滤除 4000Hz
Q_audio = 50
b_audio, a_audio = signal.iirnotch(noise_freq, Q_audio, fs_audio)

# 滤波
restored_audio = signal.filtfilt(b_audio, a_audio, noisy_audio)

fig2 = plt.figure(figsize=(12, 6))

# 只画频谱（时域波形不便直接看出差异）
plot_spectrum(noisy_audio, fs_audio, '带噪音频 (含4000Hz)')
plot_spectrum(restored_audio, fs_audio, '去噪后音频')

plt.title('作业2：音频信号陷波前后频谱对比')
plt.legend()
plt.grid(True)
plt.xlim(0, 5000)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
fig2.savefig(OUTPUT_DIR / "audio_notch_spectrum.png", dpi=300, bbox_inches="tight")
plt.close(fig2)

# 隔直（去直流）滤波器：一阶 IIR
R = 0.98

# H(z) = (1 - z^-1) / (1 - R z^-1)
b_dc = [1, -1]
a_dc = [1, -R]

# 带直流偏置的信号：正弦 + 2
bias_signal = np.sin(2 * np.pi * 5 * t) + 2.0 
removed_dc_signal = signal.lfilter(b_dc, a_dc, bias_signal)

fig3 = plt.figure(figsize=(10, 4))
plt.plot(t[:1000], bias_signal[:1000], label='带直流偏置 (Offset=2)')
plt.plot(t[:1000], removed_dc_signal[:1000], label='隔直后 (Centered at 0)')
plt.title('隔直滤波器演示')
plt.legend()
plt.grid()
fig3.savefig(OUTPUT_DIR / "dc_block_demo.png", dpi=300, bbox_inches="tight")
plt.close(fig3)
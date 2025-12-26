import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

# 尝试为 matplotlib 选择一个支持中文的系统字体，避免保存图片时出现缺字警告
def enable_cjk_font():
    candidates = ['noto', 'wqy', 'simhei', 'msyh', 'arialuni', 'sourcehan', 'hanazono']
    sys_fonts = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
    for path in sys_fonts:
        lower = path.lower()
        if any(c in lower for c in candidates):
            try:
                font_manager.fontManager.addfont(path)
                fp = font_manager.FontProperties(fname=path)
                rcParams['font.family'] = fp.get_name()
                rcParams['axes.unicode_minus'] = False
                print(f'使用系统字体: {fp.get_name()} ({path})')
                return
            except Exception:
                continue
    # 没找到合适字体，给出安装建议
    print('未检测到支持中文的系统字体。可通过安装 `fonts-noto-cjk` 或 `fonts-wqy-zenhei` 解决。')

enable_cjk_font()

# 生成模拟信号（采样率 1000 Hz，时长 1.5 秒）
Fs = 1000            # 采样频率（Hz）
T = 1 / Fs           # 采样间隔（秒）
L = 1500             # 总样本点数（1.5 秒）
t = np.arange(L) * T # 时间轴

# 合成信号：50 Hz 和 120 Hz 的正弦相加，并叠加高斯噪声
f1 = 50
f2 = 120
S = 0.7 * np.sin(2 * np.pi * f1 * t) + 1.0 * np.sin(2 * np.pi * f2 * t)
X = S + 2 * np.random.randn(len(t))  # 加入噪声，模拟真实测量

# 计算 FFT，并由双侧谱得到单侧幅度谱
Y = np.fft.fft(X)
P2 = np.abs(Y / L)            # 双侧幅度并归一化
P1 = P2[: L // 2 + 1]        # 取单侧（实信号频谱对称）
P1[1:-1] = 2 * P1[1:-1]      # 除直流与 Nyquist 外，其余成分乘 2
f = Fs * np.arange((L / 2) + 1) / L

# 绘图：上图为时域（只画前100点），下图为 0-200 Hz 的幅度谱
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(t[:100], X[:100])
plt.title('时域信号（前100个样本）')
plt.xlabel('时间 (s)')
plt.ylabel('幅值')
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(f, P1)
plt.title('单侧幅度谱')
plt.xlabel('频率 (Hz)')
plt.ylabel('幅值')
plt.xlim(0, 200)
plt.grid()

# 在频谱上标注已知的成分峰值
plt.annotate(f'峰值 {f1} Hz', xy=(f1, 0.7), xytext=(f1 + 10, 0.9),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate(f'峰值 {f2} Hz', xy=(f2, 1.0), xytext=(f2 + 10, 1.2),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.tight_layout()
try:
    plt.show()
except Exception:
    pass

# 将图片保存到当前脚本所在目录（exp3）
out_dir = os.path.dirname(__file__) or '.'
out_path = os.path.join(out_dir, 'fft_result.png')
plt.savefig(out_path, bbox_inches='tight')
print(f'实验完成，结果已保存为 {out_path}')
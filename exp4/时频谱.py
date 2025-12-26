import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib import rcParams
import librosa
import librosa.display
from pathlib import Path

# 解决中文字体缺失导致的标题/坐标轴中文无法显示问题
# （Codespaces 常见：默认 DejaVu Sans 不含 CJK 字形）
_CJK_FONT_CANDIDATES = [
	"Noto Sans CJK SC",
	"Noto Sans CJK",
	"WenQuanYi Zen Hei",
	"WenQuanYi Micro Hei",
	"Source Han Sans CN",
]

for _font_name in _CJK_FONT_CANDIDATES:
	try:
		_font_path = font_manager.findfont(
			font_manager.FontProperties(family=_font_name),
			fallback_to_default=False,
		)
		rcParams["font.family"] = "sans-serif"
		rcParams["font.sans-serif"] = [_font_name]
		rcParams["axes.unicode_minus"] = False
		print(f"使用中文字体：{_font_name} ({_font_path})")
		break
	except Exception:
		continue

# 1. 加载音频
folder = Path(__file__).resolve().parent
out_dir = folder / 'output_figs'
out_dir.mkdir(parents=True, exist_ok=True)

wav_files = sorted(folder.glob('*.wav'))
if wav_files:
	filename = str(wav_files[0])
	print(f"发现本地 WAV: {filename}")
	# sr=None 表示保留原采样率
	y, sr = librosa.load(filename, sr=None)
else:
	print("未找到本地 WAV，使用 librosa 示例音频 trumpet。")
	filename = librosa.ex('trumpet')
	y, sr = librosa.load(filename, sr=None)

print(f"采样率: {sr} Hz")
print(f"时长: {len(y)/sr:.2f} 秒")

plt.figure(figsize=(14, 5))
librosa.display.waveshow(y, sr=sr)
plt.title('时域图 (Time Domain Waveform)')
plt.xlabel('时间 (Time)')
plt.ylabel('振幅 (Amplitude)')
plt.grid(True)
wave_path = out_dir / '01_waveform.png'
plt.savefig(wave_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"已保存：{wave_path}")

# 计算 FFT
n = len(y)
yf = np.fft.fft(y)
xf = np.fft.fftfreq(n, 1/sr)

# 取前一半（正频率部分）
half_n = n // 2
freqs = xf[:half_n]
magnitudes = np.abs(yf[:half_n])

plt.figure(figsize=(14, 5))
plt.plot(freqs, magnitudes)
plt.title('频谱图 (FFT Spectrum)')
plt.xlabel('频率 (Frequency Hz)')
plt.ylabel('幅度 (Magnitude)')
# 为了看清细节，通常限制x轴范围，例如 0到5000Hz (人声/乐器主要范围)
plt.xlim(0, 5000) 
plt.grid(True)
fft_path = out_dir / '02_fft_spectrum.png'
plt.savefig(fft_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"已保存：{fft_path}")

# 1. 计算 STFT
# 课件第7页指定参数：n_fft=2048, hop=512, window='hann'
D = librosa.stft(y, n_fft=2048, hop_length=512, window='hann')

# 2. 将幅度转换为分贝 (dB)，因为人耳对声音是对数感知的，且便于可视化
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# 3. 绘制热力图 (Spectrogram)
plt.figure(figsize=(14, 6))
img = librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')
plt.title('时频图 (STFT Spectrogram)')
plt.ylim(0, 8000) # 限制显示频率范围以便观察
stft_path = out_dir / '03_stft_spectrogram.png'
plt.savefig(stft_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"已保存：{stft_path}")
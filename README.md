# SCAU-DSP-Lab

华农数字信号处理实验课作业与实验代码整理（基于 Codespaces/Dev Container 开发）。

## 环境要求

- Python 3.x
- 依赖：见 requirements.txt

可选（主要给 exp4 音频相关脚本）：

- ffmpeg：将 mp3/m4a 等转成 wav，提升读取稳定性
- file：用于检测音频真实容器类型（exp4 辅助脚本会用到）
- CJK 字体：若保存图片出现中文缺字，可安装（例如 fonts-noto-cjk）

Ubuntu/Debian 可用：

    sudo apt update
    sudo apt install -y ffmpeg file fonts-noto-cjk

## 安装依赖

    python -m pip install -r requirements.txt

可选：使用虚拟环境

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt

## 运行各实验

### exp1

目录：

- exp1/note_collection/：numpy/scipy/matplotlib 的练习与笔记
- exp1/skin-smoothing_demo/：磨皮示例

运行：

    python exp1/skin-smoothing_demo/skin_smoothing.py

输出：

- exp1/skin-smoothing_demo/comparison_result.png

### exp2

傅里叶级数拟合方波/三角波并保存图片。

    python exp2/script.py

输出（在 exp2/ 下）：

- exp2/square_wave_vs_ideal_n10_20_30.png
- exp2/triangle_wave_n20.png

### exp3

FFT 实验：生成含噪信号并绘制时域与单侧幅度谱。

    python exp3/fft_experiment.py

输出：

- exp3/fft_result.png

### exp4

音频可视化：时域、FFT 频谱、STFT 时频谱。图片输出到 exp4/output_figs/。

    python exp4/时频谱.py

读取逻辑：

- 若 exp4/ 目录存在 .wav 文件，优先读取第一个
- 否则使用 librosa 自带示例音频

输出：

- exp4/output_figs/01_waveform.png
- exp4/output_figs/02_fft_spectrum.png
- exp4/output_figs/03_stft_spectrogram.png

本目录还提供了本地音频修复/规范化脚本（需要系统安装 ffmpeg）：

    python exp4/repair_audio_local.py path/to/audio.mp3 --to-wav

### exp5

FIR 设计示例（高通），并保存综合图。

    python exp5/fir_design.py

输出：

- exp5_result.png

### exp6

IIR 相关：陷波、音频陷波、隔直（去直流）。输出图片保存到 exp6/output_figs/。

    python exp6/IIR.py

输出：

- exp6/output_figs/iir_notch_time_freq.png
- exp6/output_figs/audio_notch_spectrum.png
- exp6/output_figs/dc_block_demo.png

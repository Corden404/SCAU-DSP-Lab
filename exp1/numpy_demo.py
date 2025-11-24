import numpy as np

"""
> 1. 数组创建与初始化
"""
# --- 基础转换 ---
# 将 Python 列表转换为 NumPy 数组
arr_list = np.array([1, 2, 3, 4, 5]) 
# 将 Python 元组转换为 NumPy 数组
arr_tuple = np.array((1, 2, 3, 4, 5)) 

# --- 序列生成 ---
# 类似于 Python 的 range()，但生成的是数组。常用于生成离散的索引。
# 生成 [0, 1, 2, 3, 4, 5, 6, 7]
arr_range = np.arange(8) 
# 指定步长：从1开始，不超过10，步长为2 -> [1, 3, 5, 7, 9]
arr_step = np.arange(1, 10, 2) 

# [DSP 重点] linspace (Linear Space)
# 生成线性等分向量。常用于生成时间轴 t 或频率轴 f。
# 在 0 到 10 之间生成 11 个点（包含端点） -> [0., 1., ..., 10.]
t = np.linspace(0, 10, 11) 

# --- 特殊矩阵初始化 ---
# 全 0 数组：常用于初始化缓存或占位
z1 = np.zeros(3)          # 一维: [0., 0., 0.]
z2 = np.zeros((3, 1))     # 二维列向量: 3行1列

# 全 1 数组：常用于掩码或初始增益为1的情况
o1 = np.ones((2, 2))      # 2x2 全1矩阵

# 单位矩阵 (Identity Matrix)：对角线为1，其余为0
# 在矩阵运算中相当于“1”，任何矩阵乘以它都不变
eye = np.identity(3)      # 3x3 单位阵

# 空数组：只申请内存，不初始化值（值是内存中残留的垃圾数据）
# 速度最快，适合后续马上会被数据填充的场景
emp = np.empty((2, 2))

"""
> 2. DSP 窗函数
"""
# Hamming 窗：边缘平滑，旁瓣较低
# 生成一个长度为 20 的汉明窗数组
win_ham = np.hamming(20)

# Blackman 窗：比 Hamming 窗有更低的旁瓣衰减，但主瓣更宽
win_blk = np.blackman(10)

# Kaiser 窗：可以通过参数 beta 调节主瓣宽度和旁瓣衰减的权衡
# 长度12，beta参数为5
win_kai = np.kaiser(12, 5)

# [应用示例]：
# 假设 signal 是长度为 20 的信号，加窗操作就是简单的“点乘”
# windowed_signal = signal * win_ham

"""
> 3. 数组比较
"""
x = np.array([1, 2, 3, 4.001, 5])
y = np.array([1, 1.999, 3, 4.01, 5.1])

# 直接比较（通常会失败）
print(np.allclose(x, y)) # -> False

# 设置相对误差 (rtol)
# 只要两数误差在 20% 以内即认为相等
print(np.allclose(x, y, rtol=0.2)) # -> True

# 设置绝对误差 (atol)
# 只要两数差值绝对值小于 0.2 即认为相等
print(np.allclose(x, y, atol=0.2)) # -> True

"""
> 4. 基础运算与广播
"""
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# --- 标量运算 ---
# 数组每个元素都乘以 2（相当于信号增益放大）
print(a * 2)  # -> [2, 4, 6]
# 数组每个元素加 2（相当于信号引入直流偏置 DC Offset）
print(a + 2)  # -> [3, 4, 5]

# --- 数组间运算 ---
# 对应位置元素相乘（Hadamard Product），注意这不是矩阵乘法！
# 在 DSP 中，这常用于信号加窗、调制等
print(a * b)  # -> [4, 10, 18]

# 对应位置相加（信号叠加）
print(a + b)  # -> [5, 7, 9]

# --- 广播机制 (Broadcasting) ---
# 当维度不完全匹配但符合规则时，NumPy 会自动扩展
matrix = np.array([[1, 2, 3], [4, 5, 6]]) # 2x3 矩阵
vector = np.array([1, 1, 1])              # 1x3 向量
# vector 会被自动“复制”成 2行，分别加到 matrix 的每一行上
print(matrix + vector)

"""
> 5. 线性代数操作
"""
a = np.array([1, 2, 3])
b = np.array([[1, 2, 3], [4, 5, 6]])

# --- 转置 (.T) ---
# 二维数组转置：行变列，列变行
print(b.T) 
# [注意]：一维数组（秩为1）转置后形状不变，还是它自己
print(a.T) # -> [1, 2, 3] 

# --- 点积/内积 (Dot Product) ---
v1 = np.array([1, 2])
v2 = np.array([3, 4])
# 向量内积：1*3 + 2*4 = 11
print(np.dot(v1, v2)) 
# 或者使用对象方法
print(v1.dot(v2))

# 矩阵乘法：(2x3) 乘以 (3x2) -> (2x2)
mat_a = np.array([[1, 2, 3], [4, 5, 6]])
mat_b = mat_a.T
print(np.dot(mat_a, mat_b))

"""
> 6. 索引与切片
"""

data = np.array([[10, 20, 30], 
                 [40, 50, 60], 
                 [70, 80, 90]])

# 基础索引：第0行，第2列
print(data[0, 2]) # -> 30

# 花式索引 (Fancy Indexing)：同时获取不连续的行
# 获取第0行和第1行
print(data[[0, 1]]) 

# 组合索引：获取 (0,1) 和 (1,2) 两个位置的元素
# 也就是 data[0,1] 和 data[1,2] -> [20, 60]
print(data[[0, 1], [1, 2]]) 

# 常用切片：获取第 1 列的所有数据
print(data[:, 1]) # -> [20, 50, 80]

"""
> 7. 通用数学函数 
"""

# 创建 0 到 3.14 的 10 个点
x = np.linspace(0, 3.14, 10)

# --- 三角函数 ---
y_sin = np.sin(x)  # 计算正弦波
y_cos = np.cos(x)  # 计算余弦波

# --- 复数处理 (DSP 核心) ---
# 创建一个复数信号
z = np.array([1+1j, 1-1j, 3+4j])

# 计算模 (Magnitude/Amplitude) -> sqrt(real^2 + imag^2)
mag = np.absolute(z) # -> [1.414, 1.414, 5.0]

# 计算相位角 (Phase)
# 默认返回弧度
phase_rad = np.angle(z) 
# 返回角度
phase_deg = np.angle(z, deg=True) # -> [45., -45., 53.13]

# --- 其他数学运算 ---
# 对数 (常用于计算分贝 dB)
val = np.log10([10, 100, 1000]) # -> [1., 2., 3.]
# 乘方
sq = np.array([2, 3]) ** 2 # -> [4, 9]

"""
> 8. 形状操作
"""

a = np.arange(10) # [0, 1, ..., 9]

# --- Reshape (返回新视图) ---
# 将一维数组重塑为 2行5列
b = a.reshape(2, 5)
# 使用 -1 让 NumPy 自动计算某一维的大小
# 下面意思是：我要5行，列数你帮我算 (结果是2列)
c = a.reshape(5, -1)

# --- Resize (原地修改) ---
# 注意：resize 会直接改变变量 a 的形状，且如果新大小大于原大小，会补0
x = np.array([1, 2, 3, 4])
x.resize((2, 3)) # 变成 2x3，原数据不够，补0
# x 变成了 [[1, 2, 3], [4, 0, 0]]

"""
> 9. 统计与聚合
"""

mat = np.matrix([[1, 2, 3], 
                 [4, 5, 6]])

# 求和
print(mat.sum()) # 所有元素之和 -> 21

# 指定轴 (Axis)
# axis=0: 沿着“纵向/列”压缩 -> [1+4, 2+5, 3+6]
print(mat.sum(axis=0)) # -> [[5, 7, 9]]

# axis=1: 沿着“横向/行”压缩 -> [1+2+3, 4+5+6]
print(mat.sum(axis=1)) # -> [[6], [15]]

# 平均值 (Mean)
print(mat.mean(axis=1)) # 每行的平均值

# 加权平均
data = np.array([10, 20])
w = np.array([0.3, 0.7]) # 权重
# 10*0.3 + 20*0.7 = 3 + 14 = 17
print(np.average(data, weights=w))

"""
> 10. 傅里叶变换
"""

import matplotlib.pyplot as plt

# 1. 生成信号
# 0 到 2pi，50个点
t = np.linspace(0, 2*np.pi, 50) 
wave = np.cos(t) # 时域信号 (余弦波)

# 2. FFT 变换
# 结果是复数数组，包含幅度和相位信息
transformed = np.fft.fft(wave) 

# 3. 移频 (FFT Shift)
# FFT 的结果默认顺序是 [0, 正频率..., 负频率...]
# fftshift 将其重新排序为 [...负频率..., 0, ...正频率...]
# 这对于绘图观察频谱非常重要，让 0 频位于中心
shifted = np.fft.fftshift(transformed)

# 4. 逆变换 (IFFT)
# 将频域数据还原回时域
restored = np.fft.ifft(transformed)

# 绘图示例 (通常取模看幅度谱)
# plt.plot(np.absolute(shifted))
# plt.show()
### 1. 数组的创建 (Array Creation)
这是 NumPy 的基础，DSP 中所有的信号（Signal）通常都存储为数组。

*   **基本转换**:
    *   `np.array()`: 可以将 Python 的列表（list）、元组（tuple）或 range 对象转换为 NumPy 数组。
    *   支持多维数组创建，例如传入嵌套列表 `[[1, 2], [3, 4]]`。
*   **生成序列**:
    *   `np.arange(start, stop, step)`: 类似于 Python 原生的 `range()`，但在处理浮点数步长时更强大。
    *   `np.linspace(start, stop, num)`: **DSP 常用**。生成等差数列，指定区间内生成 `num` 个点。常用于生成时间轴（Time Vector）或频率轴。

### 2. 特殊矩阵与初始化 (Special Matrices)
在算法初始化或构建滤波器系数矩阵时常用。

*   **全0与全1**:
    *   `np.zeros(shape)`: 生成全 0 数组。
    *   `np.ones(shape)`: 生成全 1 数组。
    *   *注意*: `shape` 可以是整数（一维）或元组（如 `(3, 1)` 代表列向量）。
*   **单位矩阵**:
    *   `np.identity(n)`: 生成 $n \times n$ 的单位矩阵（对角线为1，其余为0）。
*   **空数组**:
    *   `np.empty(shape)`: 只申请内存空间但不初始化值（内容是内存中的随机垃圾值），速度最快，适合后续会立即填充数据的场景。

### 3. DSP 窗函数 (Window Functions)
这是 DSP 特有的部分，用于在进行 FFT 之前对信号加窗，以减少频谱泄漏（Spectral Leakage）。

*   **常见窗函数**:
    *   `np.hamming(N)`: 汉明窗。
    *   `np.blackman(N)`: 布莱克曼窗。
    *   `np.kaiser(N, beta)`: 凯撒窗（需要指定 $\beta$ 参数）。
*   *笔记*: 这些函数返回的是一个长度为 N 的一维数组，通常与信号数组进行点乘操作。

### 4. 数组比较 (Comparison)
由于浮点数精度问题，不能直接用 `==` 比较浮点数组。

*   **近似比较**:
    *   `np.allclose(x, y, rtol, atol)`: 检查两个数组在给定误差范围内是否相等。
        *   `rtol`: 相对误差。
        *   `atol`: 绝对误差。

### 5. 基础算术运算 (Arithmetic Operations)
NumPy 的核心特性：**广播机制 (Broadcasting)** 和 **元素级运算 (Element-wise)**。

*   **数组与标量**:
    *   `x * 2`, `x + 2` 等：数组中每个元素都与该标量进行运算。
*   **数组与数组**:
    *   `a * b`, `a / b`, `a + b`: **对应位置元素**进行运算（并非矩阵乘法）。
    *   **广播 (Broadcasting)**: 当两个数组维度不完全一致但满足特定规则时（如 `(4,)` 加 `(1,)`），NumPy 会自动扩展较小的数组以匹配较大数组的维度。

### 6. 线性代数操作 (Linear Algebra)
*   **转置**:
    *   `.T`: 矩阵转置。*注意*: 一维数组（秩为1）转置后形状不变，必须先 reshape 成二维（如 $1 \times N$）才能看到转置效果。
*   **点积/内积**:
    *   `np.dot(a, b)` 或 `a.dot(b)`:
        *   对于一维数组：计算向量内积（标量）。
        *   对于二维数组：计算矩阵乘法。
        *   也可以计算矩阵与向量的乘积。

### 7. 索引与切片 (Indexing & Slicing)
*   **基础索引**: `b[row, col]` 取代了 Python 的 `b[row][col]` 写法，效率更高。
*   **花式索引 (Fancy Indexing)**:
    *   `b[[0, 1]]`: 选取第0行和第1行。
    *   `x[[1, 3, 5]]`: 同时选取多个不连续位置的元素。

### 8. 通用数学函数 (Universal Functions - ufunc)
NumPy 提供了大量对数组进行元素级运算的函数，速度远快于 Python 循环。

*   **三角函数**: `np.sin()`, `np.cos()`, `np.arccos()` (反余弦)。
*   **取整**:
    *   `np.round()`: 四舍五入。
    *   `np.floor()`: 向下取整。
    *   `np.ceil()`: 向上取整。
*   **复数处理 (DSP 核心)**:
    *   `np.angle(z, deg=True/False)`: 计算复数的相位角（支持角度或弧度）。
    *   `np.absolute(z)`: 计算复数的模（Magnitude）或实数的绝对值。
*   **其他**: `np.log10()`, `np.sqrt()`, `np.isnan()`。

### 9. 函数向量化 (Vectorization)
如果有一个自定义的 Python 函数只能处理标量，可以使用 NumPy 将其“升级”为能处理数组的函数。

*   `np.vectorize(func)`: 将普通函数转换为能对数组中每个元素执行的函数，避免显式写 `for` 循环。

### 10. 形状操作 (Shape Manipulation)
在 DSP 中，经常需要将一维信号重塑为矩阵进行分帧处理。

*   **改变形状**:
    *   `a.shape = (r, c)`: 原地修改属性。
    *   `a.reshape(r, c)`: 返回一个新形状的数组（数据共享）。
    *   `-1` 参数: `reshape(5, -1)` 表示第二维度自动计算。
*   **改变大小**:
    *   `a.resize(shape)`: **注意**，这与 reshape 不同，它可以改变数组的总元素个数（如果变大，新元素补0），且是原地修改。

### 11. 矩阵对象 (Matrix Class)
*注意：NumPy 官方建议主要使用 Array，但在某些旧代码或纯线性代数计算中会用到 Matrix。*

*   `np.matrix()`: 创建后的对象默认 `*` 号代表矩阵乘法，而不是元素级乘法。
*   `np.linalg.inv()`: 求逆矩阵。
*   `np.corrcoef()`: 计算相关系数矩阵。

### 12. 统计运算 (Statistics)
可以指定 `axis`（轴）进行计算。

*   `sum()`, `mean()`, `max()`:
    *   `axis=0`: 沿着纵轴（列）操作，通常是“跨行”计算。
    *   `axis=1`: 沿着横轴（行）操作。
*   `np.average(weights=...)`: 计算加权平均值。

### 13. 傅里叶变换 (FFT)
这是 DSP 最核心的功能模块 `np.fft`。

*   **正变换**: `np.fft.fft(wave)` 将时域信号转换为频域信号。
*   **反变换**: `np.fft.ifft(transformed)` 将频域信号还原为时域。
*   **移频**: `np.fft.fftshift(transformed)` 将零频分量（DC）移动到频谱中心。这在绘图分析频谱时非常重要，否则零频通常位于数组的开头。

### 14. 常用常量 (Constants)
*   `np.Inf`: 无穷大。
*   `np.NAN`: Not a Number（非数值），常用于处理缺失数据。
*   `np.pi`: 圆周率（代码中虽未显式列出常量列表，但在 `np.linspace(0, 2*np.pi, ...)` 中有使用）。

这份笔记涵盖了 **SciPy** 库中对 DSP（数字信号处理）至关重要的模块，主要是 `scipy.signal`（信号处理）、`scipy.constants`（物理常数）和 `scipy.linalg`（线性代数）。

SciPy 是建立在 NumPy 之上的，提供了更多高级的科学计算功能。

---

### 1. 物理常数库 (Constants)
DSP 经常涉及物理世界的信号处理（如光、声、电磁波），`scipy.constants` 提供了精确的物理常数，避免手动输入导致的精度误差。

*   **常用常数**: `pi` (圆周率), `c` (光速), `h` (普朗克常数).
*   **单位换算**: 包含如 `mile` (英里转米), `inch` 等比例因子。

### 2. 中值滤波 (Median Filtering)
中值滤波是一种**非线性滤波**技术，特别擅长去除**椒盐噪声**（Salt and pepper noise，即图像或信号中的离群点），同时能比均值滤波更好地**保留边缘**。

*   **一维处理**: `signal.medfilt(x, kernel_size)`。窗口大小必须是奇数。
*   **二维处理**: `signal.medfilt2d(x)`。常用于图像去噪。注意数据类型通常需要转换为 float32 或 float64。

### 3. 卷积与相关 (Convolution & Correlation)
这是线性系统理论的核心。

*   **时域卷积**: `signal.convolve(x, h)`。计算两个信号的卷积，相当于输入信号经过滤波器系统的输出。
*   **频域卷积**: `signal.fftconvolve()`。利用 FFT 计算卷积，对于长信号序列，速度远快于直接卷积（复杂度从 $O(N^2)$ 降为 $O(N \log N)$）。
*   **自相关**: 信号与自身的卷积（通常其中一个需要翻转），用于检测信号中的周期性或在噪声中寻找信号。

### 4. LTI 系统与频域响应 (LTI Systems & Bode Plot)
用于分析线性时不变（LTI）系统的频率特性。

*   **传递函数**: `signal.TransferFunction(num, den, dt)`。用分子分母系数定义系统。
*   **波德图 (Bode Plot)**: `sys.bode()`。绘制系统的幅频响应和相频响应，是控制理论和滤波器设计的重要工具。

### 5. 滤波器设计与应用 (Filter Design & Application)
这是 DSP 中最实用的部分。

*   **设计滤波器**: `signal.butter(N, Wn)`。设计巴特沃斯滤波器，返回系数 `b` (分子) 和 `a` (分母)。
*   **单向滤波 (Causal)**: `signal.lfilter(b, a, x)`。标准的递归滤波，输出会有**相位延迟**。
    *   `zi`: 初始状态 (Initial Conditions)，用于消除滤波开始时的瞬态响应。
*   **双向滤波 (Zero-phase)**: `signal.filtfilt(b, a, x)`。先正向滤波再反向滤波，**消除相位延迟**，但只能用于离线处理（因为需要未来的数据）。

### 6. 线性代数 (Linear Algebra)
`scipy.linalg` 比 `numpy.linalg` 效率更高，功能更全。

*   **求逆**: `linalg.inv(a)`。
*   **解方程组**: `linalg.solve(a, b)`。用于求解 $Ax=b$。在数值计算中，**推荐使用 `solve` 而不是求逆后相乘**，因为 `solve` 速度更快且数值稳定性更好。

---

按照您的要求，我将 **Matplotlib** 的学习内容严格划分为**「学习笔记」**和**「代码示例」**两大部分。

---

Matplotlib 是 Python 中最常用的绘图库。在 DSP（数字信号处理）中，它主要用于绘制波形图（时域）、频谱图（频域）、星座图以及三维曲面图。

### 1. 基础绘图与中文支持
*   **核心函数**: `plt.plot(x, y)` 是最常用的绘图函数，用于绘制连续信号（折线图）。
*   **中文乱码问题**: Matplotlib 默认字体不支持中文，直接输出会显示方框。
    *   **解决方案**: 使用 `font_manager.FontProperties` 加载本地字体文件（推荐），或者修改全局 `rcParams`。
*   **数学公式 (LaTeX)**: 在字符串前后加上 `$` 符号（如 `r'$sin(x)$'`），Matplotlib 会调用内置引擎渲染专业的数学公式，这在标注信号公式时非常有用。

### 2. 散点图 (Scatter Plot)
*   **用途**: 在 DSP 中常用于绘制**抽样点**、**星座图 (Constellation Diagram)** 或特征分布。
*   **与 plot 的区别**: `scatter` 允许单独控制**每一个点**的大小 (`s`)、颜色 (`c`) 和形状 (`marker`)，而 `plot` 通常对所有点应用统一式样。
*   **参数**:
    *   `s`: 尺寸 (Size)，可以是标量也可以是数组（气泡图）。
    *   `c`: 颜色 (Color)。
    *   `linewidths`: 标记的边缘线宽。

### 3. 饼状图与坐标系布局
*   **非标准用法**: 示例代码展示了一种高级用法——在直角坐标系中的不同位置绘制多个饼图。
*   **关键参数**:
    *   `explode`: “炸裂”效果，将某一块分离出来以突出显示。
    *   `autopct`: 自动计算并显示百分比格式。
    *   `aspect='equal'`: **至关重要**。如果不设置，饼图会被拉伸成椭圆。

### 4. 多子图布局 (Subplots)
*   **用途**: 在信号处理中，经常需要同时对比**输入信号 vs 输出信号**，或者**时域波形 vs 频域频谱**。
*   **`subplot(行, 列, 序号)`**: 将画布切分为网格。
    *   例如 `subplot(2, 2, 1)` 表示 2x2 网格的第 1 个。
    *   `subplot(2, 1, 2)` 可以跨列占据整行。
*   **`plt.sca(ax)`**: "Set Current Axis"，用于在不同的子图之间切换绘图焦点。

### 5. 三维绘图 (3D Plotting)
*   **环境**: 需要导入 `mpl_toolkits.mplot3d` 并指定 `projection='3d'`。
*   **数据生成 (`np.mgrid`)**:
    *   切片语法 `start:stop:step`。
    *   **技巧**: 如果 step 是**虚数**（如 `20j`），它表示生成的**点数**（linspace），而不是步长。
*   **绘图类型**:
    *   `plot_surface`: 绘制 3D 曲面（如时频图、模糊函数）。
    *   `plot`: 绘制 3D 曲线（如螺旋线）。

---
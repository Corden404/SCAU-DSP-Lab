"""
> 1. 基础绘图、中文支持与 LaTeX 公式
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# --- 配置中文支持 ---
# 这是一个跨平台痛点，需要指定具体的字体文件路径
# Windows 示例路径，Mac/Linux 用户需修改
myfont = fm.FontProperties(fname=r'C:\Windows\Fonts\STKAITI.ttf') 

# --- 数据生成 ---
t = np.arange(0.0, 2.0*np.pi, 0.01)  # 采样间隔 0.01
s = np.sin(t)
z = np.cos(t)
z_sq = np.cos(t*t) # 变频信号

# --- 绘图 ---
plt.figure(figsize=(8, 6)) # 设置画布大小

# label 参数用于图例显示
plt.plot(t, s, label='正弦 (Sine)') 
plt.plot(t, z, label='余弦 (Cosine)')
# 使用 LaTeX 语法渲染数学公式：$公式$
plt.plot(t, z_sq, 'b--', label=r'$cos(t^2)$') 

# --- 装饰与标注 ---
# 使用 fontproperties 应用中文字体
plt.xlabel('x-变量 (Time)', fontproperties='STKAITI', fontsize=14)
plt.ylabel('y-幅值 (Amplitude)', fontproperties='STKAITI', fontsize=14)
plt.title('三角函数波形图', fontproperties='STKAITI', fontsize=20)

# 图例也需要指定字体属性
plt.legend(prop=myfont) 
plt.grid(True) # 显示网格
plt.show()
"""
> 2. 散点图 (自定义样式)
"""
import matplotlib.pyplot as plt
import numpy as np

# 数据准备
a = np.arange(0, 2.0*np.pi, 0.1)
b = np.cos(a)

plt.figure(figsize=(10, 4))

# --- 子图1: 基础修改 ---
plt.subplot(1, 2, 1)
# s=20: 点的大小
# marker='+': 点的形状
# linewidths=2: '+' 号线条的粗细
plt.scatter(a, b, s=50, linewidths=2, marker='+', c='blue')
plt.title("Basic Scatter")

# --- 子图2: 高级气泡图 ---
plt.subplot(1, 2, 2)
x = np.random.random(50)
y = np.random.random(50)
# s=x*500: 每个点的大小根据 x 值变化 (气泡图效果)
# c='r': 红色
# marker='*': 星型
plt.scatter(x, y, s=x*500, c='r', marker='*')
plt.title("Bubble Chart")

plt.show()
"""
> 3. 饼状图 (自由布局)
"""
import numpy as np
import matplotlib.pyplot as plt

# 标签与颜色配置
labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
colors = ['yellowgreen', 'gold', '#FF0000', 'lightcoral']
explode = (0, 0.1, 0, 0.1)  # 使第2和第4片裂开

fig = plt.figure()
ax = fig.gca() # 获取当前轴

# --- 在同一坐标系下的不同位置画饼 ---

# 饼图1: 中心在 (0,0)
ax.pie(np.random.random(4), explode=explode, labels=labels, colors=colors,
       autopct='%1.1f%%', shadow=True, startangle=90,
       radius=0.25, center=(0, 0), frame=True) # frame=True 显示背后的坐标轴框

# 饼图2: 中心在 (1,1)
ax.pie(np.random.random(4), explode=explode, labels=labels, colors=colors,
       autopct='%1.1f%%', shadow=True, startangle=45,
       radius=0.25, center=(1, 1), frame=True)

# 饼图3: 中心在 (0,1)
ax.pie(np.random.random(4), explode=explode, labels=labels, colors=colors,
       autopct='%1.1f%%', shadow=True, startangle=90,
       radius=0.25, center=(0, 1), frame=True)

# 饼图4: 中心在 (1,0)
ax.pie(np.random.random(4), explode=explode, labels=labels, colors=colors,
       autopct='%1.2f%%', shadow=False, startangle=135,
       radius=0.35, center=(1, 0), frame=True)

# --- 调整坐标轴 ---
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Sunny", "Cloudy"])
ax.set_yticklabels(["Dry", "Rainy"])
ax.set_xlim((-0.5, 1.5))
ax.set_ylim((-0.5, 1.5))

# [关键] 设置纵横比相等，保证饼图是圆的
ax.set_aspect('equal')

plt.show()
"""
> 4. 多子图布局 (Subplots)
"""
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 500)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(x*x)

plt.figure(1)

# --- 布局规划 ---
# ax1: 第一行第一列 (左上)
ax1 = plt.subplot(2, 2, 1) 
# ax2: 第一行第二列 (右上)
ax2 = plt.subplot(2, 2, 2) 
# ax3: 第二行 (占据整行)
# facecolor: 设置背景色为黄色
ax3 = plt.subplot(2, 1, 2, facecolor='yellow') 

# --- 绘制 ax1 ---
plt.sca(ax1)                # 切换焦点到 ax1
plt.plot(x, y1, color='red')
plt.ylim(-1.2, 1.2)
plt.title("Signal 1")

# --- 绘制 ax2 ---
plt.sca(ax2)                # 切换焦点到 ax2
plt.plot(x, y2, 'b--')
plt.ylim(-1.2, 1.2)
plt.title("Signal 2")

# --- 绘制 ax3 ---
plt.sca(ax3)                # 切换焦点到 ax3
plt.plot(x, y3, 'g--')
plt.ylim(-1.2, 1.2)
plt.title("Chirp Signal")

plt.tight_layout() # 自动调整间距防止重叠
plt.show()
"""
> 5. 三维绘图 (3D Surface & Curve)
"""
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d # 必须导入，尽管不直接调用，它注册了 '3d' 投影

# --- 示例 1: 3D 曲面图 ---
fig = plt.figure(figsize=(10, 4))

# 创建子图，指定 projection='3d'
ax = fig.add_subplot(1, 2, 1, projection='3d')

# 生成网格数据
# 20j 表示生成 20 个点 (complex number trick in mgrid)
x, y = np.mgrid[-2:2:20j, -2:2:20j] 
z = 50 * np.sin(x + y)

# plot_surface: 绘制曲面
# rstride/cstride: 行列采样跨度，越小网格越密
# cmap: 颜色映射 (Blue_reversed)
ax.plot_surface(x, y, z, rstride=2, cstride=1, cmap=plt.cm.Blues_r)
ax.set_title("3D Surface")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# --- 示例 2: 3D 参数曲线 ---
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
z_line = np.linspace(-4, 4, 100) * 0.3
r = z_line**3 + 1
x_line = r * np.sin(theta)
y_line = r * np.cos(theta)

# plot: 绘制 3D 线条
ax2.plot(x_line, y_line, z_line, label='Parametric Curve', color='purple')
ax2.legend()
ax2.set_title("3D Curve")

plt.show()
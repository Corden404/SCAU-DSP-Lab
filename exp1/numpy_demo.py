import numpy as np

# 1.1 numpy简单应用
# 这部分展示了如何把 Python 原生的数据结构变成 NumPy 数组。
# 1. 列表转数组
arr_list = np.array([1, 2, 3, 4, 5])

# 2. 元组转数组
arr_tuple = np.array((1, 2, 3, 4, 5))

# 3. range对象转数组
arr_range = np.array(range(5))

# 创建二维数组（2行3列）
matrix = np.array([[1, 2, 3], [4, 5, 6]])

# 1. 默认从0开始
seq_1 = np.arange(8)           # 输出: [0 1 2 3 4 5 6 7]

# 2. 指定 起点、终点、步长
seq_2 = np.arange(1, 10, 2)    # 输出: [1 3 5 7 9]
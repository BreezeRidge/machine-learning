"""
NumPy 深度教程 — 不只是 API，理解背后的原理

本文件覆盖：
1. 数组的内存模型与性能差异
2. 向量化运算 vs for 循环（带计时对比）
3. 广播机制的规则与 ML 应用
4. 机器学习中最常用的矩阵操作
5. 完整练习题（附答案）

参考：《Python Data Science Handbook》Jake VanderPlas 第2章
"""

import numpy as np
import time

# ============================================================
# 1. 内存模型：为什么 NumPy 比 Python 列表快？
# ============================================================

print("=" * 60)
print("1. 性能对比：NumPy vs Python 列表")
print("=" * 60)

size = 1_000_000

# Python 列表
py_list = list(range(size))
start = time.time()
py_result = [x * 2 + 1 for x in py_list]
py_time = time.time() - start

# NumPy 数组
np_array = np.arange(size)
start = time.time()
np_result = np_array * 2 + 1
np_time = time.time() - start

print(f"Python 列表: {py_time:.4f} 秒")
print(f"NumPy 数组:  {np_time:.4f} 秒")
print(f"NumPy 快了:  {py_time / np_time:.0f} 倍")
print()
print("原因：NumPy 使用连续内存 + C 底层实现 + SIMD 并行指令")
print("      Python 列表每个元素是独立对象，有类型检查和指针跳转开销")

# ============================================================
# 2. 数组创建与属性 — 理解 shape 和 dtype
# ============================================================

print("\n" + "=" * 60)
print("2. 数组的核心属性")
print("=" * 60)

# 在 ML 中，shape 是最重要的属性，维度不匹配是最常见的 bug
X = np.random.randn(100, 5)  # 100 个样本，5 个特征
print(f"数据矩阵 X:")
print(f"  shape: {X.shape}    — (样本数, 特征数)")
print(f"  dtype: {X.dtype}    — 64位浮点数，ML 默认精度")
print(f"  ndim:  {X.ndim}       — 2维（矩阵）")
print(f"  size:  {X.size}      — 总元素数 = 100 × 5")
print(f"  内存:  {X.nbytes} 字节 — 每个 float64 占 8 字节")

# ============================================================
# 3. 广播机制 — ML 中最常用的隐式运算
# ============================================================

print("\n" + "=" * 60)
print("3. 广播机制（Broadcasting）")
print("=" * 60)

# 广播规则：从右往左对齐 shape，维度为 1 或缺失的自动扩展
# 这在 ML 中无处不在

# 示例1：数据标准化（每个特征减去均值，除以标准差）
X = np.array([
    [170, 65, 25],   # 身高(cm), 体重(kg), 年龄
    [180, 80, 30],
    [160, 50, 22],
    [175, 70, 28],
])
print("原始数据 X (4个样本, 3个特征):")
print(X)

mean = X.mean(axis=0)   # shape: (3,) — 每个特征的均值
std = X.std(axis=0)      # shape: (3,) — 每个特征的标准差
X_normalized = (X - mean) / std  # (4,3) - (3,) → 广播！

print(f"\n均值: {mean}  (shape: {mean.shape})")
print(f"标准差: {std}")
print(f"\n标准化后（均值≈0，标准差≈1）:")
print(X_normalized.round(2))
print("\n为什么要标准化？不同特征量纲差异大（身高170 vs 年龄25），")
print("会导致梯度下降时步长不一致，模型收敛慢甚至不收敛。")

# 示例2：给矩阵每行加不同的偏置（神经网络中的 bias）
W = np.random.randn(3, 4)   # 权重矩阵
b = np.array([0.1, 0.2, 0.3, 0.4])  # 偏置向量 shape: (4,)
output = W + b  # (3,4) + (4,) → 广播，每行都加上 b
print(f"\n神经网络偏置广播: W{W.shape} + b{b.shape} = output{output.shape}")

# ============================================================
# 4. ML 核心矩阵操作
# ============================================================

print("\n" + "=" * 60)
print("4. 机器学习核心矩阵操作")
print("=" * 60)

# --- 4.1 矩阵乘法：线性变换 ---
X = np.random.randn(5, 3)   # 5个样本, 3个特征
W = np.random.randn(3, 2)   # 权重: 3个输入 → 2个输出
Y = X @ W                    # 等价于 np.dot(X, W)
print(f"线性变换: X{X.shape} @ W{W.shape} = Y{Y.shape}")
print("含义：把3维特征空间映射到2维空间（降维/特征提取）")

# --- 4.2 转置：交换行列 ---
print(f"\nX 的转置: {X.shape} → {X.T.shape}")
print("用途：计算协方差矩阵 X^T @ X，这是 PCA 和线性回归的核心")

# --- 4.3 逆矩阵：求解线性方程组 ---
A = np.array([[2, 1], [1, 3]])
A_inv = np.linalg.inv(A)
print(f"\n矩阵 A:\n{A}")
print(f"A 的逆矩阵:\n{A_inv.round(4)}")
print(f"验证 A @ A_inv = I:\n{(A @ A_inv).round(4)}")
print("用途：线性回归的解析解 w = (X^T X)^(-1) X^T y")

# --- 4.4 特征值分解：PCA 的数学基础 ---
cov_matrix = X.T @ X / (X.shape[0] - 1)  # 协方差矩阵
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
print(f"\n协方差矩阵的特征值: {eigenvalues.round(4)}")
print("含义：特征值越大，对应方向的数据方差越大（信息量越多）")
print("PCA 就是选取最大特征值对应的特征向量作为新坐标轴")

# ============================================================
# 5. 练习题（附答案）
# ============================================================

print("\n" + "=" * 60)
print("5. 练习题")
print("=" * 60)

# --- 练习1：手动实现数据标准化 ---
print("\n练习1：标准化函数")
def standardize(X):
    """标准化：(X - 均值) / 标准差，使每个特征均值为0，标准差为1"""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / std

test_data = np.array([[1, 200], [2, 400], [3, 600], [4, 800]])
result = standardize(test_data.astype(float))
print(f"标准化后均值: {result.mean(axis=0).round(10)}")  # 应接近 [0, 0]
print(f"标准化后标准差: {result.std(axis=0).round(10)}")  # 应接近 [1, 1]

# --- 练习2：计算欧氏距离矩阵（KNN 算法的核心） ---
print("\n练习2：欧氏距离矩阵")
def euclidean_distance_matrix(X):
    """计算 X 中所有样本两两之间的欧氏距离

    数学推导：
    ||xi - xj||² = ||xi||² + ||xj||² - 2 * xi · xj
    这个技巧避免了双重 for 循环，是 KNN 高效实现的关键
    """
    sq_norms = np.sum(X ** 2, axis=1, keepdims=True)  # (m, 1)
    # ||xi - xj||² = ||xi||² + ||xj||² - 2 * xi^T * xj
    dist_sq = sq_norms + sq_norms.T - 2 * X @ X.T
    return np.sqrt(np.maximum(dist_sq, 0))  # maximum 防止浮点误差导致负数

points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
distances = euclidean_distance_matrix(points.astype(float))
print("4个点的距离矩阵:")
print(distances.round(4))
print("对角线为0（自己到自己），[0,3]为√2≈1.4142（对角线距离）")

# --- 练习3：Softmax 函数（神经网络分类的输出层） ---
print("\n练习3：Softmax 函数")
def softmax(z):
    """Softmax: 将任意实数向量转换为概率分布

    公式：softmax(zi) = exp(zi) / Σ exp(zj)
    技巧：减去最大值防止 exp 溢出（数值稳定性）
    """
    z_shifted = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)
print(f"输入 logits: {logits}")
print(f"Softmax 输出: {probs.round(4)}")
print(f"概率之和: {probs.sum():.4f}")  # 应为 1.0
print("含义：模型认为第1类概率最高(0.659)，这就是分类的预测结果")

print("\n" + "=" * 60)
print("完成！下一步：02_pandas_practical.py")
print("=" * 60)

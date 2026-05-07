# 机器学习系统学习

从零开始的机器学习学习之路，理论推导与代码实践并重。

## 推荐学习资源

### 书籍（按优先级排序）

| 书名 | 作者 | 适合阶段 | 说明 |
|------|------|----------|------|
| 《统计学习方法》 | 李航 | 全程 | 国内 ML 经典，数学推导严谨，覆盖主流算法 |
| 《机器学习》（西瓜书） | 周志华 | 全程 | 体系完整，理论与直觉兼顾，适合入门到进阶 |
| 《Pattern Recognition and Machine Learning》 | Bishop | 进阶 | 贝叶斯视角的 ML 圣经，数学要求较高 |
| 《Deep Learning》（花书） | Goodfellow et al. | 阶段 06+ | 深度学习理论标准参考 |
| 《Hands-On Machine Learning》 | Aurélien Géron | 实践 | 偏工程实践，scikit-learn + TensorFlow |

### 在线课程

- **吴恩达 Machine Learning (Coursera)** — 最经典的入门课，直觉解释极好
- **吴恩达 Deep Learning Specialization** — 深度学习五门课，理论+编程作业
- **[《动手学深度学习》d2l-zh](https://github.com/d2l-ai/d2l-zh)** — 李沐著，70+国家500+大学采用，可运行 Notebook，在线阅读：[zh.d2l.ai](https://zh.d2l.ai)
- **[林轩田机器学习笔记](https://github.com/apachecn/ntu-hsuantienlin-ml)** — 台大课程，数学推导最严谨的中文课程

### GitHub 资源库

| 仓库 | 说明 |
|------|------|
| [awesome-machine-learning](https://github.com/josephmisiti/awesome-machine-learning/blob/master/books.md) | 60k+ stars，ML 资源总索引 |
| [Mikoto10032/DeepLearning](https://github.com/Mikoto10032/DeepLearning) | 经典书籍 PDF 合集 |
| [awesome-AI-books](https://github.com/zslucky/awesome-AI-books) | AI 书籍合集，含 PDF 和配套代码 |
| [西瓜书笔记](https://github.com/Vay-keen/Machine-learning-learning-notes) | 周志华《机器学习》详细读书笔记 |
| [Hands-On ML 中译](https://github.com/apachecn/hands-on-ml-2e-zh) | scikit-learn 实战指南中文版 |
| [d2l-zh 配套课程](https://github.com/d2l-ai/courses-zh-v2) | 动手学深度学习配套视频 |
- **林轩田《机器学习基石》+《机器学习技法》** — 数学推导最严谨的中文课程
- **3Blue1Brown 线性代数/微积分系列 (YouTube)** — 数学直觉可视化，强烈推荐

## 学习路线

每个模块包含：**理论笔记（数学推导）** + **完整代码实践（从数据到评估）** + **参考文献**

### 前置知识依赖关系

```
00 线性代数直觉（零基础起点，可选）
 └──▶ 01 Python 数据科学基础
       └──▶ 02 数学基础（线性代数、概率统计、微积分）
             └──▶ 03 监督学习（回归、分类）
                   ├──▶ 04 无监督学习（聚类、降维）
                   └──▶ 05 模型评估与优化（交叉验证、正则化）
                         └──▶ 06 神经网络入门（感知机、反向传播、PyTorch）
```

> **零基础数学的读者**：强烈建议从阶段 00 开始。如果你从未系统学过线性代数（或学过但没有几何直觉），直接进入阶段 01 的 `theory.md` 会看到一堆符号后放弃。阶段 00 用 2-3 周建立直觉，后面一切都会顺利。

| 阶段 | 目录 | 主题 | 核心数学 | 状态 |
|------|------|------|----------|------|
| 00 | `00-linear-algebra-intuition/` | 向量、矩阵、变换、特征值的几何直觉 | 零基础起点，无符号推导 | 🔲 |
| 01 | `01-python-basics/` | NumPy, Pandas, Matplotlib | 向量/矩阵运算基础 | 🔲 |
| 02 | `02-math-foundations/` | 线性代数, 概率论, 微积分 | 矩阵分解, 贝叶斯定理, 梯度 | 🔲 |
| 03 | `03-supervised-learning/` | 线性回归→逻辑回归→SVM→决策树 | 最小二乘法, 极大似然, 核函数, 信息增益 | 🔲 |
| 04 | `04-unsupervised-learning/` | K-Means, GMM, PCA, t-SNE | EM 算法, 特征值分解, KL 散度 | 🔲 |
| 05 | `05-model-evaluation/` | 偏差-方差权衡, 交叉验证, 正则化 | L1/L2 范数, AIC/BIC | 🔲 |
| 06 | `06-neural-networks/` | 感知机→MLP→反向传播→PyTorch | 链式法则, 计算图, 梯度下降变体 | 🔲 |

## 环境准备

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
# 阶段 06 额外需要:
# pip install torch torchvision
```

## 每个模块结构

```
XX-topic/
├── README.md              # 模块概览与学习目标
├── theory.md              # 理论笔记：概念解释 + 数学推导
├── code/
│   ├── 01_basics.py       # 基础概念代码实现
│   ├── 02_from_scratch.py # 从零实现算法（不用库）
│   └── 03_sklearn.py      # 用 scikit-learn 实战完整案例
└── references.md          # 参考书目章节、论文、扩展阅读
```

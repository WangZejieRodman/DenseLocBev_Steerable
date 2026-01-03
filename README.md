基于对您提供的代码 (`DenseLocBev_clean` vs `DenseLocBev_Steerable`) 以及评估日志 (`rotation_results_*.txt`) 的深入分析，我为您撰写了以下 README.md。

这份文档重点突出了 Steerable 版本与 Clean 版本在**核心卷积机制**、**旋转鲁棒性**以及**理论与工程实现的差异**上的对比。

---

# DenseLocBev Steerable

## 简介

**DenseLocBev Steerable** 是 [DenseLocBev](https://www.google.com/search?q=../DenseLocBev_clean) 的演进版本，专为解决地下矿道环境中**视角旋转导致的位置识别失效**问题而设计。

原版 DenseLocBev 使用标准 CNN，对旋转极其敏感。本项目引入了 **群等变卷积神经网络 (Group Equivariant CNNs / Steerable CNNs)**，利用 `e2cnn` 库构建了具备  旋转等变性的 Backbone，旨在提取几何上真正的旋转不变特征。

## 核心差异

| 特性 | DenseLocBev (Clean) | DenseLocBev (Steerable) |
| --- | --- | --- |
| **卷积算子** | 标准 `nn.Conv2d` | `e2cnn.R2Conv` (Steerable Convolution) |
| **特征表示** | 标量特征图 (Scalars) | 几何场 (Geometric Fields: 标量/向量/张量) |
| **激活函数** | `ReLU` | `NormNonLinearity` (基于模长的非线性) |
| **旋转对称性** | 无 (依赖数据增强) | **数学上严格的等变性/不变性** |
| **参数量** | 较小 | 较大 (因使用更宽的通道以补偿约束) |
| **依赖库** | `torch` | `torch`, **`e2cnn`** |

## 理论与架构

### 1. 几何代数视角

与传统 CNN 将图像视为像素矩阵不同，本模型将 BEV 输入视为定义在连续平面上的**几何场**。

* **群定义**: 模型定义在  旋转群上（代码中使用 `Rot2dOnR2`，最大频率 ）。
* **特征类型**: 网络层不仅学习特征的“强度”（标量），还学习特征的“方向”（向量）和“形变”（张量）。

### 2. 网络结构 (`models/densebev.py`)

* **Backbone**: `DenseBEVBackbone` 采用了 Steerable Bottleneck 结构。
* 输入：32个标量通道 (Occupancy Grid)。
* 中间层：混合了 标量 (0阶)、向量 (1阶) 和 张量 (2阶) 表征。
* 通道扩充：为了弥补等变约束带来的表达能力限制，通道数相比原版增加了 50%~100% (96 -> 192 -> 384)。


* **不变性投影 (`InvariantMagnitude`)**:
* 在网络末端，显式地计算几何特征的**模长 (Norm)**，将方向性信息剥离，只保留旋转不变的能量值。



## 性能表现与分析

基于 [Chilean Underground Mine Dataset](https://www.google.com/search?q=https://minkloc3d.cs.cs.put.poznan.pl/) 的评估结果对比：

### 1. 标准卷积 vs. Steerable 卷积

| 旋转角度 | 标准 CNN Recall@1 | Steerable CNN Recall@1 | 现象分析 |
| --- | --- | --- | --- |
| **0°** | **93.58%** | 80.15% | 标准卷积在特定视角过拟合能力更强 |
| **90°** | 4.22% (崩塌) | **80.34%** (完美保持) | **Steerable 实现了完美的网格对称性** |
| **180°** | 31.12% | **80.97%** (完美保持) | 标准卷积即便有数据增强也无法泛化 |

### 2. 离散化效应与混叠 (The Aliasing Problem)

虽然模型实现了 90° 的完美等变，但在 45° 等非网格对齐角度上表现下降 (Recall@1 降至 ~16%)。

* **原因**: 输入数据是 0/1 二值的 Occupancy Grid。在 45° 旋转时，直线的墙壁在正方形网格上产生锯齿（混叠），导致高频几何特征（向量/张量）的相位计算出错。
* **解决方案 (WIP)**: 建议在后续改进中从连续群 () 切换到正规表达 (, Regular Representation) 或对输入进行高斯平滑。

## 环境依赖

除了基础依赖外，必须安装 `e2cnn`:

```bash
pip install e2cnn
pip install torch numpy tqdm

```

## 快速开始

### 训练

```bash
# 训练 Steerable 版本
python training/train_chilean_bev.py

```

*注意：由于计算几何特征涉及大量张量积运算，训练速度会比标准 CNN 慢。*

### 验证旋转不变性

我们提供了一个脚本用于验证模型在数学上的等变性：

```bash
python test_invariance.py

```

如果输出显示 **0° 与 90° 特征余弦相似度 > 0.99**，则证明代码实现正确。

### 评估

```bash
python eval/evaluate_chilean.py

```

## 目录结构

```
DenseLocBev_Steerable/
├── models/
│   ├── densebev.py      # [核心] 基于 e2cnn 重写的 Backbone
│   ├── layers/          # 适配几何张量的池化层
│   └── ...
├── training/            
│   ├── train_chilean_bev.py  # 训练入口
│   └── ...
├── eval/
│   ├── evaluate_chilean.py   # 评估脚本
│   └── rotation_results_*.txt # 实验日志
├── test_invariance.py   # 旋转等变性单元测试
└── ...

```

## 致谢

本项目深度使用了 [e2cnn](https://github.com/QUVA-Lab/e2cnn) 库来实现欧几里得群的等变卷积。

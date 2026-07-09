# HOT: Harmonic Oscillator Transformer

> 基于频率-相位解耦的时序建模新范式

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

## 概述

HOT（Harmonic Oscillator Transformer）是一种摒弃静态位置嵌入的替代架构，采用**频率-相位解耦**设计：
- **频率 ω**：由 Query-Key 能量差决定，表征 Token 的"内在节拍"
- **相位 θ**：可训练的隐状态，跨层传递，每层通过 `θ ← θ + ω·Δt` 更新
- **残差耦合门控**：`Score = Q·K/√d + α·cos(θ_i - θ_j)`，相位只做调制不做开关
- **因果序参量**：`r_i = |1/i Σ_{j≤i} exp(iθ_j)|`，驱动自适应残差缩放

## 核心特性

- **频率-相位解耦**：无循环依赖，可并行训练
- **残差耦合门控**：内容注意力永不丢失
- **因果序参量**：缓解深层范数消失
- **兼容现有范式**：可与 FlashAttention 协同工作

## 文档

- [研究论文](article.md) — 详述理论框架、工程实施策略及收敛性保障机制
- [开发计划](dev-plan.md) — 42M 参数规模的研究原型实现计划

## 快速开始

### 安装

```bash
# 从源码安装
git clone https://github.com/CloseAI-ai/hot.git
cd hot
pip install -e .

# 安装依赖
pip install -r requirements.txt
```

### 训练模型

```bash
python scripts/train.py --config configs/hot_42m.yaml
```

### 评估模型

```bash
python scripts/evaluate.py --config configs/hot_42m.yaml --checkpoint checkpoints/best_model.pt
```

### 运行消融实验

```bash
python scripts/ablation.py --experiments no_gating full_coupling no_annealing
```

### 可视化

```bash
python scripts/visualize.py --config configs/hot_42m.yaml --checkpoint checkpoints/best_model.pt
```

## 项目结构

```
hot/
├── README.md                   # 项目说明
├── requirements.txt            # 依赖列表
├── setup.py                    # 包安装
├── configs/                    # 训练配置
│   ├── base.yaml               # 基础配置
│   ├── hot_42m.yaml            # HOT 42M 配置
│   ├── rope_42m.yaml           # RoPE 基线配置
│   └── ablation/               # 消融实验配置
├── hot/                        # 核心代码
│   ├── model/                  # 模型实现
│   ├── training/               # 训练相关
│   ├── data/                   # 数据处理
│   ├── evaluation/             # 评估模块
│   └── utils/                  # 工具函数
├── scripts/                    # 训练和评估脚本
├── tests/                      # 单元测试
└── notebooks/                  # Jupyter notebooks
```

## 配置

### 模型配置（HOT 42M）

```yaml
model:
  hidden_size: 400
  num_heads: 8
  head_dim: 50
  num_layers: 12
  vocab_size: 50257
  ffn_size: 1550
  dropout: 0.1
```

### HOT 特定配置

```yaml
hot:
  alpha_init: 0.1              # 残差耦合强度初始值
  gate_position: "pre_softmax" # 门控位置（pre_softmax/none）
  annealing:
    schedule: "cosine"         # 退火调度类型
    warmup_steps: 2000         # 退火步数 K
```

## 硬件要求

| 配置 | 最低要求 | 推荐配置 |
|------|----------|----------|
| GPU | 1× RTX 3090 24GB | 1× A100 40GB |
| 内存 | 32GB | 64GB |
| 存储 | 200GB SSD | 500GB NVMe |
| 训练时间 | ~3 天（单卡） | ~1 天（单卡） |

## 开发

### 运行测试

```bash
pytest tests/
```

### 代码格式化

```bash
black hot/ tests/ scripts/
isort hot/ tests/ scripts/
```

### 类型检查

```bash
mypy hot/
```

## 许可证

本项目基于 [GNU Affero General Public License v3.0](LICENSE) 发布。

## 组织

本项目由 [CloseAI.ai](https://github.com/CloseAI-ai) 维护。

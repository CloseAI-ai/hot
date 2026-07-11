# HOT 项目状态

> 最后更新: 2026-07-11 (验证集分割完成)

## 项目概述

HOT (Harmonic Oscillator Transformer) — 一种基于频率-相位解耦的时序建模新范式，替换静态位置编码为内容自适应的相位动力学。

## 当前进度

### 训练状态

- **模型规模**: HOT 8M (120 hidden, 6 heads, 10 layers)
- **数据集**: TinyStories-Zh-2M
- **当前步数**: 0 / 150,000 (待开始)
- **训练设备**: RTX 4060 Laptop 8GB
- **学习率**: 4e-4 (保守策略，避免梯度饱和)
- **预热步数**: 1500 (适中，平衡稳定性和效率)
- **权重衰减**: 0.01 (AdamW黄金默认值，原0.1过高)
- **最小LR比例**: 0.05 (最终LR=2e-5，精细微调)
- **归一化**: Pre-LN (前置归一化，稳定深层训练)

### 可用模型配置

| 模型 | 参数量 | 配置文件 | 训练命令 |
|------|--------|----------|----------|
| HOT 42M | 42.72M | `configs/hot_42m.yaml` | `make train` |
| HOT 8M | 7.77M | `configs/hot_8m.yaml` | `make train-hot-8m` |

### 数据集分割

- **总样本数**: 1,994,314 条
- **训练集**: 1,794,883 条 (90%)
- **验证集**: 199,431 条 (10%)
- **随机种子**: 42 (确保可复现)
- **分割时间**: 2026-07-11
- **分割脚本**: `scripts/create_validation_split.py`
- **数据路径**: `data/splits/`

### 已完成工作

1. **核心架构** (✅ 完成)
   - IntrinsicFrequency: 频率从 Q/K 能量差派生
   - PhaseDynamics: Euler 更新 `θ ← θ + ω·Δt`
   - PhaseGating: 残差耦合门控 `α·cos(θ_i - θ_j)`
   - CausalOrderParameter: 因果序参量驱动残差缩放

2. **训练优化** (✅ 完成)
   - FlexAttention 支持（编译后 fused kernel）
   - 梯度检查点（用计算换内存）
   - 分块交叉熵（避免实体化完整 logits）
   - 因果 mask 预计算
   - torch.compile 优化（default mode, dynamic=True）

3. **代码清理** (✅ 完成)
   - 清理 72 个中间检查点（从 6.9G 减少到 664M）
   - 清理旧日志文件
   - .gitignore 正确配置

### 待完成工作

1. **训练完成** (🔄 进行中)
   - 继续训练到 100K 步
   - 监控验证集损失

2. **评估** (⏳ 待开始)
   - 困惑度评估
   - 外推能力测试
   - 相位频谱分析

3. **消融实验** (⏳ 待开始)
   - no_gating: 无门控基线
   - full_coupling: 完整 Kuramoto 耦合
   - no_annealing: 无退火

4. **可视化** (⏳ 待开始)
   - 相位动力学可视化
   - 频率分布分析
   - 训练曲线

## 项目结构

```
hot/
├── README.md                   # 项目说明
├── STATUS.md                   # 本文件
├── LICENSE                     # AGPL-3.0
├── Makefile                    # 常用命令
├── requirements.txt            # 依赖列表
├── setup.py                    # 包安装
├── article.md                  # 研究论文
├── dev-plan.md                 # 开发计划
├── monitor.py                  # 训练监控脚本
├── configs/                    # 训练配置
│   ├── base.yaml               # 基础配置
│   ├── hot_42m.yaml            # HOT 42M 配置
│   ├── hot_10m.yaml            # HOT 10M 配置
│   ├── rope_42m.yaml           # RoPE 基线配置
│   └── ablation/               # 消融实验配置
├── hot/                        # 核心代码
│   ├── model/                  # 模型实现
│   │   ├── frequency.py        # IntrinsicFrequency
│   │   ├── phase_dynamics.py   # PhaseDynamics
│   │   ├── phase_gating.py     # PhaseGating
│   │   ├── hot_layer.py        # HOTLayer
│   │   └── hot_model.py        # HOTModel
│   ├── training/               # 训练相关
│   │   ├── trainer.py          # Trainer
│   │   ├── optimizer.py        # 优化器配置
│   │   ├── scheduler.py        # 学习率调度
│   │   └── annealing.py        # ProgressivePhaseAnnealing
│   ├── data/                   # 数据处理
│   │   ├── dataset.py          # 数据集加载
│   │   ├── collator.py         # 数据整理
│   │   └── tokenizer.py        # 分词器
│   ├── evaluation/             # 评估模块
│   │   ├── perplexity.py       # 困惑度
│   │   ├── extrapolation.py    # 外推能力
│   │   ├── spectrum.py         # 频谱分析
│   │   └── order_parameter.py  # 序参量
│   └── utils/                  # 工具函数
│       ├── checkpoint.py       # 检查点管理
│       ├── logging.py          # 日志配置
│       └── visualization.py    # 可视化
├── scripts/                    # 训练和评估脚本
│   ├── train.py                # 训练脚本
│   ├── evaluate.py             # 评估脚本
│   ├── ablation.py             # 消融实验
│   └── visualize.py            # 可视化脚本
├── tests/                      # 单元测试
├── notebooks/                  # Jupyter notebooks
├── checkpoints/                # 模型检查点 (gitignored)
├── logs/                       # 训练日志 (gitignored)
├── data/                       # 数据集 (gitignored)
│   ├── raw/                    # 原始数据缓存
│   ├── tokenized/              # 预分词数据
│   └── splits/                 # 训练/验证集分割
│       ├── train/              # 训练集 (90%)
│       ├── val/                # 验证集 (10%)
│       ├── split_info.json     # 分割元数据
│       ├── train_indices.txt   # 训练集索引（可审计）
│       └── val_indices.txt     # 验证集索引（可审计）
└── cloudflare-monitor/         # Cloudflare 监控
```

## 关键配置

### 模型配置 (HOT 42M)

```yaml
model:
  hidden_size: 400
  num_heads: 8
  head_dim: 50
  num_layers: 12
  vocab_size: 50257
  ffn_size: 1550
```

### 训练配置

```yaml
training:
  batch_size: 23               # RTX 4060 8GB 显存上限
  gradient_accumulation: 2     # 有效 batch = 46
  max_steps: 100000
  learning_rate: 3e-4
  precision: "bf16"
  compile: true
```

## 性能指标

### 训练吞吐量

- **步骤/秒**: ~0.18 step/s
- **Token/秒**: ~4,200 tok/s
- **每步时间**: ~5.5 秒
- **预计完成**: ~44 小时剩余

### 显存使用

- **峰值显存**: ~7.3 GB (RTX 4060 8GB)
- **优化技术**:
  - 分块交叉熵: 省 ~400MB
  - 梯度检查点: 可选，与 FlexAttention 不兼容
  - FlexAttention: fused kernel，比 SDPA 快 10x+

## 常用命令

```bash
# 训练
make train

# 评估
make evaluate

# 测试
make test

# 代码检查
make check

# 可视化
make visualize

# 创建验证集分割（只需运行一次）
make create-val-split
```

## 注意事项

1. **训练中断恢复**: 使用 `latest.pt` 恢复训练
2. **显存限制**: RTX 4060 8GB 限制了 batch_size
3. **FlexAttention**: 需要 PyTorch 2.13+，与梯度检查点不兼容
4. **HF 镜像**: 自动使用 hf-mirror.com 加速下载

## 下一步

1. 完成 100K 步训练
2. 运行评估脚本
3. 执行消融实验
4. 生成可视化图表
5. 撰写实验报告

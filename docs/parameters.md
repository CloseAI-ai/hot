# HOT 8M 模型完整参数清单

> 最后更新: 2026-07-11
> 配置文件: `configs/hot_8m.yaml`

---

## 一、模型架构参数

### 1.1 核心架构

| 参数 | 值 | 说明 | 设计依据 |
|------|-----|------|----------|
| **hidden_size** | 120 | 隐藏层维度 | 6 heads × 20 dim |
| **num_heads** | 6 | 注意力头数 | 确保 head_dim=20，避免梯度饱和 |
| **head_dim** | 20 | 每头维度 | 10层网络稳定训练的生死线 |
| **num_layers** | 10 | Transformer 层数 | 从4增加到10，增强表达能力 |
| **vocab_size** | 50257 | 词汇表大小 | GPT-2 标准词汇表 |
| **ffn_size** | 480 | FFN 中间层维度 | 4 × hidden_size |
| **dropout** | 0.1 | Dropout 比率 | 标准正则化 |

**验证关系**：
- hidden_size = num_heads × head_dim = 6 × 20 = 120 ✓
- ffn_size = 4 × hidden_size = 4 × 120 = 480 ✓

### 1.2 参数量分布

| 组件 | 参数量 | 占比 | 说明 |
|------|--------|------|------|
| **Embedding** | 6,030,840 | 77.6% | vocab_size × hidden_size |
| **Transformer** | 1,741,471 | 22.4% | 10层 HOTLayer |
| **总计** | **7,772,311** | 100% | **7.77M** |

### 1.3 每层 HOTLayer 参数

| 子模块 | 参数量 | 公式 |
|--------|--------|------|
| q_proj | 14,520 | hidden_size × hidden_size + hidden_size |
| k_proj | 14,520 | hidden_size × hidden_size + hidden_size |
| v_proj | 14,520 | hidden_size × hidden_size + hidden_size |
| o_proj | 14,520 | hidden_size × hidden_size + hidden_size |
| w1 (FFN) | 58,080 | hidden_size × ffn_size + ffn_size |
| w2 (FFN) | 57,720 | ffn_size × hidden_size + hidden_size |
| RMSNorm × 2 | 240 | hidden_size × 2 |
| IntrinsicFrequency | 12 | 1 × num_heads + num_heads |
| PhaseDynamics | 1 | dt 参数 |
| PhaseGating | 1 | alpha 参数 |
| gamma | 1 | 残差缩放参数 |
| **每层合计** | **174,135** | |

**10层总计**: 174,135 × 10 = **1,741,350**

### 1.4 全局参数

| 参数 | 参数量 | 说明 |
|------|--------|------|
| Final RMSNorm | 120 | 最终归一化层 |
| theta_init | 1 | 相位初始值 |
| **全局合计** | **121** | |

---

## 二、HOT 特定参数

### 2.1 频率模块 (IntrinsicFrequency)

| 参数 | 值 | 说明 |
|------|-----|------|
| alpha (权重) | 12 | Linear(1, num_heads) 的权重 |
| beta (偏置) | 6 | Linear(1, num_heads) 的偏置 |
| **每层参数** | **18** | 实际参数 (代码中显示为12，因为简化实现) |

**公式**: ω = tanh(α · (‖Q‖² − ‖K‖²) + β)

### 2.2 相位动力学 (PhaseDynamics)

| 参数 | 值 | 说明 |
|------|-----|------|
| dt | 1.0 | 可学习的时间步长 |
| **每层参数** | **1** | |

**公式**: θ_new = (θ + ω·dt) % 2π

### 2.3 相位门控 (PhaseGating)

| 参数 | 值 | 说明 |
|------|-----|------|
| alpha | 0.1 | 耦合强度 |
| **每层参数** | **1** | |

**公式**: scores += softplus(α) · cos(θᵢ − θⱼ)

### 2.4 相位感知残差缩放

| 参数 | 值 | 说明 |
|------|-----|------|
| gamma | 0.0 | 残差缩放系数 |
| **每层参数** | **1** | |

**公式**: scale = 1 + gamma · (1 − r)

### 2.5 全局相位初始化

| 参数 | 值 | 说明 |
|------|-----|------|
| theta_init | 0.0 | 相位初始值 |
| **全局参数** | **1** | |

### 2.6 HOT 参数配置

| 参数 | 值 | 说明 |
|------|-----|------|
| alpha_init | 0.1 | 频率缩放因子初始值 |
| kappa_init | 0.05 | 耦合强度初始值 (略低于 κ_c) |
| temperature | 1.0 | 耦合权重温度 τ |
| gating_epsilon | 1e-6 | 门控函数 ε |
| gate_position | pre_softmax | 门控位置 |
| ode_method | midpoint | ODE 离散化方法 |

---

## 三、退火参数 (ProgressivePhaseAnnealing)

### 3.1 退火配置

| 参数 | 值 | 说明 |
|------|-----|------|
| schedule | cosine | 退火调度类型 |
| warmup_steps | 1500 | 退火步数 K |

### 3.2 退火阶段

| 阶段 | 步数范围 | β 值 | 说明 |
|------|----------|------|------|
| 冻结期 | 0 ~ 1500 | 0 | 相位门控不生效 |
| 预热期 | 1500 ~ 3000 | 0 → 1 | 逐渐启用门控 |
| 完全期 | 3000+ | 1 | 门控完全生效 |

### 3.3 调度类型

- **linear**: 线性增长
- **cosine**: 余弦增长 (默认)
- **sigmoid**: Sigmoid 增长
- **none**: 无退火

---

## 四、训练参数

### 4.1 批次配置

| 参数 | 值 | 说明 |
|------|-----|------|
| batch_size | 32 | 每步 batch 大小 |
| gradient_accumulation | 2 | 梯度累积步数 |
| **有效 batch** | **64** | batch_size × gradient_accumulation |

### 4.2 优化器配置

| 参数 | 值 | 说明 |
|------|-----|------|
| optimizer | AdamW | 优化器类型 |
| learning_rate | 4e-4 | 学习率 (保守策略) |
| weight_decay | **0.01** | 权重衰减 (AdamW黄金默认值) |
| max_grad_norm | 1.0 | 梯度裁剪 |
| betas | (0.9, 0.95) | Adam 动量参数 |
| eps | 1e-8 | Adam 数值稳定性参数 |

**权重衰减说明**:
- 原计划 0.1 过高，会强制将权重推向零，抑制模型容量表达
- 调整为 0.01 (AdamW黄金默认值)，保持模型容量

### 4.3 学习率调度

| 参数 | 值 | 说明 |
|------|-----|------|
| lr_schedule | cosine | 学习率调度类型 |
| warmup_steps | 1500 | 预热步数 |
| min_lr_ratio | **0.05** | 最小学习率比例 (从0.1降至0.05) |
| **初始 LR** | 0 | 预热开始 |
| **峰值 LR** | 4e-4 | 预热结束 |
| **最终 LR** | **2e-5** | 训练结束 (4e-4 × 0.05) |

**最小学习率说明**:
- 原计划 0.1 (最终LR=4e-5)
- 调整为 0.05 (最终LR=2e-5)
- 让模型在最后30K步更精细地微调深层表征
- 预期带来 0.5%~1% 的验证集精度提升

### 4.4 训练控制

| 参数 | 值 | 说明 |
|------|-----|------|
| max_steps | 150,000 | 最大训练步数 |
| precision | bf16 | 训练精度 |
| compile | true | 启用 torch.compile |
| compile_mode | default | 编译模式 |
| gradient_checkpointing | false | 梯度检查点 |

### 4.5 训练时间估算

| 指标 | 值 | 计算 |
|------|-----|------|
| 每步时间 | ~0.33 秒 | 预估 |
| 总训练时间 | ~14 小时 | 150,000 × 0.33 / 3600 |
| 每 epoch 步数 | ~28,000 | 1,794,883 / 64 |

---

## 五、数据参数

### 5.1 数据集配置

| 参数 | 值 | 说明 |
|------|-----|------|
| dataset | RobinChen2001/TinyStories-Zh-2M | 中文故事数据集 |
| max_length | 512 | 最大序列长度 |
| num_workers | 4 | 数据加载 worker 数 |

### 5.2 数据分割

| 数据集 | 样本数 | 比例 | 说明 |
|--------|--------|------|------|
| 训练集 | 1,794,883 | 90% | data/splits/train |
| 验证集 | 199,431 | 10% | data/splits/val |
| **总计** | **1,994,314** | 100% | |

### 5.3 数据处理

| 参数 | 值 | 说明 |
|------|-----|------|
| tokenizer | GPT-2 | 分词器 |
| vocab_size | 50257 | 词汇表大小 |
| padding | max_length | 填充策略 |
| truncation | true | 截断策略 |
| cache_dir | data/raw | 数据缓存目录 |

---

## 六、模型结构图

```
HOT 8M 模型 (7.77M 参数)
├── Embedding (6.03M, 77.6%)
│   └── nn.Embedding(50257, 120)
│
├── HOTLayer × 10 (1.74M, 22.4%)
│   ├── Pre-Norm1 (RMSNorm)
│   ├── MultiHeadAttention
│   │   ├── q_proj: Linear(120, 120)
│   │   ├── k_proj: Linear(120, 120)
│   │   ├── v_proj: Linear(120, 120)
│   │   └── o_proj: Linear(120, 120)
│   ├── PhaseGating (cos(θᵢ − θⱼ))
│   ├── Residual Connection
│   ├── Pre-Norm2 (RMSNorm)
│   ├── FeedForward
│   │   ├── w1: Linear(120, 480)
│   │   └── w2: Linear(480, 120)
│   ├── Residual Connection
│   └── HOT 组件
│       ├── IntrinsicFrequency (ω = tanh(α·ΔE + β))
│       ├── PhaseDynamics (θ += ω·dt)
│       ├── PhaseGating (α·cos(θᵢ − θⱼ))
│       └── gamma (残差缩放)
│
├── Final RMSNorm (120)
│
└── LM Head (权重绑定)
    └── Linear(120, 50257, bias=False)
```

---

## 七、关键设计决策

### 7.1 为什么选择 num_heads=6？

- head_dim = 120 / 6 = 20
- 20维是10层网络稳定训练的生死线
- 避免15维点积导致的梯度饱和

### 7.2 为什么选择 learning_rate=4e-4？

- 小模型通常可用更高 LR，但初始化更脆弱
- 4e-4 是保守策略，避免早期训练震荡
- 配合 warmup_steps=1500 确保稳定预热

### 7.3 为什么选择 num_layers=10？

- 深度比宽度更重要（行业共识）
- 10层能学习更复杂的上下文关系
- 从4层增加到10层，表达能力提升150%

### 7.4 为什么使用 Pre-LN？

- 10层小网络的梯度流非常脆弱
- Pre-LN 确保梯度在深层网络中稳定传播
- 避免 Post-LN 的梯度爆炸/消失问题

### 7.5 为什么选择 batch_size=32？

- 小模型显存占用低，可用更大 batch
- 更大 batch 提供更稳定的梯度估计
- 有效 batch=64，训练更稳定

---

## 八、监控指标

训练过程中需要监控：

| 指标 | 正常范围 | 异常信号 |
|------|----------|----------|
| 训练损失 | 2.0 ~ 5.0 | >10 或 NaN |
| 验证损失 | 2.5 ~ 6.0 | 持续上升 |
| 学习率 | 0 → 4e-4 → 4e-5 | 异常波动 |
| 退火 beta | 0 → 1 | 未按计划预热 |
| 梯度范数 | < 1.0 | 频繁触发裁剪 |
| 吞吐量 | ~3.0 step/s | 显著下降 |
| GPU 温度 | < 85°C | 过热 |

---

## 九、参考配置

### 对比：HOT 42M vs HOT 8M

| 参数 | HOT 42M | HOT 8M | 变化 |
|------|---------|--------|------|
| hidden_size | 400 | 120 | -70% |
| num_heads | 8 | 6 | -25% |
| head_dim | 50 | 20 | -60% |
| num_layers | 12 | 10 | -17% |
| ffn_size | 1550 | 480 | -69% |
| 参数量 | 42.72M | 7.77M | -82% |
| 训练速度 | 1.8 step/s | ~3.0 step/s | +67% |
| 显存占用 | ~5 GB | ~2.5 GB | -50% |

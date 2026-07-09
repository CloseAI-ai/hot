# HOT 开发计划

> 项目代号：HOT（Harmonic Oscillator Transformer）
> 目标：研究原型，在 125M 参数规模上验证核心假设
> 框架：PyTorch 原生
> 版本：v1.0

---

## 1. 项目概述

### 1.1 目标

在 125M 参数规模上实现 HOT 架构的完整训练流水线，验证论文中的四个核心预测：

1. 部分同步态（$\kappa \approx \kappa_c$）对应最优性能
2. 相位门控对长程依赖的提升大于短程依赖
3. 频段分化在深层更加显著
4. Scaling 行为与标准 Transformer 相似但斜率不同

### 1.2 验收标准

- [ ] HOT 125M 在 The Pile 上的困惑度与同等规模的 RoPE Transformer 相当（差距 < 5%）
- [ ] 在 8K/16K 外推测试中，HOT 的困惑度衰减优于 RoPE（衰减率低 20%+）
- [ ] 消融实验验证 PPA 策略、耦合结构、门控函数的独立贡献
- [ ] 频谱可视化展示多头频率分布的分化趋势

### 1.3 不包含

- FlashAttention CUDA kernel 定制（使用 PyTorch 原生 attention）
- 大规模分布式训练（单机多卡即可）
- 生产级推理优化（KV Cache 压缩、量化等）

---

## 2. 目录结构

```
hot/
├── README.md                   # 项目说明
├── requirements.txt            # 依赖列表
├── setup.py                    # 包安装
├── configs/                    # 训练配置
│   ├── base.yaml               # 基础配置
│   ├── hot_125m.yaml           # HOT 125M 配置
│   ├── rope_125m.yaml          # RoPE 基线配置
│   └── ablation/               # 消融实验配置
│       ├── no_gating.yaml
│       ├── full_coupling.yaml
│       └── no_annealing.yaml
├── hot/                        # 核心代码
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── transformer.py      # 标准 Transformer 基类
│   │   ├── hot_model.py        # HOT 模型
│   │   ├── hot_layer.py        # HOT 层（含相位动力学）
│   │   ├── phase_dynamics.py   # 相位 ODE 求解器
│   │   ├── phase_gating.py     # 相位同步门控
│   │   └── frequency.py        # 固有频率计算
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py          # 训练主循环
│   │   ├── annealing.py        # PPA 退火调度
│   │   ├── optimizer.py        # 优化器配置
│   │   └── scheduler.py        # 学习率调度
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py          # 数据集加载
│   │   ├── tokenizer.py        # 分词器
│   │   └── collator.py         # 数据整理
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── perplexity.py       # 困惑度计算
│   │   ├── extrapolation.py    # 长度外推测试
│   │   ├── order_parameter.py  # 序参量分析
│   │   └── spectrum.py         # 频谱可视化
│   └── utils/
│       ├── __init__.py
│       ├── logging.py          # 日志工具
│       ├── checkpoint.py       # 检查点管理
│       └── visualization.py    # 可视化工具
├── scripts/
│   ├── train.py                # 训练入口
│   ├── evaluate.py             # 评估入口
│   ├── ablation.py             # 消融实验入口
│   └── visualize.py            # 可视化入口
├── tests/
│   ├── test_phase_dynamics.py  # 相位动力学单元测试
│   ├── test_phase_gating.py    # 门控函数单元测试
│   ├── test_frequency.py       # 频率计算单元测试
│   ├── test_model.py           # 模型前向传播测试
│   └── test_annealing.py       # 退火调度测试
└── notebooks/
    ├── 01_phase_analysis.ipynb # 相位分析
    ├── 02_spectrum.ipynb       # 频谱可视化
    └── 03_scaling.ipynb        # Scaling 分析
```

---

## 3. 模块分解

### 3.1 PhaseDynamics（相位动力学模块）

**文件**：`hot/model/phase_dynamics.py`

**职责**：实现 Kuramoto ODE 的离散化求解器。

**接口设计**：

```python
class PhaseDynamics(nn.Module):
    """
    Kuramoto 相位动力学求解器
    
    实现 §2.2 的 ODE 方程：
    dθ_i/dl = ω_i + κ Σ_j w_ij sin(θ_j - θ_i)
    """
    
    def __init__(self, config):
        super().__init__()
        self.kappa = nn.Parameter(torch.tensor(config.kappa_init))
        self.tau = config.temperature  # 耦合权重温度
        
    def compute_coupling_weights(self, prev_attention: Tensor) -> Tensor:
        """
        从上一层注意力权重计算耦合权重 w_ij
        w_ij = softmax_j(Â_ij / τ)
        
        Args:
            prev_attention: [B, H, N, N] 上一层的注意力权重
        Returns:
            coupling: [B, H, N, N] 耦合权重矩阵
        """
        ...
    
    def compute_rhs(self, theta: Tensor, omega: Tensor, 
                    coupling: Tensor) -> Tensor:
        """
        计算 ODE 右端项 f(θ)
        f_i = ω_i + κ Σ_j w_ij sin(θ_j - θ_i)
        
        使用三角恒等式优化：
        Σ_j w_ij sin(θ_j - θ_i) = sin(θ_i)(-Σ_j w_ij cos(θ_j)) 
                                    + cos(θ_i)(Σ_j w_ij sin(θ_j))
        
        Args:
            theta: [B, H, N] 当前相位
            omega: [B, H, N] 固有频率
            coupling: [B, H, N, N] 耦合权重
        Returns:
            d_theta: [B, H, N] 相位导数
        """
        ...
    
    def step_euler(self, theta, omega, coupling) -> Tensor:
        """单步 Euler 更新"""
        ...
    
    def step_midpoint(self, theta, omega, coupling) -> Tensor:
        """中点法更新（默认）"""
        ...
    
    def forward(self, theta, omega, prev_attention, method='midpoint') -> Tensor:
        """
        完整的相位更新步骤
        
        Args:
            theta: [B, H, N] 当前相位
            omega: [B, H, N] 固有频率
            prev_attention: [B, H, N, N] 上一层注意力权重
            method: 'euler' | 'midpoint'
        Returns:
            theta_new: [B, H, N] 更新后的相位（已归一化到 [0, 2π)）
        """
        ...
```

**关键实现细节**：
- 三角恒等式优化：将 $O(N^2)$ 的逐对 sin 计算转化为两次 $O(N)$ 的加权求和
- 相位归一化：`theta_new = theta_new % (2 * math.pi)`
- 耦合强度 $\kappa$ 作为可学习参数，使用 `clamp(0, 2)` 约束

**单元测试**：
- `test_phase_wrapping`：验证相位归一化到 $[0, 2\pi)$
- `test_coupling_weights_sum_to_one`：验证 softmax 归一化
- `test_euler_vs_midpoint`：比较两种方法的精度
- `test_triangle_identity`：验证三角恒等式优化的正确性

---

### 3.2 PhaseGating（相位同步门控模块）

**文件**：`hot/model/phase_gating.py`

**职责**：实现门控函数 $\Phi(\Delta\theta)$ 及其与注意力分数的融合。

```python
class PhaseGating(nn.Module):
    """
    相位同步门控机制
    
    实现 §3.1 的门控函数：
    Φ(Δθ) = ReLU(cos(Δθ)) + ε
    """
    
    def __init__(self, config):
        super().__init__()
        self.epsilon = config.gating_epsilon  # 默认 1e-6
        self.gate_position = config.gate_position  # 'pre_softmax' | 'post_softmax'
        
    def compute_phase_matrix(self, theta: Tensor) -> Tensor:
        """
        计算相位差矩阵
        
        Args:
            theta: [B, H, N] 相位
        Returns:
            delta_theta: [B, H, N, N] 相位差矩阵 Δθ_ij = θ_i - θ_j
        """
        ...
    
    def gating_function(self, delta_theta: Tensor) -> Tensor:
        """
        门控函数 Φ(Δθ) = ReLU(cos(Δθ)) + ε
        
        Args:
            delta_theta: [B, H, N, N] 相位差
        Returns:
            gate: [B, H, N, N] 门控值，范围 [ε, 1+ε]
        """
        return F.relu(torch.cos(delta_theta)) + self.epsilon
    
    def forward(self, attn_scores: Tensor, theta: Tensor) -> Tensor:
        """
        将门控应用到注意力分数
        
        Args:
            attn_scores: [B, H, N, N] Q·K/√d 注意力分数
            theta: [B, H, N] 当前相位
        Returns:
            gated_scores: [B, H, N, N] 门控后的注意力分数
        """
        delta_theta = self.compute_phase_matrix(theta)
        gate = self.gating_function(delta_theta)
        
        if self.gate_position == 'pre_softmax':
            return attn_scores * gate
        else:  # post_softmax
            attn_weights = F.softmax(attn_scores, dim=-1)
            return attn_weights * gate
```

**单元测试**：
- `test_gate_at_zero`：$\Delta\theta = 0$ 时 $\Phi = 1 + \epsilon$
- `test_gate_at_pi_half`：$\Delta\theta = \pi/2$ 时 $\Phi = \epsilon$
- `test_gate_gradient`：验证梯度在 $\Delta\theta \approx \pi/4$ 处最大
- `test_gate_range`：验证 $\Phi \in [\epsilon, 1+\epsilon]$

---

### 3.3 Frequency（固有频率模块）

**文件**：`hot/model/frequency.py`

**职责**：实现 §2.1 的频率计算。

```python
class IntrinsicFrequency(nn.Module):
    """
    Token 固有频率计算
    
    实现 §2.1：
    ω_i = tanh(α · (Q_i^(1)·K_i^(1) - Q_i^(2)·K_i^(2)))
    """
    
    def __init__(self, config):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(config.alpha_init))  # 默认 0.1
        self.head_dim = config.head_dim
        
    def forward(self, Q: Tensor, K: Tensor) -> Tensor:
        """
        计算每个 Token 的固有频率
        
        Args:
            Q: [B, H, N, d] Query 向量
            K: [B, H, N, d] Key 向量
        Returns:
            omega: [B, H, N] 固有频率，范围 (-1, 1)
        """
        # 分割为两组分量
        d_half = self.head_dim // 2
        Q1, Q2 = Q[..., :d_half], Q[..., d_half:]
        K1, K2 = K[..., :d_half], K[..., d_half:]
        
        # Q-K 对称交互
        interaction = (Q1 * K1).sum(-1) - (Q2 * K2).sum(-1)
        
        return torch.tanh(self.alpha * interaction)
```

**单元测试**：
- `test_frequency_range`：验证 $\omega \in (-1, 1)$
- `test_frequency_symmetry`：验证 Q-K 交换时频率变号（物理直觉）
- `test_alpha_gradient`：验证 $\alpha$ 可正常接收梯度

---

### 3.4 HOTLayer（HOT 层）

**文件**：`hot/model/hot_layer.py`

**职责**：整合相位动力学、门控和注意力计算。

```python
class HOTLayer(nn.Module):
    """
    HOT 层：标准 Transformer 层 + 相位动力学
    
    前向传播流程（对应论文 Algorithm 1）：
    1. 计算 Q, K, V 投影
    2. 计算固有频率 ω
    3. 计算耦合权重 w（来自上层注意力）
    4. 相位 ODE 更新
    5. 计算门控矩阵 Φ
    6. 计算门控注意力
    7. 输出投影
    """
    
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        
        # 标准组件
        self.attn = MultiHeadAttention(config)
        self.ffn = FeedForward(config)
        self.norm1 = RMSNorm(config.hidden_size)
        self.norm2 = RMSNorm(config.hidden_size)
        
        # HOT 组件
        self.frequency = IntrinsicFrequency(config)
        self.phase_dynamics = PhaseDynamics(config)
        self.phase_gating = PhaseGating(config)
        
    def forward(self, x: Tensor, theta: Tensor, 
                prev_attention: Tensor, 
                annealing_beta: float = 1.0) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            x: [B, N, D] 隐藏状态
            theta: [B, H, N] 当前相位
            prev_attention: [B, H, N, N] 上一层注意力权重
            annealing_beta: 退火系数 β ∈ [0, 1]
        Returns:
            x_new: [B, N, D] 更新后的隐藏状态
            theta_new: [B, H, N] 更新后的相位
            attn_weights: [B, H, N, N] 当前层注意力权重
        """
        # 1. Q, K, V 投影
        residual = x
        x = self.norm1(x)
        Q, K, V = self.attn.project(x)
        
        # 2. 计算固有频率
        omega = self.frequency(Q, K)
        
        # 3-4. 相位更新
        theta_new = self.phase_dynamics(theta, omega, prev_attention)
        
        # 5. 计算门控注意力
        attn_scores = self.attn.compute_scores(Q, K)
        
        # 退火混合：Φ' = (1-β) + β·Φ
        if annealing_beta < 1.0:
            gate = self.phase_gating.gating_function(
                self.phase_gating.compute_phase_matrix(theta_new)
            )
            gate = (1 - annealing_beta) + annealing_beta * gate
            attn_scores = attn_scores * gate
            attn_weights = F.softmax(attn_scores, dim=-1)
        else:
            attn_weights = self.phase_gating(attn_scores, theta_new)
            if self.phase_gating.gate_position == 'pre_softmax':
                attn_weights = F.softmax(attn_weights, dim=-1)
        
        # 6. 注意力输出
        attn_out = self.attn.compute_output(attn_weights, V)
        x = residual + attn_out
        
        # 7. FFN
        x = x + self.ffn(self.norm2(x))
        
        return x, theta_new, attn_weights
```

---

### 3.5 HOTModel（完整模型）

**文件**：`hot/model/hot_model.py`

**职责**：完整的 HOT 模型，包含嵌入层、多层 HOTLayer、输出头。

```python
class HOTModel(nn.Module):
    """
    完整的 HOT 模型
    
    结构：Embedding → N × HOTLayer → RMSNorm → LM Head
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            HOTLayer(config, i) for i in range(config.num_layers)
        ])
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # 相位状态（不作为参数，作为 buffer）
        self.register_buffer('theta_init', torch.zeros(1))
        
    def forward(self, input_ids: Tensor, labels: Tensor = None) -> Dict:
        """
        Args:
            input_ids: [B, N] 输入 token IDs
            labels: [B, N] 标签（训练时）
        Returns:
            logits: [B, N, V] 输出 logits
            loss: 标量损失（如果提供了 labels）
            aux: 辅助信息（相位、注意力权重等）
        """
        B, N = input_ids.shape
        H = self.config.num_heads
        
        # 初始化相位为全零
        theta = torch.zeros(B, H, N, device=input_ids.device)
        prev_attention = None
        
        x = self.embed(input_ids)
        
        # 逐层前向传播
        all_thetas = []
        all_attentions = []
        
        for layer in self.layers:
            # 计算当前层的退火系数
            beta = self.annealing_schedule(self.global_step)
            
            x, theta, attn_weights = layer(x, theta, prev_attention, beta)
            
            all_thetas.append(theta)
            all_attentions.append(attn_weights)
            prev_attention = attn_weights
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return {
            'logits': logits,
            'loss': loss,
            'thetas': all_thetas,
            'attentions': all_attentions,
        }
```

---

### 3.6 Annealing（退火调度模块）

**文件**：`hot/training/annealing.py`

**职责**：实现 PPA 策略的三种退火调度。

```python
class ProgressivePhaseAnnealing:
    """
    渐进式相位退火调度
    
    实现 §4.2 的三种调度：
    - 线性退火：β(t) = min(1, (t-K)/K)
    - 余弦退火：β(t) = 0.5 * (1 - cos(π*(t-K)/K))
    - Sigmoid 退火：β(t) = σ((t-1.5K)/(K/5))
    """
    
    def __init__(self, warmup_steps: int, schedule: str = 'linear'):
        self.K = warmup_steps
        self.schedule = schedule
        
    def get_beta(self, step: int) -> float:
        """返回当前步的退火系数 β ∈ [0, 1]"""
        if step < self.K:
            return 0.0  # 冻结期
        
        t = step - self.K
        
        if self.schedule == 'linear':
            return min(1.0, t / self.K)
        elif self.schedule == 'cosine':
            return 0.5 * (1 - math.cos(math.pi * t / self.K))
        elif self.schedule == 'sigmoid':
            return torch.sigmoid(torch.tensor((t - 1.5 * self.K) / (self.K / 5))).item()
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")
```

---

### 3.7 Evaluation（评估模块）

**文件**：`hot/evaluation/`

**核心模块**：

```python
# order_parameter.py
def compute_order_parameter(thetas: Tensor) -> Tensor:
    """
    计算序参量 r = |1/N Σ_j exp(iθ_j)|
    
    Args:
        thetas: [B, H, N] 相位
    Returns:
        r: [B, H] 序参量，范围 [0, 1]
    """
    complex_phases = torch.exp(1j * thetas)
    r = torch.abs(complex_phases.mean(dim=-1))
    return r

# spectrum.py
def compute_frequency_spectrum(model: HOTModel, dataloader) -> Dict:
    """
    分析各头的频率分布
    
    Returns:
        {
            'freq_mean': [L, H] 各层各头的平均频率
            'freq_std': [L, H] 各层各头的频率标准差
            'freq_distribution': [L, H, B] 频率分布直方图
        }
    """
    ...

# extrapolation.py
def length_extrapolation_test(model, train_len, test_lengths):
    """
    长度外推测试
    
    Args:
        model: 训练好的模型
        train_len: 训练时的最大长度
        test_lengths: [8192, 16384, 32768, ...] 测试长度列表
    Returns:
        perplexities: Dict[int, float] 各长度的困惑度
    """
    ...
```

---

## 4. 开发阶段

### 阶段一：核心组件实现（第 1-2 周）

**目标**：实现 HOT 的核心计算组件，通过单元测试验证正确性。

| 任务 | 文件 | 验收标准 | 预估工时 |
|------|------|----------|----------|
| T1.1 相位动力学求解器 | `phase_dynamics.py` | 4 个单元测试通过 | 2 天 |
| T1.2 相位门控模块 | `phase_gating.py` | 4 个单元测试通过 | 1 天 |
| T1.3 固有频率计算 | `frequency.py` | 3 个单元测试通过 | 1 天 |
| T1.4 HOT 层集成 | `hot_layer.py` | 前向传播无报错，输出形状正确 | 2 天 |
| T1.5 退火调度 | `annealing.py` | 三种调度曲线可视化正确 | 0.5 天 |
| T1.6 数据管道 | `data/` | The Pile 数据加载、分词、batching | 2 天 |

**里程碑 M1**：`tests/` 全部通过，HOT 层可在 dummy 数据上前向传播。

---

### 阶段二：模型训练与基线对比（第 3-4 周）

**目标**：完成 HOT 125M 和 RoPE 125M 的训练，获取基线困惑度。

| 任务 | 文件 | 验收标准 | 预估工时 |
|------|------|----------|----------|
| T2.1 HOT 完整模型 | `hot_model.py` | 参数量 ≈ 125M | 2 天 |
| T2.2 RoPE 基线模型 | `transformer.py` | 参数量 ≈ 125M | 1 天 |
| T2.3 训练循环 | `trainer.py` | 支持混合精度、梯度累积 | 2 天 |
| T2.4 HOT 125M 训练 | `train.py` | 困惑度收敛，无 NaN/Inf | 5 天 |
| T2.5 RoPE 125M 训练 | `train.py` | 作为基线对照 | 3 天 |
| T2.6 基础评估 | `perplexity.py` | 输出困惑度报告 | 1 天 |

**里程碑 M2**：HOT 125M 困惑度与 RoPE 125M 差距 < 5%。

---

### 阶段三：消融实验与验证（第 5-6 周）

**目标**：通过消融实验验证各组件的独立贡献，验证四个理论预测。

| 任务 | 文件 | 验收标准 | 预估工时 |
|------|------|----------|----------|
| T3.1 门控函数消融 | `ablation.py` | 3 种配置对比 | 2 天 |
| T3.2 耦合结构消融 | `ablation.py` | 3 种配置对比 | 2 天 |
| T3.3 退火调度消融 | `ablation.py` | 4 种配置对比 | 2 天 |
| T3.4 耦合强度消融 | `ablation.py` | 5 个 κ 值对比 | 3 天 |
| T3.5 长度外推测试 | `extrapolation.py` | 8K/16K/32K 困惑度曲线 | 2 天 |
| T3.6 序参量分析 | `order_parameter.py` | 绘制 r 随训练步数的变化 | 1 天 |
| T3.7 频谱可视化 | `spectrum.py` | 各头频率分布热力图 | 1 天 |

**里程碑 M3**：
- 预测 1 验证：耦合强度消融显示 $\kappa \approx \kappa_c$ 时最优
- 预测 2 验证：长度外推测试显示 HOT 衰减率低于 RoPE
- 预测 3 验证：频谱可视化显示深层频率分布出现多峰

---

### 阶段四：分析、文档与发布（第 7 周）

**目标**：整理实验结果，撰写分析报告，准备代码开源。

| 任务 | 文件 | 验收标准 | 预估工时 |
|------|------|----------|----------|
| T4.1 实验结果汇总 | `notebooks/` | 所有图表生成 | 2 天 |
| T4.2 结果分析报告 | `results.md` | 含所有消融实验的结论 | 2 天 |
| T4.3 代码清理 | 全部 | 代码注释、类型标注、docstring | 1 天 |
| T4.4 README 撰写 | `README.md` | 含安装、使用、复现说明 | 0.5 天 |
| T4.5 论文更新 | `article.md` | 根据实验结果更新论文 | 1 天 |

**里程碑 M4**：代码可复现，实验结果可支撑论文的核心主张。

---

## 5. 训练配置

### 5.1 模型配置（HOT 125M）

```yaml
# configs/hot_125m.yaml
model:
  hidden_size: 768
  num_heads: 12
  head_dim: 64
  num_layers: 12
  vocab_size: 50257
  ffn_size: 3072
  dropout: 0.1

hot:
  alpha_init: 0.1              # 频率缩放因子初始值
  kappa_init: 0.05             # 耦合强度初始值（略低于 κ_c）
  temperature: 1.0             # 耦合权重温度 τ
  gating_epsilon: 1e-6         # 门控函数 ε
  gate_position: "pre_softmax" # 门控位置
  ode_method: "midpoint"       # ODE 离散化方法
  annealing:
    schedule: "cosine"         # 退火调度类型
    warmup_steps: 2000         # 退火步数 K

training:
  batch_size: 8
  gradient_accumulation: 4     # 有效 batch = 32
  max_steps: 100000
  learning_rate: 3e-4
  lr_schedule: "cosine"
  warmup_steps: 2000
  weight_decay: 0.1
  max_grad_norm: 1.0
  precision: "bf16"

data:
  dataset: "the_pile"
  max_length: 2048
  num_workers: 8
```

### 5.2 基线配置（RoPE 125M）

```yaml
# configs/rope_125m.yaml
model:
  hidden_size: 768
  num_heads: 12
  head_dim: 64
  num_layers: 12
  vocab_size: 50257
  ffn_size: 3072
  dropout: 0.1
  position_encoding: "rope"

training:
  # 与 HOT 相同的训练配置
  ...
```

### 5.3 硬件需求

| 配置 | 最低要求 | 推荐配置 |
|------|----------|----------|
| GPU | 1× A100 40GB | 4× A100 80GB |
| 内存 | 64GB | 128GB |
| 存储 | 500GB SSD | 1TB NVMe |
| 训练时间 | ~7 天（单卡） | ~2 天（4 卡） |

---

## 6. 依赖列表

```
# requirements.txt
torch>=2.1.0
transformers>=4.35.0
datasets>=2.14.0
tokenizers>=0.15.0
wandb>=0.16.0
pyyaml>=6.0
matplotlib>=3.7.0
seaborn>=0.12.0
numpy>=1.24.0
tqdm>=4.65.0
```

---

## 7. 风险与应对

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| 相位动力学导致训练不稳定 | 中 | 高 | PPA 策略 + 降低 κ_init + 梯度裁剪 |
| HOT 困惑度显著差于 RoPE | 中 | 高 | 逐步排查：先关闭门控验证基础功能，再逐步启用 |
| 消融实验结果不显著 | 低 | 中 | 增大模型规模或训练步数 |
| 训练速度过慢 | 低 | 中 | 三角恒等式优化 + 混合精度 |
| 数据加载瓶颈 | 低 | 低 | 预处理 + 多 worker |

---

## 8. 里程碑总览

```
第 1 周  ████████░░░░░░░░░░░░░░░░░░░░  M1: 核心组件
第 2 周  ████████████░░░░░░░░░░░░░░░░  M1: 组件集成
第 3 周  ████████████████░░░░░░░░░░░░  M2: 模型训练
第 4 周  ████████████████████░░░░░░░░  M2: 基线对比
第 5 周  ████████████████████████░░░░  M3: 消融实验
第 6 周  ████████████████████████████  M3: 验证预测
第 7 周  ████████████████████████████  M4: 文档发布
```

---

## 9. 后续扩展（不在 v1.0 范围内）

- [ ] FlashAttention CUDA kernel 集成
- [ ] 分布式训练支持（FSDP / DeepSpeed）
- [ ] 350M / 1.3B 规模扩展
- [ ] 多模态扩展（视觉 Token）
- [ ] KV Cache 压缩与推理优化
- [ ] 与其他架构的公平对比（RWKV、Mamba）

"""
HOT 层模块

整合频率-相位解耦动力学、相位门控和注意力计算。

优化：
- 共享 sin/cos 计算（phase_gating + causal_order_parameter 共用）
- 预计算 position 索引（注册为 buffer）
- 使用 FlexAttention（编译后 fused kernel，比 SDPA math backend 快 10x+）
- 回退到 F.scaled_dot_product_attention（自动选择 FlashAttention）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .phase_dynamics import PhaseDynamics
from .phase_gating import PhaseGating
from .frequency import IntrinsicFrequency

# FlexAttention 可用性检测
_FLEX_ATTENTION_AVAILABLE = False
try:
    from torch.nn.attention.flex_attention import (
        flex_attention as _raw_flex_attention,
        create_block_mask,
    )
    # 必须在 import 时编译 flex_attention 本身，否则会回退到
    # 未融合的实现（实体化完整 attention matrix，极慢）
    _compiled_flex_attention = torch.compile(_raw_flex_attention)

    @torch._dynamo.allow_in_graph
    def _flex_attention_call(Q, K, V, score_mod, block_mask):
        """允许 dynamo 将此函数纳入计算图，避免 graph break。
        内部的 _compiled_flex_attention 已预编译，梯度通过 autograd 正常流动。"""
        return _compiled_flex_attention(Q, K, V, score_mod=score_mod,
                                        block_mask=block_mask)

    _FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    pass


class MultiHeadAttention(nn.Module):
    """标准多头注意力（使用原生 SDPA）"""

    def __init__(self, config):
        super().__init__()
        if isinstance(config, dict):
            mc = config.get('model', {})
            self.num_heads = mc.get('num_heads', 12)
            self.head_dim = mc.get('head_dim', 64)
            self.hidden_size = mc.get('hidden_size', 768)
        else:
            self.num_heads = config.model.num_heads
            self.head_dim = config.model.head_dim
            self.hidden_size = config.model.hidden_size

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)

    def project(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """投影到 Q, K, V 并重塑为多头形式"""
        B, N, _ = x.shape
        Q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        return Q, K, V

    def compute_output(self, attn_weights: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """计算注意力输出并投影"""
        out = attn_weights @ V
        B, H, N, D = out.shape
        out = out.transpose(1, 2).reshape(B, N, H * D)
        return self.o_proj(out)


class FeedForward(nn.Module):
    """前馈网络"""

    def __init__(self, config):
        super().__init__()
        if isinstance(config, dict):
            mc = config.get('model', {})
            h = mc.get('hidden_size', 768)
            f = mc.get('ffn_size', 3072)
            d = mc.get('dropout', 0.1)
        else:
            h, f, d = config.model.hidden_size, config.model.ffn_size, config.model.dropout

        self.w1 = nn.Linear(h, f)
        self.w2 = nn.Linear(f, h)
        self.dropout = nn.Dropout(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.gelu(self.w1(x))))


class RMSNorm(nn.Module):
    """RMS 归一化"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


def causal_order_parameter(cos_theta: torch.Tensor, sin_theta: torch.Tensor,
                           pos: torch.Tensor) -> torch.Tensor:
    """
    因果序参量：仅使用当前位置及之前的相位信息

    r_i = |1/i Σ_{j=1}^{i} exp(iθ_j)|

    使用纯实数运算（bf16 兼容）：
    - cos 累积和：C_i = Σ_{j<=i} cos(θ_j)
    - sin 累积和：S_i = Σ_{j<=i} sin(θ_j)
    - r_i = sqrt(C_i² + S_i²) / i

    Args:
        cos_theta: [B, H, N] 预计算的 cos(θ)
        sin_theta: [B, H, N] 预计算的 sin(θ)
        pos: [N] 预计算的位置索引 [1, 2, ..., N]
    Returns:
        r: [B, N] 因果序参量，范围 [0, 1]，对 head 取平均
    """
    # 因果累积和
    cos_cumsum = cos_theta.cumsum(dim=-1)  # [B, H, N]
    sin_cumsum = sin_theta.cumsum(dim=-1)  # [B, H, N]

    # r_i = sqrt(C_i² + S_i²) / i
    # 添加 eps 防止 sqrt(0) 和除零的数值问题
    r_sq = cos_cumsum.pow(2) + sin_cumsum.pow(2)
    r = torch.sqrt(r_sq + 1e-8) / (pos + 1e-8)  # [B, H, N]
    r = torch.clamp(r, max=1.0)  # r 理论上 ∈ [0,1]，clamp 防止浮点溢出

    # 对 head 取平均 → [B, N]
    return r.mean(dim=1)


class HOTLayer(nn.Module):
    """
    HOT 层：Transformer + 频率-相位解耦动力学 + 相位感知残差缩放

    前向传播流程：
    1. Q, K, V 投影
    2. 计算固有频率 ω（由 Q/K 能量差决定）
    3. 更新相位 θ（θ ← θ + ω·Δt，无耦合依赖）
    4. 计算注意力分数 + 残差耦合相位门控 cos(θ_i - θ_j)
    5. 计算因果序参量 r_i → 残差缩放 s = 1 + γ(1-r)
    6. 输出投影 + FFN
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        if isinstance(config, dict):
            hidden_size = config.get('model', {}).get('hidden_size', 768)
        else:
            hidden_size = config.model.hidden_size

        self.attn = MultiHeadAttention(config)
        self.ffn = FeedForward(config)
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)

        # HOT 组件
        self.frequency = IntrinsicFrequency(config)
        self.phase_dynamics = PhaseDynamics(config)
        self.phase_gating = PhaseGating(config)

        # 相位感知残差缩放参数 γ（初始化为 0 -> 标准残差）
        self.gamma = nn.Parameter(torch.zeros(1))

        # FlexAttention block mask（由 HOTModel 设置，作为模块属性
        # 而非函数参数传递，避免 dynamo graph break）
        self._flex_block_mask = None

    def forward(self, x: torch.Tensor, theta: torch.Tensor,
                annealing_beta: float = 1.0,
                pos: torch.Tensor = None,
                causal_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, N, D] 隐藏状态
            theta: [B, H, N] 当前相位
            annealing_beta: 退火系数 β ∈ [0, 1]
            pos: [N] 预计算的位置索引
            causal_mask: [N, N] 预计算的因果布尔 mask（SDPA 回退路径用）
            flex_block_mask: FlexAttention 的因果 block mask（编译路径用）
        Returns:
            x_new: [B, N, D] 更新后的隐藏状态
            theta_new: [B, H, N] 更新后的相位
            attn_weights: None（训练时不返回以节省内存）
        """
        # 1. Pre-norm + Q, K, V 投影
        residual = x
        x_norm = self.norm1(x)
        Q, K, V = self.attn.project(x_norm)

        # 2. 计算固有频率（由 Q/K 能量差决定）
        omega = self.frequency(Q, K)  # [B, H, N]

        # 3. 更新相位（跨层传递，无耦合依赖）
        theta_new = self.phase_dynamics(theta, omega)  # [B, H, N]

        # 从 theta_new 计算 sin/cos（确保梯度流经 frequency -> omega -> theta_new）
        cos_theta = torch.cos(theta_new)
        sin_theta = torch.sin(theta_new)

        # 4. 注意力 + 相位门控
        if self.phase_gating.gate_position == 'none' or annealing_beta == 0.0:
            # 纯标准注意力，使用原生 SDPA（自动选择 FlashAttention）
            attn_out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        elif self._flex_block_mask is not None:
            # FlexAttention 路径：on-the-fly score_mod（避免实体化 [B,H,N,N] 张量）
            #
            # 将 phase bias = β·α·(cos_i·cos_j + sin_i·sin_j) 分解为
            # 4 个独立张量各索引一次：
            #   cos_q[i] · cos_kv[j] + sin_q[i] · sin_kv[j]
            # 其中 cos_q/sin_q 被 detach（不通过 FlexAttention backward），
            # cos_kv/sin_kv 保持可微（梯度流回 alpha, theta_new）
            # 这避免了 PyTorch 2.13 FlexAttention backward 的两个 bug：
            #   1. 同一张量多次索引
            #   2. requires_grad 标量捕获
            alpha = torch.nn.functional.softplus(self.phase_gating.alpha)
            alpha = torch.clamp(alpha, max=5.0)
            scale = (annealing_beta * alpha).sqrt()

            cos_kv = cos_theta * scale   # [B, H, N], requires_grad
            sin_kv = sin_theta * scale   # [B, H, N], requires_grad
            cos_q = cos_kv.detach().clone()  # 独立副本，无梯度
            sin_q = sin_kv.detach().clone()

            def score_mod(score, b, h, q_idx, kv_idx):
                return score + (
                    cos_q[b, h, q_idx] * cos_kv[b, h, kv_idx] +
                    sin_q[b, h, q_idx] * sin_kv[b, h, kv_idx]
                )

            attn_out = _flex_attention_call(Q, K, V, score_mod=score_mod,
                                            block_mask=self._flex_block_mask)
        else:
            # SDPA 回退路径（math backend，慢但兼容性好）
            alpha = torch.nn.functional.softplus(self.phase_gating.alpha)
            alpha = torch.clamp(alpha, max=5.0)

            cos_delta = (cos_theta.unsqueeze(-1) * cos_theta.unsqueeze(-2) +
                         sin_theta.unsqueeze(-1) * sin_theta.unsqueeze(-2))

            phase_bias = annealing_beta * alpha * cos_delta
            attn_mask = phase_bias.masked_fill(causal_mask, float('-inf'))

            attn_out = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask)

        B, H, N, D = attn_out.shape
        attn_out = attn_out.transpose(1, 2).reshape(B, N, H * D)
        attn_out = self.attn.o_proj(attn_out)

        # 5. 因果序参量 → 相位感知残差缩放
        r = causal_order_parameter(cos_theta, sin_theta, pos)  # [B, N]
        scale_val = 1.0 + self.gamma * (1.0 - r)  # [B, N]
        scale_val = scale_val.unsqueeze(-1)  # [B, N, 1]

        # 6. 缩放残差 + 注意力输出 + FFN
        x = scale_val * residual + attn_out
        x = x + self.ffn(self.norm2(x))

        return x, theta_new, None

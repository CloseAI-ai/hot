"""
HOT 层模块

整合频率-相位解耦动力学、相位门控和注意力计算。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .phase_dynamics import PhaseDynamics
from .phase_gating import PhaseGating
from .frequency import IntrinsicFrequency


class MultiHeadAttention(nn.Module):
    """标准多头注意力"""

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

    def compute_scores(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """计算注意力分数 Q·K/√d"""
        scale = self.head_dim ** -0.5
        return torch.matmul(Q, K.transpose(-2, -1)) * scale

    def compute_output(self, attn_weights: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """计算注意力输出并投影"""
        B, H, N, D = (attn_weights @ V).shape
        out = (attn_weights @ V).transpose(1, 2).contiguous().view(B, N, H * D)
        return self.o_proj(out)


class FeedForward(nn.Module):
    """前馈网络（SwiGLU 风格的两层 FFN）"""

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


def causal_order_parameter(theta: torch.Tensor) -> torch.Tensor:
    """
    因果序参量：仅使用当前位置及之前的相位信息

    r_i = |1/i Σ_{j=1}^{i} exp(iθ_j)|

    Args:
        theta: [B, H, N] 相位
    Returns:
        r: [B, N] 因果序参量，范围 [0, 1]，对 head 取平均
    """
    # exp(iθ) → 复数表示
    z = torch.exp(1j * theta)  # [B, H, N] complex

    # 因果累积和：S_i = Σ_{j=1}^{i} z_j
    S = z.cumsum(dim=-1)  # [B, H, N] complex

    # 位置索引：[1, 2, ..., N]
    N = theta.shape[-1]
    pos = torch.arange(1, N + 1, device=theta.device, dtype=torch.float32)  # [N]

    # r_i = |S_i / i|
    r = (S.abs() / pos)  # [B, H, N]

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

        # 相位感知残差缩放参数 γ（初始化为 0 → 标准残差）
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, theta: torch.Tensor,
                annealing_beta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, N, D] 隐藏状态
            theta: [B, H, N] 当前相位
            annealing_beta: 退火系数 β ∈ [0, 1]
        Returns:
            x_new: [B, N, D] 更新后的隐藏状态
            theta_new: [B, H, N] 更新后的相位
            attn_weights: [B, H, N, N] 当前层注意力权重
        """
        # 1. Pre-norm + Q, K, V 投影
        residual = x
        x_norm = self.norm1(x)
        Q, K, V = self.attn.project(x_norm)

        # 2. 计算固有频率（由 Q/K 能量差决定）
        omega = self.frequency(Q, K)  # [B, H, N]

        # 3. 更新相位（跨层传递，无耦合依赖）
        theta_new = self.phase_dynamics(theta, omega)  # [B, H, N]

        # 4. 注意力分数 + 残差耦合相位门控
        attn_scores = self.attn.compute_scores(Q, K)  # [B, H, N, N]

        if self.phase_gating.gate_position == 'none' or annealing_beta == 0.0:
            attn_weights = F.softmax(attn_scores, dim=-1)
        else:
            gated_scores = self.phase_gating(attn_scores, theta_new)

            if annealing_beta < 1.0:
                gated_scores = (1 - annealing_beta) * attn_scores + annealing_beta * gated_scores

            attn_weights = F.softmax(gated_scores, dim=-1)

        attn_out = self.attn.compute_output(attn_weights, V)

        # 5. 因果序参量 → 相位感知残差缩放
        # r_i = |1/i Σ_{j≤i} exp(iθ_j)|  （因果，不泄露未来信息）
        # s_i = 1 + γ·(1 - r_i)           （相位越分散，残差越强）
        r = causal_order_parameter(theta_new)  # [B, N]
        scale = 1.0 + self.gamma * (1.0 - r)  # [B, N]
        scale = scale.unsqueeze(-1)  # [B, N, 1]，广播到 [B, N, D]

        # 6. 缩放残差 + 注意力输出 + FFN
        x = scale * residual + attn_out
        x = x + self.ffn(self.norm2(x))

        return x, theta_new, attn_weights

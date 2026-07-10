"""
相位门控模块（残差耦合方案）

核心思想：相位门控只做调制，不做开关。

Score_ij = Q·K/√d + α·cos(θ_i - θ_j)
attn_weights = softmax(Score_ij)

优化：接收预计算的 sin/cos，避免重复 trig 计算。
"""

import torch
import torch.nn as nn


class PhaseGating(nn.Module):
    """
    残差耦合相位门控

    在 logit 空间加性调制：Score += α·cos(θ_i - θ_j)

    - α 是可学习标量，初始化为 0.1
    - 当 α=0 退化为标准注意力
    - cos(Δθ) 在 logit 空间做加性偏置，永远不会"关闭"内容注意力
    """

    def __init__(self, config):
        super().__init__()
        if isinstance(config, dict):
            hot_config = config.get('hot', {})
            self.gate_position = hot_config.get('gate_position', 'pre_softmax')
            alpha_init = hot_config.get('alpha_init', 0.1)
        else:
            self.gate_position = config.hot.gate_position
            alpha_init = config.hot.alpha_init

        # 可学习的耦合强度
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, attn_scores: torch.Tensor,
                cos_theta: torch.Tensor, sin_theta: torch.Tensor) -> torch.Tensor:
        """
        残差耦合门控

        Args:
            attn_scores: [B, H, N, N] Q·K/√d 注意力分数
            cos_theta: [B, H, N] 预计算的 cos(θ)
            sin_theta: [B, H, N] 预计算的 sin(θ)
        Returns:
            gated_scores: [B, H, N, N] 调制后的注意力分数（未 softmax）
        """
        if self.gate_position == 'none':
            return attn_scores

        # 使用 softplus 确保 α 非负，并 clamp 防止过大
        alpha = torch.nn.functional.softplus(self.alpha)
        alpha = torch.clamp(alpha, max=5.0)

        # cos(θ_i - θ_j) = cos(θ_i)cos(θ_j) + sin(θ_i)sin(θ_j)
        # 外积计算：[B, H, N, 1] * [B, H, 1, N] → [B, H, N, N]
        cos_delta = (cos_theta.unsqueeze(-1) * cos_theta.unsqueeze(-2) +
                     sin_theta.unsqueeze(-1) * sin_theta.unsqueeze(-2))

        # clamp 到 [-1, 1] 防止浮点误差累积
        cos_delta = torch.clamp(cos_delta, -1.0, 1.0)

        # 残差耦合：attn_scores += α * cos(Δθ)
        return attn_scores + alpha * cos_delta

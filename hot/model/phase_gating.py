"""
相位门控模块（残差耦合方案）

核心思想：相位门控只做调制，不做开关。

Score_ij = Q·K/√d + α·cos(θ_i - θ_j)
attn_weights = softmax(Score_ij)

- α 是可学习标量，初始化为 0.1
- 当 α=0 退化为标准注意力
- cos(Δθ) 在 logit 空间做加性偏置，永远不会"关闭"内容注意力
"""

import torch
import torch.nn as nn


class PhaseGating(nn.Module):
    """
    残差耦合相位门控

    在 logit 空间加性调制：Score += α·cos(θ_i - θ_j)

    - 同相（cos=+1）：增强注意力 → 语义节律同步的 token 更受关注
    - 反相（cos=-1）：削弱注意力，但不归零 → 内容信息始终保留
    - α=0：退化为标准 Transformer
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

        # 可学习的耦合强度，初始化为小值
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, attn_scores: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        残差耦合门控

        Args:
            attn_scores: [B, H, N, N] Q·K/√d 注意力分数
            theta: [B, H, N] 当前相位
        Returns:
            gated_scores: [B, H, N, N] 调制后的注意力分数（未 softmax）
        """
        if self.gate_position == 'none':
            return attn_scores

        # 相位差矩阵：Δθ_ij = θ_i - θ_j
        delta_theta = theta.unsqueeze(-1) - theta.unsqueeze(-2)

        # 残差耦合：在 logit 空间加性调制
        # α 使用 softplus 确保非负（调制强度 ≥ 0）
        alpha = torch.nn.functional.softplus(self.alpha)
        gated_scores = attn_scores + alpha * torch.cos(delta_theta)

        return gated_scores

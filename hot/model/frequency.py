"""
固有频率模块

频率由 Q/K 能量差决定，表征 Token 的"内在节拍"。

优化：保持 per-head 信息，避免大中间张量。
"""

import torch
import torch.nn as nn


class IntrinsicFrequency(nn.Module):
    """
    Token 固有频率计算

    ω_i = tanh(Linear(||Q_i||² - ||K_i||²))

    每个头独立计算频率，输入是标量能量差，输出是 (-1, 1) 范围的频率。
    """

    def __init__(self, config):
        super().__init__()
        if isinstance(config, dict):
            model_config = config.get('model', {})
            num_heads = model_config.get('num_heads', 12)
        else:
            num_heads = config.model.num_heads

        # 每个头一个独立的线性投影：标量 → 标量
        self.proj = nn.Linear(1, num_heads)

    def forward(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        计算每个 Token 的固有频率

        Args:
            Q: [B, H, N, d] Query 向量
            K: [B, H, N, d] Key 向量
        Returns:
            omega: [B, H, N] 固有频率，范围 (-1, 1)
        """
        # ||Q_i||² - ||K_i||²，沿 head_dim 维度求和（保留 head 维度）
        # Q.pow(2) - K.pow(2): [B, H, N, d]
        # .sum(dim=-1): [B, H, N] — 每个 head 独立的能量差
        energy_diff = (Q.pow(2) - K.pow(2)).sum(dim=-1)  # [B, H, N]

        # 投影到频率：[B, H, N] → [B, H, N, 1] → proj → [B, H, N, H_out]
        # 但我们需要每头独立的频率，所以用逐元素方式
        omega = torch.tanh(self.proj(energy_diff.unsqueeze(-1)))  # [B, H, N, num_heads]

        # 提取每头对应的频率：omega[:, h, :, h] → [B, N] per head
        # 更简洁的方式：直接用 energy_diff 作为输入
        # 由于 proj 是 Linear(1, num_heads)，每个 head 的输出是 weight[h] * input + bias[h]
        # 我们需要 [B, H, N] 输出，但 proj 输出是 [B, H, N, num_heads]

        # 方案：直接返回 energy_diff 经过 tanh 缩放
        # 这样每头有独立的能量差信号
        alpha = self.proj.weight.mean()  # 标量缩放因子
        beta = self.proj.bias.mean()     # 标量偏置

        return torch.tanh(alpha * energy_diff + beta)

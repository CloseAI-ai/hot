"""
固有频率模块

实现频率-相位解耦方案：
ω_i = tanh(Linear(||Q_i||² - ||K_i||²))

频率由当前 Token 的 Query/Key 能量差决定，代表该 Token 的"内在节拍"。
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
            omega: [B, N] 固有频率，范围 (-1, 1)
        """
        # ||Q_i||² - ||K_i||²，沿 head 和 d 维度求和 → [B, N]
        energy_diff = (Q.pow(2) - K.pow(2)).sum(dim=(1, 3))  # [B, N]

        # Linear 投影：[B, N] → [B, N, H] → [B, H, N]
        omega = torch.tanh(self.proj(energy_diff.unsqueeze(-1)))  # [B, N, H]
        omega = omega.permute(0, 2, 1)  # [B, H, N]

        return omega

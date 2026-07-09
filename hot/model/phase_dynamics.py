"""
相位动力学模块

频率-相位解耦方案：
- 频率 ω：由 Q/K 能量差决定（在 frequency.py 中计算）
- 相位 θ：跨层传递的可训练隐状态，每层更新一次：
    θ_i^(l+1) = θ_i^(l) + ω_i · Δt

彻底消除了对上层注意力权重的依赖（原 Kuramoto 耦合方案的鸡生蛋问题）。
"""

import math
import torch
import torch.nn as nn


class PhaseDynamics(nn.Module):
    """
    相位动力学：简单的 Euler 更新

    θ_i^(l+1) = θ_i^(l) + ω_i · Δt

    其中 Δt 是可学习的层间时间步长。
    """

    def __init__(self, config):
        super().__init__()
        # 可学习的时间步长 Δt，初始化为 1.0
        self.dt = nn.Parameter(torch.tensor(1.0))

    def forward(self, theta: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        """
        相位更新（无耦合，纯频率驱动）

        Args:
            theta: [B, H, N] 当前相位
            omega: [B, H, N] 固有频率
        Returns:
            theta_new: [B, H, N] 更新后的相位，归一化到 [0, 2π)
        """
        theta_new = theta + omega * self.dt
        theta_new = theta_new % (2 * math.pi)
        return theta_new

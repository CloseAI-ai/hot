"""
退火调度模块

实现 PPA 策略的三种退火调度
"""

import math
import torch


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
        if self.schedule == 'none':
            return 1.0

        if step < self.K:
            return 0.0  # 冻结期

        t = step - self.K

        if self.schedule == 'linear':
            return min(1.0, t / self.K)
        elif self.schedule == 'cosine':
            # 余弦退火：在 [0, K] 内完成半周期
            tc = min(t, self.K)
            return 0.5 * (1 - math.cos(math.pi * tc / self.K))
        elif self.schedule == 'sigmoid':
            return torch.sigmoid(torch.tensor((t - 1.5 * self.K) / (self.K / 5))).item()
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

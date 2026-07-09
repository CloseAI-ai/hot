"""
学习率调度模块
"""

import math
from torch.optim.lr_scheduler import LambdaLR
from typing import Dict, Any


def get_scheduler(optimizer, config: Dict[str, Any]) -> LambdaLR:
    """
    获取学习率调度器

    Args:
        optimizer: 优化器
        config: 训练配置
    Returns:
        scheduler: 学习率调度器
    """
    schedule_type = config.get('lr_schedule', 'cosine')
    warmup_steps = config.get('warmup_steps', 2000)
    max_steps = config.get('max_steps', 100000)

    if schedule_type == 'cosine':
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))

            progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    elif schedule_type == 'linear':
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))

            progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
            return max(0.0, 1.0 - progress)

    elif schedule_type == 'constant':
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 1.0

    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")

    return LambdaLR(optimizer, lr_lambda)

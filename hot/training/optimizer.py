"""
优化器配置模块
"""

import torch
from torch.optim import AdamW
from typing import List, Dict, Any


def configure_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> AdamW:
    """
    配置优化器

    Args:
        model: 模型
        config: 训练配置
    Returns:
        optimizer: AdamW 优化器
    """
    # 分离需要权重衰减和不需要权重衰减的参数
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'bias' in name or 'norm' in name or 'embedding' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {'params': decay_params, 'weight_decay': config.get('weight_decay', 0.1)},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]

    optimizer = AdamW(
        param_groups,
        lr=config.get('learning_rate', 3e-4),
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    return optimizer

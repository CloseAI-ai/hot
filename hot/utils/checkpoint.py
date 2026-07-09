"""
检查点管理模块
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Optional


def save_checkpoint(model: nn.Module, optimizer, scheduler, step: int,
                   loss: float, path: str):
    """
    保存检查点

    Args:
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        step: 当前步数
        loss: 当前损失
        path: 保存路径
    """
    # 创建目录
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # 保存检查点
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'step': step,
        'loss': loss,
    }

    torch.save(checkpoint, path)


def load_checkpoint(model: nn.Module, optimizer=None, scheduler=None,
                   path: str = "checkpoint.pt") -> Dict:
    """
    加载检查点

    Args:
        model: 模型
        optimizer: 优化器（可选）
        scheduler: 学习率调度器（可选）
        path: 检查点路径
    Returns:
        checkpoint: 检查点字典
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # 加载检查点
    checkpoint = torch.load(path, map_location='cpu')

    # 加载模型参数
    model.load_state_dict(checkpoint['model_state_dict'])

    # 加载优化器参数（可选）
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 加载调度器参数（可选）
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint

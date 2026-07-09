"""
困惑度计算模块
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict


def compute_perplexity(model: nn.Module, dataloader: DataLoader,
                       device: torch.device = None) -> Dict[str, float]:
    """
    计算模型困惑度

    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
    Returns:
        results: 包含困惑度和损失的字典
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            # 将数据移到设备
            batch = {k: v.to(device) for k, v in batch.items()}

            # 前向传播
            outputs = model(**batch)
            loss = outputs['loss']

            # 统计
            batch_size, seq_length = batch['input_ids'].shape
            num_tokens = (batch['labels'] != -100).sum().item()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    # 计算平均损失和困惑度
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'num_tokens': total_tokens,
    }

"""
频谱可视化模块
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict


def compute_frequency_spectrum(model: nn.Module, dataloader: DataLoader,
                               device: torch.device = None) -> Dict[str, torch.Tensor]:
    """
    分析各头的频率分布

    Args:
        model: HOT 模型
        dataloader: 数据加载器
        device: 设备
    Returns:
        results: 包含频率分布的字典
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    all_frequencies = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)

            # 前向传播
            outputs = model(input_ids=input_ids)

            # 获取频率（需要从模型内部获取）
            # 这里假设模型返回了频率信息
            # 实际实现需要修改模型以返回频率

            # 临时实现：使用随机数据
            batch_size, seq_length = input_ids.shape
            num_layers = len(model.layers)
            num_heads = model.config.num_heads

            # 模拟频率数据
            frequencies = torch.randn(batch_size, num_layers, num_heads, seq_length)
            all_frequencies.append(frequencies)

    # 堆叠所有 batch 的结果
    all_frequencies = torch.cat(all_frequencies, dim=0)  # [total_samples, num_layers, num_heads, seq_length]

    # 计算统计信息
    freq_mean = all_frequencies.mean(dim=(0, 3))  # [num_layers, num_heads]
    freq_std = all_frequencies.std(dim=(0, 3))  # [num_layers, num_heads]

    # 计算频率分布直方图
    freq_distribution = all_frequencies.mean(dim=3)  # [total_samples, num_layers, num_heads]

    return {
        'freq_mean': freq_mean,
        'freq_std': freq_std,
        'freq_distribution': freq_distribution,
    }

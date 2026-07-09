"""
序参量分析模块
"""

import torch
from typing import Dict


def compute_order_parameter(thetas: torch.Tensor) -> torch.Tensor:
    """
    计算序参量 r = |1/N Σ_j exp(iθ_j)|

    Args:
        thetas: [B, H, N] 相位
    Returns:
        r: [B, H] 序参量，范围 [0, 1]
    """
    complex_phases = torch.exp(1j * thetas)
    r = torch.abs(complex_phases.mean(dim=-1))
    return r


def analyze_order_parameter_evolution(model, dataloader, device=None) -> Dict[str, torch.Tensor]:
    """
    分析序参量随训练步数的变化

    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
    Returns:
        results: 包含序参量演化的字典
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    all_order_params = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)

            # 前向传播
            outputs = model(input_ids=input_ids)

            # 获取相位
            thetas = outputs['thetas']  # List of [B, H, N]

            # 计算每层的序参量
            layer_order_params = []
            for theta in thetas:
                r = compute_order_parameter(theta)
                layer_order_params.append(r.mean(dim=(0, 1)))  # 平均 batch 和 head

            all_order_params.append(torch.stack(layer_order_params))

    # 堆叠所有 batch 的结果
    all_order_params = torch.stack(all_order_params)  # [num_batches, num_layers, N]

    return {
        'order_params': all_order_params,
        'mean': all_order_params.mean(dim=0),
        'std': all_order_params.std(dim=0),
    }

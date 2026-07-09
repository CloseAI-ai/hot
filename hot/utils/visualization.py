"""
可视化工具模块
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional, List


def plot_attention(attention_weights: torch.Tensor, layer_idx: int = 0,
                  head_idx: int = 0, save_path: Optional[str] = None):
    """
    绘制注意力权重热力图

    Args:
        attention_weights: [B, H, N, N] 注意力权重
        layer_idx: 层索引
        head_idx: 头索引
        save_path: 保存路径（可选）
    """
    # 获取指定层和头的注意力权重
    attn = attention_weights[0, head_idx].cpu().numpy()

    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn, cmap='viridis', xticklabels=False, yticklabels=False)
    plt.title(f'Attention Weights (Layer {layer_idx}, Head {head_idx})')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_phase_evolution(thetas: List[torch.Tensor], save_path: Optional[str] = None):
    """
    绘制相位演化图

    Args:
        thetas: 各层的相位 [List of [B, H, N]]
        save_path: 保存路径（可选）
    """
    fig, axes = plt.subplots(len(thetas), 1, figsize=(12, 3 * len(thetas)))

    for i, theta in enumerate(thetas):
        # 获取第一个 batch 和第一个 head
        theta_np = theta[0, 0].cpu().numpy()

        ax = axes[i] if len(thetas) > 1 else axes
        ax.scatter(range(len(theta_np)), theta_np, alpha=0.6, s=10)
        ax.set_title(f'Layer {i} Phase Distribution')
        ax.set_xlabel('Token Position')
        ax.set_ylabel('Phase (radians)')
        ax.set_ylim(0, 2 * np.pi)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_frequency_distribution(frequencies: torch.Tensor, layer_idx: int = 0,
                               save_path: Optional[str] = None):
    """
    绘制频率分布图

    Args:
        frequencies: [B, H, N] 频率
        layer_idx: 层索引
        save_path: 保存路径（可选）
    """
    # 获取指定层的频率
    freq = frequencies[:, layer_idx].cpu().numpy()

    # 绘制直方图
    plt.figure(figsize=(10, 6))
    for head_idx in range(freq.shape[1]):
        plt.hist(freq[:, head_idx].flatten(), bins=50, alpha=0.5,
                label=f'Head {head_idx}')

    plt.title(f'Frequency Distribution (Layer {layer_idx})')
    plt.xlabel('Frequency')
    plt.ylabel('Count')
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()

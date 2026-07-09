#!/usr/bin/env python3
"""
HOT 可视化脚本

用法：
    python scripts/visualize.py --config configs/hot_42m.yaml --checkpoint checkpoints/best_model.pt
"""

import argparse
import logging
import os
import yaml
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.utils.data import DataLoader

from hot.model import HOTModel
from hot.data import HOTDataset, get_tokenizer, DataCollator
from hot.evaluation import compute_order_parameter, compute_frequency_spectrum
from hot.utils import load_checkpoint, plot_attention, plot_phase_evolution


def load_config(config_path: str) -> dict:
    """加载并合并配置"""
    with open('configs/base.yaml', 'r') as f:
        config = yaml.safe_load(f)
    with open(config_path, 'r') as f:
        model_config = yaml.safe_load(f)

    def merge(base, override):
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                merge(base[k], v)
            else:
                base[k] = v
        return base

    return merge(config, model_config)


def main():
    parser = argparse.ArgumentParser(description='HOT Visualization')
    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='检查点路径')
    parser.add_argument('--device', type=str, default=None,
                        help='设备')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                        help='输出目录')
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device:
        config['device'] = args.device

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    device_str = config.get('device', 'auto')
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)

    os.makedirs(args.output_dir, exist_ok=True)

    # 分词器和数据
    tokenizer = get_tokenizer()
    data_cfg = config.get('data', {})

    dataset = HOTDataset(
        dataset_name=data_cfg.get('dataset', 'the_pile'),
        tokenizer=tokenizer,
        max_length=data_cfg.get('max_length', 2048),
        split='validation',
        num_samples=100,
    )

    collator = DataCollator(pad_token_id=tokenizer.pad_token_id)
    dataloader = DataLoader(
        dataset, batch_size=4, shuffle=False,
        collate_fn=collator, num_workers=0,
    )

    # 模型
    model = HOTModel(config)
    checkpoint = load_checkpoint(model, path=args.checkpoint)
    logging.info(f"已加载检查点: step={checkpoint.get('global_step', 'unknown')}")
    model.to(device)
    model.eval()

    # 获取一个 batch
    batch = next(iter(dataloader))
    input_ids = batch['input_ids'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    # 1. 注意力权重
    logging.info("绘制注意力权重...")
    for layer_idx in range(min(3, len(model.layers))):
        plot_attention(
            outputs['attentions'][layer_idx],
            layer_idx=layer_idx,
            save_path=os.path.join(args.output_dir, f'attention_layer_{layer_idx}.png'),
        )

    # 2. 相位演化
    logging.info("绘制相位演化...")
    plot_phase_evolution(
        outputs['thetas'],
        save_path=os.path.join(args.output_dir, 'phase_evolution.png'),
    )

    # 3. 频率分布热力图
    logging.info("计算频率分布...")
    freq_results = compute_frequency_spectrum(model, dataloader, device)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        freq_results['freq_mean'].cpu().numpy(),
        annot=True, fmt='.3f',
        xticklabels=[f'H{i}' for i in range(freq_results['freq_mean'].shape[1])],
        yticklabels=[f'L{i}' for i in range(freq_results['freq_mean'].shape[0])],
    )
    plt.title('Mean Intrinsic Frequency per Head per Layer')
    plt.xlabel('Head')
    plt.ylabel('Layer')
    plt.savefig(os.path.join(args.output_dir, 'frequency_heatmap.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # 4. 序参量
    logging.info("计算序参量...")
    order_params = []
    for theta in outputs['thetas']:
        r = compute_order_parameter(theta)
        order_params.append(r.mean().item())

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(order_params)), order_params, 'o-', linewidth=2, markersize=8)
    plt.title('Order Parameter by Layer')
    plt.xlabel('Layer')
    plt.ylabel('Order Parameter r')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.output_dir, 'order_parameter.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"可视化已保存到: {args.output_dir}")
    logging.info("完成!")


if __name__ == '__main__':
    main()

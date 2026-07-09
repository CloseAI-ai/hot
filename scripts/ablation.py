#!/usr/bin/env python3
"""
HOT 消融实验脚本

用法：
    python scripts/ablation.py --experiments no_gating full_coupling no_annealing
    python scripts/ablation.py --experiments no_gating --max_steps 1000
"""

import argparse
import json
import logging
import yaml
import torch
from torch.utils.data import DataLoader

from hot.model import HOTModel
from hot.training import Trainer
from hot.data import HOTDataset, get_tokenizer, DataCollator
from hot.evaluation import compute_perplexity


EXPERIMENT_CONFIGS = {
    'no_gating': 'configs/ablation/no_gating.yaml',
    'full_coupling': 'configs/ablation/full_coupling.yaml',
    'no_annealing': 'configs/ablation/no_annealing.yaml',
}


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


def run_experiment(config: dict, experiment_name: str,
                   max_steps: int, device: torch.device) -> dict:
    """运行单个消融实验"""
    logging.info(f"\n{'='*60}")
    logging.info(f"实验: {experiment_name}")
    logging.info(f"{'='*60}")

    # 覆盖训练步数
    config['training']['max_steps'] = max_steps

    # 设置随机种子
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 分词器和数据
    tokenizer = get_tokenizer()
    data_cfg = config.get('data', {})

    train_dataset = HOTDataset(
        dataset_name=data_cfg.get('dataset', 'the_pile'),
        tokenizer=tokenizer,
        max_length=data_cfg.get('max_length', 2048),
        split='train',
        num_samples=5000,
    )

    collator = DataCollator(pad_token_id=tokenizer.pad_token_id)
    train_cfg = config.get('training', {})

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get('batch_size', 16),
        shuffle=True,
        collate_fn=collator,
        num_workers=data_cfg.get('num_workers', 4),
        pin_memory=True,
        drop_last=True,
    )

    # 模型
    model = HOTModel(config)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"参数量: {total_params / 1e6:.2f}M")

    # 训练
    trainer = Trainer(model, config)
    trainer.train(train_dataloader)

    # 评估
    eval_dataset = HOTDataset(
        dataset_name=data_cfg.get('dataset', 'the_pile'),
        tokenizer=tokenizer,
        max_length=data_cfg.get('max_length', 2048),
        split='validation',
        num_samples=500,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=train_cfg.get('batch_size', 16),
        shuffle=False,
        collate_fn=collator,
        num_workers=data_cfg.get('num_workers', 4),
    )

    results = compute_perplexity(model, eval_dataloader, device)
    logging.info(f"{experiment_name} 结果: Loss={results['loss']:.4f}, "
                 f"Perplexity={results['perplexity']:.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='HOT Ablation Experiments')
    parser.add_argument('--experiments', nargs='+',
                        default=list(EXPERIMENT_CONFIGS.keys()),
                        help='要运行的实验')
    parser.add_argument('--device', type=str, default=None,
                        help='设备')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='每个实验的最大训练步数')
    parser.add_argument('--output', type=str, default='ablation_results.json',
                        help='结果输出文件')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"设备: {device}")

    # 运行实验
    all_results = {}

    for exp_name in args.experiments:
        if exp_name not in EXPERIMENT_CONFIGS:
            logging.warning(f"未知实验: {exp_name}，跳过")
            continue

        config = load_config(EXPERIMENT_CONFIGS[exp_name])
        results = run_experiment(config, exp_name, args.max_steps, device)
        all_results[exp_name] = results

    # 保存结果
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    logging.info(f"结果已保存到: {args.output}")

    # 打印总结
    print(f"\n{'='*60}")
    print("消融实验总结:")
    print(f"{'='*60}")
    for exp_name, results in all_results.items():
        print(f"  {exp_name:<20s}  Loss={results['loss']:.4f}  "
              f"Perplexity={results['perplexity']:.2f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

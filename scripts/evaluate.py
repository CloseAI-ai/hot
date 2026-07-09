#!/usr/bin/env python3
"""
HOT 评估脚本

用法：
    python scripts/evaluate.py --config configs/hot_42m.yaml --checkpoint checkpoints/best_model.pt
    python scripts/evaluate.py --config configs/hot_42m.yaml --checkpoint checkpoints/best_model.pt --eval_type extrapolation
"""

import argparse
import logging
import yaml
import torch
from torch.utils.data import DataLoader

from hot.model import HOTModel
from hot.data import HOTDataset, get_tokenizer, DataCollator
from hot.evaluation import compute_perplexity, length_extrapolation_test
from hot.utils import load_checkpoint


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
    parser = argparse.ArgumentParser(description='HOT Evaluation')
    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='检查点路径')
    parser.add_argument('--device', type=str, default=None,
                        help='设备')
    parser.add_argument('--eval_type', type=str, default='perplexity',
                        choices=['perplexity', 'extrapolation'],
                        help='评估类型')
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    if args.device:
        config['device'] = args.device

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 设备
    device_str = config.get('device', 'auto')
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    logging.info(f"设备: {device}")

    # 分词器
    tokenizer = get_tokenizer()

    # 模型
    model = HOTModel(config)
    checkpoint = load_checkpoint(model, path=args.checkpoint)
    logging.info(f"已加载检查点: step={checkpoint.get('global_step', 'unknown')}, "
                 f"loss={checkpoint.get('best_loss', 'unknown')}")
    model.to(device)

    if args.eval_type == 'perplexity':
        # 验证数据集
        data_cfg = config.get('data', {})
        eval_dataset = HOTDataset(
            dataset_name=data_cfg.get('dataset', 'the_pile'),
            tokenizer=tokenizer,
            max_length=data_cfg.get('max_length', 2048),
            split='validation',
        )

        collator = DataCollator(pad_token_id=tokenizer.pad_token_id)
        train_cfg = config.get('training', {})

        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=train_cfg.get('batch_size', 16),
            shuffle=False,
            collate_fn=collator,
            num_workers=data_cfg.get('num_workers', 4),
        )

        # 计算困惑度
        logging.info("计算困惑度...")
        results = compute_perplexity(model, eval_dataloader, device)

        print(f"\n{'='*50}")
        print(f"评估结果:")
        print(f"  Loss:      {results['loss']:.4f}")
        print(f"  Perplexity: {results['perplexity']:.2f}")
        print(f"  Tokens:    {results['num_tokens']}")
        print(f"{'='*50}")

    elif args.eval_type == 'extrapolation':
        test_lengths = [8192, 16384, 32768]
        logging.info(f"长度外推测试: {test_lengths}")

        results = length_extrapolation_test(
            model=model,
            tokenizer=tokenizer,
            train_len=config.get('data', {}).get('max_length', 2048),
            test_lengths=test_lengths,
            device=device,
        )

        print(f"\n{'='*50}")
        print("长度外推结果:")
        for length, ppl in results.items():
            print(f"  Length {length:>5d}: Perplexity = {ppl:.2f}")
        print(f"{'='*50}")

    logging.info("评估完成!")


if __name__ == '__main__':
    main()

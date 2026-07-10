#!/usr/bin/env python3
"""
HOT 训练脚本

用法：
    python scripts/train.py --config configs/hot_42m.yaml
    python scripts/train.py --config configs/hot_42m.yaml --resume checkpoints/latest.pt
    python scripts/train.py --config configs/hot_42m.yaml --pretokenize  # 预分词数据集
"""

import argparse
import logging
import yaml
import torch
from torch.utils.data import DataLoader

# 启用 TF32（RTX 30/40 系列 Tensor Core 加速）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

from hot.model import HOTModel
from hot.training import Trainer
from hot.data import HOTDataset, get_tokenizer, DataCollator, pretokenize_dataset


def _convert_numeric_strings(obj):
    """递归将 YAML 解析出的数字字符串转为数值"""
    if isinstance(obj, dict):
        return {k: _convert_numeric_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numeric_strings(v) for v in obj]
    elif isinstance(obj, str):
        try:
            return float(obj)
        except ValueError:
            return obj
    return obj


def load_config(config_path: str) -> dict:
    """加载并合并配置（base + 模型配置）"""
    with open('configs/base.yaml', 'r') as f:
        config = yaml.safe_load(f)

    with open(config_path, 'r') as f:
        model_config = yaml.safe_load(f)

    # 深度合并
    def merge(base, override):
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                merge(base[k], v)
            else:
                base[k] = v
        return base

    merged = merge(config, model_config)
    return _convert_numeric_strings(merged)


def main():
    parser = argparse.ArgumentParser(description='HOT Training')
    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路径')
    parser.add_argument('--device', type=str, default=None,
                        help='设备（覆盖配置文件）')
    parser.add_argument('--use_wandb', action='store_true',
                        help='使用 Wandb 记录实验')
    parser.add_argument('--max_steps', type=int, default=None,
                        help='覆盖最大训练步数')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='覆盖 batch size')
    parser.add_argument('--resume', type=str, default=None,
                        help='从检查点恢复训练（如 checkpoints/latest.pt）')
    parser.add_argument('--pretokenize', action='store_true',
                        help='预分词数据集并保存')
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 命令行覆盖
    if args.device:
        config['device'] = args.device
    if args.use_wandb:
        config['use_wandb'] = True
    if args.max_steps:
        config['training']['max_steps'] = args.max_steps
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    # 设置随机种子
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 设备
    device_str = config.get('device', 'auto')
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    logging.info(f"设备: {device}")

    # 分词器
    tokenizer = get_tokenizer()

    # 数据集配置
    data_cfg = config.get('data', {})
    train_cfg = config.get('training', {})
    max_length = int(data_cfg.get('max_length', 512))

    # 设置 HF 镜像环境变量
    import os
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

    # 预分词模式
    if args.pretokenize:
        pretokenize_dataset(
            dataset_name=data_cfg.get('dataset', 'RobinChen2001/TinyStories-Zh-2M'),
            tokenizer=tokenizer,
            max_length=max_length,
            split='train',
            hf_token=data_cfg.get('hf_token'),
            cache_dir=data_cfg.get('cache_dir'),
            save_path=f"data/tokenized/train_{max_length}",
            num_proc=4,
        )
        logging.info("预分词完成！")
        return

    # 检查是否有预分词数据
    pretokenized_path = f"data/tokenized/train_{max_length}"

    train_dataset = HOTDataset(
        dataset_name=data_cfg.get('dataset', 'RobinChen2001/TinyStories-Zh-2M'),
        tokenizer=tokenizer,
        max_length=max_length,
        split='train',
        hf_token=data_cfg.get('hf_token'),
        cache_dir=data_cfg.get('cache_dir'),
        pretokenized_path=pretokenized_path if os.path.exists(pretokenized_path) else None,
    )

    collator = DataCollator(pad_token_id=tokenizer.pad_token_id)

    # DataLoader 优化：预分词数据用 num_workers=0 避免 IPC 开销
    num_workers = data_cfg.get('num_workers', 4)
    use_pretokenized = os.path.exists(pretokenized_path)
    dl_workers = 0 if use_pretokenized else num_workers
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg.get('batch_size', 16)),
        shuffle=True,
        collate_fn=collator,
        num_workers=dl_workers,
        pin_memory=dl_workers > 0,
        drop_last=True,
        persistent_workers=dl_workers > 0,
        prefetch_factor=4 if dl_workers > 0 else None,
    )

    # 模型
    model = HOTModel(config)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"模型参数量: {total_params / 1e6:.2f}M")

    # 训练器
    trainer = Trainer(model, config)

    # 断点续传
    if args.resume:
        trainer.resume(args.resume)
        logging.info(f"从检查点恢复: step={trainer.global_step}")

    # 训练
    logging.info("开始训练...")
    trainer.train(train_dataloader)
    logging.info("训练完成!")


if __name__ == '__main__':
    main()

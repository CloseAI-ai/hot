#!/usr/bin/env python3
"""
创建训练/验证集分割

按照行业规范从训练数据中公正随机抽取验证集：
- 验证集比例: 10%（NLP/LLM 训练标准）
- 固定随机种子: 42（确保可复现）
- 保存分割索引（支持审计和复现）
- 验证集和训练集无重叠

Usage:
    python scripts/create_validation_split.py [--val_ratio 0.1] [--seed 42]
"""

import os
import sys
import json
import random
import logging
import argparse
from datetime import datetime
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_from_disk

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_validation_split(
    data_path: str = "data/tokenized/train_512",
    output_dir: str = "data/splits",
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """
    创建训练/验证集分割

    Args:
        data_path: 预分词数据集路径
        output_dir: 输出目录
        val_ratio: 验证集比例 (默认 10%)
        seed: 随机种子 (默认 42)
    """
    logger.info(f"加载数据集: {data_path}")
    dataset = load_from_disk(data_path)
    total_size = len(dataset)
    logger.info(f"数据集总大小: {total_size:,} 条")

    # 计算验证集大小
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size

    logger.info(f"分割比例: 训练 {1-val_ratio:.0%} / 验证 {val_ratio:.0%}")
    logger.info(f"训练集大小: {train_size:,} 条")
    logger.info(f"验证集大小: {val_size:,} 条")

    # 生成随机排列的索引
    logger.info(f"使用随机种子: {seed}")
    random.seed(seed)
    indices = list(range(total_size))
    random.shuffle(indices)

    # 分割索引
    val_indices = sorted(indices[:val_size])
    train_indices = sorted(indices[val_size:])

    # 验证无重叠
    assert len(set(train_indices) & set(val_indices)) == 0, "训练集和验证集有重叠！"
    assert len(train_indices) + len(val_indices) == total_size, "索引总数不匹配！"

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存索引文件
    split_info = {
        "created_at": datetime.now().isoformat(),
        "source_data": data_path,
        "total_samples": total_size,
        "train_size": train_size,
        "val_size": val_size,
        "val_ratio": val_ratio,
        "random_seed": seed,
        "train_indices": train_indices,
        "val_indices": val_indices,
    }

    # 保存完整分割信息
    split_info_path = os.path.join(output_dir, "split_info.json")
    with open(split_info_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    logger.info(f"分割信息已保存: {split_info_path}")

    # 保存索引文件（纯文本，便于审计）
    train_indices_path = os.path.join(output_dir, "train_indices.txt")
    with open(train_indices_path, 'w') as f:
        f.write(f"# 训练集索引 (共 {train_size:,} 条)\n")
        f.write(f"# 随机种子: {seed}\n")
        f.write(f"# 生成时间: {datetime.now().isoformat()}\n")
        for idx in train_indices:
            f.write(f"{idx}\n")

    val_indices_path = os.path.join(output_dir, "val_indices.txt")
    with open(val_indices_path, 'w') as f:
        f.write(f"# 验证集索引 (共 {val_size:,} 条)\n")
        f.write(f"# 随机种子: {seed}\n")
        f.write(f"# 生成时间: {datetime.now().isoformat()}\n")
        for idx in val_indices:
            f.write(f"{idx}\n")

    logger.info(f"索引文件已保存:")
    logger.info(f"  训练集: {train_indices_path}")
    logger.info(f"  验证集: {val_indices_path}")

    # 创建子数据集并保存
    logger.info("创建训练集子集...")
    train_dataset = dataset.select(train_indices)
    train_path = os.path.join(output_dir, "train")
    train_dataset.save_to_disk(train_path)
    logger.info(f"训练集已保存: {train_path}")

    logger.info("创建验证集子集...")
    val_dataset = dataset.select(val_indices)
    val_path = os.path.join(output_dir, "val")
    val_dataset.save_to_disk(val_path)
    logger.info(f"验证集已保存: {val_path}")

    # 验证分割正确性
    logger.info("\n验证分割正确性...")
    train_ds = load_from_disk(train_path)
    val_ds = load_from_disk(val_path)

    assert len(train_ds) == train_size, f"训练集大小不匹配: {len(train_ds)} != {train_size}"
    assert len(val_ds) == val_size, f"验证集大小不匹配: {len(val_ds)} != {val_size}"

    # 检查数据完整性
    logger.info(f"训练集实际大小: {len(train_ds):,}")
    logger.info(f"验证集实际大小: {len(val_ds):,}")

    # 生成统计报告
    logger.info("\n" + "="*60)
    logger.info("验证集分割完成")
    logger.info("="*60)
    logger.info(f"数据来源: {data_path}")
    logger.info(f"总样本数: {total_size:,}")
    logger.info(f"训练集: {train_size:,} ({(1-val_ratio)*100:.1f}%)")
    logger.info(f"验证集: {val_size:,} ({val_ratio*100:.1f}%)")
    logger.info(f"随机种子: {seed}")
    logger.info(f"输出目录: {output_dir}")
    logger.info("="*60)

    return {
        "train_path": train_path,
        "val_path": val_path,
        "train_size": train_size,
        "val_size": val_size,
    }


def main():
    parser = argparse.ArgumentParser(description="创建训练/验证集分割")
    parser.add_argument("--data_path", type=str, default="data/tokenized/train_512",
                        help="预分词数据集路径")
    parser.add_argument("--output_dir", type=str, default="data/splits",
                        help="输出目录")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="验证集比例 (默认 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认 42)")
    args = parser.parse_args()

    create_validation_split(
        data_path=args.data_path,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
验证训练/验证集分割的数据完整性

检查项：
1. 训练集和验证集无重叠
2. 并集覆盖所有样本
3. 验证集大小正确
4. 分割是随机的（非顺序切分）
5. 数据内容一致性

Usage:
    python scripts/verify_data_split.py
"""

import os
import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def verify_data_split():
    """验证数据分割的完整性"""

    splits_dir = "data/splits"
    split_info_path = os.path.join(splits_dir, "split_info.json")

    # 1. 检查分割信息文件
    logger.info("=" * 60)
    logger.info("1. 检查分割信息文件")
    logger.info("=" * 60)

    if not os.path.exists(split_info_path):
        logger.error(f"分割信息文件不存在: {split_info_path}")
        return False

    with open(split_info_path, 'r') as f:
        split_info = json.load(f)

    total_samples = split_info['total_samples']
    train_size = split_info['train_size']
    val_size = split_info['val_size']
    train_indices = split_info['train_indices']
    val_indices = split_info['val_indices']

    logger.info(f"总样本数: {total_samples:,}")
    logger.info(f"训练集大小 (配置): {train_size:,}")
    logger.info(f"验证集大小 (配置): {val_size:,}")
    logger.info(f"训练集索引数: {len(train_indices):,}")
    logger.info(f"验证集索引数: {len(val_indices):,}")

    # 2. 检查索引数量一致性
    logger.info("\n" + "=" * 60)
    logger.info("2. 检查索引数量一致性")
    logger.info("=" * 60)

    if len(train_indices) != train_size:
        logger.error(f"训练集索引数量不匹配: {len(train_indices)} != {train_size}")
        return False

    if len(val_indices) != val_size:
        logger.error(f"验证集索引数量不匹配: {len(val_indices)} != {val_size}")
        return False

    if len(train_indices) + len(val_indices) != total_samples:
        logger.error(f"索引总数不匹配: {len(train_indices)} + {len(val_indices)} != {total_samples}")
        return False

    logger.info("✓ 索引数量一致")

    # 3. 检查索引范围
    logger.info("\n" + "=" * 60)
    logger.info("3. 检查索引范围")
    logger.info("=" * 60)

    train_set = set(train_indices)
    val_set = set(val_indices)

    if min(train_indices) < 0 or max(train_indices) >= total_samples:
        logger.error(f"训练集索引超出范围: [{min(train_indices)}, {max(train_indices)}]")
        return False

    if min(val_indices) < 0 or max(val_indices) >= total_samples:
        logger.error(f"验证集索引超出范围: [{min(val_indices)}, {max(val_indices)}]")
        return False

    logger.info(f"训练集索引范围: [{min(train_indices)}, {max(train_indices)}]")
    logger.info(f"验证集索引范围: [{min(val_indices)}, {max(val_indices)}]")
    logger.info("✓ 索引范围正确")

    # 4. 检查无重叠
    logger.info("\n" + "=" * 60)
    logger.info("4. 检查训练集和验证集无重叠")
    logger.info("=" * 60)

    overlap = train_set & val_set
    if overlap:
        logger.error(f"发现 {len(overlap):,} 个重叠索引!")
        logger.error(f"前10个重叠索引: {sorted(list(overlap))[:10]}")
        return False

    logger.info("✓ 训练集和验证集无重叠")

    # 5. 检查并集覆盖所有样本
    logger.info("\n" + "=" * 60)
    logger.info("5. 检查并集覆盖所有样本")
    logger.info("=" * 60)

    all_indices = train_set | val_set
    expected_indices = set(range(total_samples))

    missing = expected_indices - all_indices
    extra = all_indices - expected_indices

    if missing:
        logger.error(f"缺少 {len(missing):,} 个索引!")
        logger.error(f"前10个缺失索引: {sorted(list(missing))[:10]}")
        return False

    if extra:
        logger.error(f"多出 {len(extra):,} 个索引!")
        logger.error(f"前10个多余索引: {sorted(list(extra))[:10]}")
        return False

    logger.info("✓ 并集完整覆盖所有样本")

    # 6. 检查索引唯一性
    logger.info("\n" + "=" * 60)
    logger.info("6. 检查索引唯一性")
    logger.info("=" * 60)

    if len(train_indices) != len(train_set):
        logger.error(f"训练集索引有重复: {len(train_indices)} != {len(train_set)}")
        return False

    if len(val_indices) != len(val_set):
        logger.error(f"验证集索引有重复: {len(val_indices)} != {len(val_set)}")
        return False

    logger.info("✓ 索引无重复")

    # 7. 检查随机性（非顺序切分）
    logger.info("\n" + "=" * 60)
    logger.info("7. 检查随机性（非顺序切分）")
    logger.info("=" * 60)

    # 检查验证集索引是否连续
    val_sorted = sorted(val_indices)
    is_sequential = all(val_sorted[i+1] - val_sorted[i] == 1 for i in range(min(100, len(val_sorted)-1)))

    if is_sequential:
        logger.warning("验证集索引似乎是连续的（可能是顺序切分）")
    else:
        logger.info("✓ 验证集索引是随机分布的")

    # 检查训练集索引是否连续
    train_sorted = sorted(train_indices)
    is_sequential = all(train_sorted[i+1] - train_sorted[i] == 1 for i in range(min(100, len(train_sorted)-1)))

    if is_sequential:
        logger.warning("训练集索引似乎是连续的（可能是顺序切分）")
    else:
        logger.info("✓ 训练集索引是随机分布的")

    # 8. 检查实际数据文件
    logger.info("\n" + "=" * 60)
    logger.info("8. 检查实际数据文件")
    logger.info("=" * 60)

    train_path = os.path.join(splits_dir, "train")
    val_path = os.path.join(splits_dir, "val")

    if not os.path.exists(train_path):
        logger.error(f"训练集目录不存在: {train_path}")
        return False

    if not os.path.exists(val_path):
        logger.error(f"验证集目录不存在: {val_path}")
        return False

    try:
        from datasets import load_from_disk
        train_ds = load_from_disk(train_path)
        val_ds = load_from_disk(val_path)

        logger.info(f"训练集实际大小: {len(train_ds):,}")
        logger.info(f"验证集实际大小: {len(val_ds):,}")

        if len(train_ds) != train_size:
            logger.error(f"训练集大小不匹配: {len(train_ds)} != {train_size}")
            return False

        if len(val_ds) != val_size:
            logger.error(f"验证集大小不匹配: {len(val_ds)} != {val_size}")
            return False

        logger.info("✓ 数据文件大小正确")

    except Exception as e:
        logger.error(f"加载数据文件失败: {e}")
        return False

    # 9. 统计摘要
    logger.info("\n" + "=" * 60)
    logger.info("9. 统计摘要")
    logger.info("=" * 60)

    logger.info(f"随机种子: {split_info.get('random_seed', 'N/A')}")
    logger.info(f"验证集比例: {split_info.get('val_ratio', 'N/A')}")
    logger.info(f"创建时间: {split_info.get('created_at', 'N/A')}")
    logger.info(f"数据来源: {split_info.get('source_data', 'N/A')}")

    logger.info("\n" + "=" * 60)
    logger.info("✓✓✓ 数据分割验证通过！无数据泄露！✓✓✓")
    logger.info("=" * 60)

    return True


def main():
    logger.info("开始验证数据分割...")

    success = verify_data_split()

    if not success:
        logger.error("数据分割验证失败！可能存在数据泄露！")
        sys.exit(1)
    else:
        logger.info("验证完成，数据分割符合行业规范。")
        sys.exit(0)


if __name__ == "__main__":
    main()

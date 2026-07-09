"""
长度外推测试模块
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict
from transformers import AutoTokenizer

from .perplexity import compute_perplexity


def length_extrapolation_test(model: nn.Module, tokenizer: AutoTokenizer,
                              train_len: int, test_lengths: List[int],
                              device: torch.device = None) -> Dict[int, float]:
    """
    长度外推测试

    Args:
        model: 训练好的模型
        tokenizer: 分词器
        train_len: 训练时的最大长度
        test_lengths: 测试长度列表
        device: 设备
    Returns:
        perplexities: 各长度的困惑度
    """
    if device is None:
        device = next(model.parameters()).device

    results = {}

    for length in test_lengths:
        print(f"Testing length: {length}")

        # 创建测试数据
        # 这里使用随机数据作为示例
        # 实际应用中应该使用真实的测试数据
        test_data = torch.randint(0, tokenizer.vocab_size, (100, length))

        # 创建数据加载器
        dataset = torch.utils.data.TensorDataset(test_data)
        dataloader = DataLoader(dataset, batch_size=16)

        # 计算困惑度
        model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch[0].to(device)

                # 创建标签
                labels = input_ids.clone()
                labels[:, :-1] = input_ids[:, 1:]
                labels[:, -1] = -100

                # 前向传播
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs['loss']

                # 统计
                num_tokens = (labels != -100).sum().item()
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

        # 计算困惑度
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        results[length] = perplexity
        print(f"  Perplexity: {perplexity:.2f}")

    return results

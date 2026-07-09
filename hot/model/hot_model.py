"""
HOT 模型模块

频率-相位解耦方案的完整 HOT 模型。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from .hot_layer import HOTLayer, RMSNorm


class Config:
    """配置类，支持字典和属性访问"""

    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            elif isinstance(value, str):
                try:
                    fval = float(value)
                    if fval == int(fval) and 'e' not in value.lower() and '.' not in value:
                        setattr(self, key, int(fval))
                    else:
                        setattr(self, key, fval)
                except ValueError:
                    setattr(self, key, value)
            else:
                setattr(self, key, value)


class HOTModel(nn.Module):
    """
    完整的 HOT 模型

    结构：Embedding → N × HOTLayer → RMSNorm → LM Head

    相位 θ 是可训练的隐状态，跨层传递：
    - θ 初始化为零（或可学习）
    - 每层通过 θ ← θ + ω·Δt 更新
    - 不依赖任何层的输出（无循环依赖）
    """

    def __init__(self, config):
        super().__init__()
        if isinstance(config, dict):
            self.config = Config(config)
        else:
            self.config = config

        mc = self.config.model
        self.embed = nn.Embedding(mc.vocab_size, mc.hidden_size)
        self.layers = nn.ModuleList([
            HOTLayer(self.config, i) for i in range(mc.num_layers)
        ])
        self.norm = RMSNorm(mc.hidden_size)
        self.lm_head = nn.Linear(mc.hidden_size, mc.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # 权重绑定

        # 相位 θ：可训练的隐状态，初始化为零
        self.theta_init = nn.Parameter(torch.zeros(1))

        # 退火调度（训练时由 Trainer 设置）
        self.annealing_schedule = None
        self.global_step = 0

    def forward(self, input_ids: torch.Tensor,
                labels: torch.Tensor = None) -> Dict:
        """
        Args:
            input_ids: [B, N] 输入 token IDs
            labels: [B, N] 标签（训练时）
        Returns:
            logits: [B, N, V]
            loss: 标量损失（如果提供了 labels）
            thetas: 各层相位列表
            attentions: 各层注意力权重列表
        """
        B, N = input_ids.shape
        H = self.config.model.num_heads

        # 初始化相位：θ_init 广播到 [B, H, N]
        theta = self.theta_init.expand(B, H, N).contiguous()

        x = self.embed(input_ids)

        all_thetas = []
        all_attentions = []

        for layer in self.layers:
            # 计算退火系数
            if self.annealing_schedule is not None:
                beta = self.annealing_schedule.get_beta(self.global_step)
            else:
                beta = 1.0

            x, theta, attn_weights = layer(x, theta, beta)

            all_thetas.append(theta)
            all_attentions.append(attn_weights)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        return {
            'logits': logits,
            'loss': loss,
            'thetas': all_thetas,
            'attentions': all_attentions,
        }

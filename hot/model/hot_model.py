"""
HOT 模型模块

频率-相位解耦方案的完整 HOT 模型。

优化：
- 预计算 sin/cos(theta) 一次，所有层共享
- 预计算 position 索引，注册为 buffer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

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

    结构：Embedding -> N x HOTLayer -> RMSNorm -> LM Head

    相位 θ 是可训练的隐状态，跨层传递。
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

        # 退火系数（由 Trainer 在编译区域外计算后传入）
        self._beta = 1.0

        # 预计算位置索引（注册为 buffer，随模型自动迁移设备）
        self.register_buffer('_pos_buf', torch.arange(1, 513, dtype=torch.float32),
                             persistent=False)

    def forward(self, input_ids: torch.Tensor,
                labels: torch.Tensor = None,
                beta: float = 1.0,
                return_details: bool = False, **kwargs) -> Dict:
        """
        Args:
            input_ids: [B, N] 输入 token IDs
            labels: [B, N] 标签（训练时）
            beta: 退火系数 β ∈ [0, 1]
            return_details: 是否返回详细信息（训练时建议关闭以节省内存）
            **kwargs: 接受但忽略额外参数
        Returns:
            logits: [B, N, V]
            loss: 标量损失（如果提供了 labels）
            thetas: 各层相位列表（仅 return_details=True）
            attentions: 各层注意力权重列表（仅 return_details=True）
        """
        B, N = input_ids.shape
        H = self.config.model.num_heads

        # 初始化相位：θ_init 广播到 [B, H, N]
        theta = self.theta_init.expand(B, H, N).contiguous()

        # 位置索引（从预注册 buffer 截取）
        pos = self._pos_buf[:N].to(dtype=theta.dtype)

        x = self.embed(input_ids)

        if return_details:
            all_thetas = []
            all_attentions = []

        for layer in self.layers:
            x, theta, attn_weights = layer(x, theta, beta, pos)

            if return_details:
                all_thetas.append(theta)
                all_attentions.append(attn_weights)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        result = {
            'logits': logits,
            'loss': loss,
        }

        if return_details:
            result['thetas'] = all_thetas
            result['attentions'] = all_attentions

        return result

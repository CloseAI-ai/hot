"""
HOT 模型模块

频率-相位解耦方案的完整 HOT 模型。

优化：
- 预计算 sin/cos(theta) 一次，所有层共享
- 预计算 position 索引，注册为 buffer
- 预计算因果 mask（布尔矩阵），所有层共享
- FlexAttention block mask 缓存（按序列长度）
- 梯度检查点：用计算换内存，允许更大 batch
- 分块交叉熵：避免实体化 [B,N,V] logits 张量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

from .hot_layer import HOTLayer, RMSNorm, _FLEX_ATTENTION_AVAILABLE


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

        # 预计算因果 mask 布尔矩阵（所有层共享，避免每层每步重建）
        self.register_buffer('_causal_mask', torch.triu(
            torch.ones(512, 512, dtype=torch.bool), diagonal=1
        ), persistent=False)

        # FlexAttention block mask 缓存（按序列长度 N 缓存）
        self._flex_block_masks = {}
        self._flex_available = _FLEX_ATTENTION_AVAILABLE

        # 梯度检查点（由 Trainer 设置）
        self._use_grad_checkpoint = False

    def set_gradient_checkpointing(self, enabled: bool):
        """启用/禁用梯度检查点"""
        self._use_grad_checkpoint = enabled

    @staticmethod
    def _chunked_cross_entropy(hidden: torch.Tensor, weight: torch.Tensor,
                                labels: torch.Tensor, chunk_size: int = 4096) -> torch.Tensor:
        """
        分块交叉熵：避免实体化 [B*N, V] logits 张量

        Args:
            hidden: [B*N, D] 隐藏状态
            weight: [V, D] LM head 权重（与 embedding 共享）
            labels: [B*N] 标签
            chunk_size: 每块处理的 token 数
        Returns:
            loss: 标量损失
        """
        total_tokens = hidden.shape[0]
        total_loss = hidden.new_zeros(())

        for i in range(0, total_tokens, chunk_size):
            end = min(i + chunk_size, total_tokens)
            chunk_logits = F.linear(hidden[i:end], weight)
            chunk_loss = F.cross_entropy(
                chunk_logits, labels[i:end], reduction='sum'
            )
            total_loss = total_loss + chunk_loss

        return total_loss / total_tokens

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

        # 初始化相位：theta_init 广播到 [B, H, N]
        # 使用 expand（视图，不拷贝）而非 expand().contiguous()（拷贝）
        # 后续 phase_dynamics 的加法会自然产生新的连续张量
        theta = self.theta_init.expand(B, H, N)

        # 位置索引（从预注册 buffer 截取）
        pos = self._pos_buf[:N].to(dtype=theta.dtype)

        # 因果 mask（所有层共享同一份）
        causal_mask = self._causal_mask[:N, :N]

        # FlexAttention block mask（按 N 缓存，避免重复创建）
        # 仅在 CUDA + 非梯度检查点模式下使用 FlexAttention
        # （梯度检查点与 FlexAttention 在 PyTorch 2.13 存在兼容性问题）
        if self._flex_available and input_ids.is_cuda and not self._use_grad_checkpoint:
            if N not in self._flex_block_masks:
                from torch.nn.attention.flex_attention import create_block_mask

                def causal_mask_fn(b, h, q_idx, kv_idx):
                    return q_idx >= kv_idx

                self._flex_block_masks[N] = create_block_mask(
                    causal_mask_fn, B=None, H=None,
                    Q_LEN=N, KV_LEN=N,
                    device=input_ids.device,
                )
            # 设为层属性而非函数参数，让 dynamo 将 BlockMask 视为
            # 模块常量，避免 graph break
            flex_bm = self._flex_block_masks[N]
            for layer in self.layers:
                layer._flex_block_mask = flex_bm
            # FlexAttention 已启用（不打印 BlockMask 对象，避免 torch.compile 兼容性问题）

        x = self.embed(input_ids)

        if return_details:
            all_thetas = []
            all_attentions = []

        for layer in self.layers:
            if self._use_grad_checkpoint and self.training and not return_details:
                x, theta, _ = grad_checkpoint(
                    layer, x, theta, beta, pos, causal_mask,
                    use_reentrant=False,
                )
            else:
                x, theta, attn_weights = layer(
                    x, theta, beta, pos, causal_mask
                )

            if return_details:
                all_thetas.append(theta)
                all_attentions.append(attn_weights)

        x = self.norm(x)

        loss = None
        if labels is not None:
            if self.training:
                # 分块交叉熵：避免实体化 [B*N, V] logits（省 ~400MB+ 显存）
                x_flat = x.reshape(-1, x.shape[-1])
                labels_flat = labels.reshape(-1)
                loss = self._chunked_cross_entropy(
                    x_flat, self.lm_head.weight, labels_flat
                )
                logits = None  # 训练时不保留 logits 以节省内存
            else:
                logits = self.lm_head(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), labels.view(-1)
                )
        else:
            logits = self.lm_head(x)

        result = {
            'logits': logits,
            'loss': loss,
        }

        if return_details:
            result['thetas'] = all_thetas
            result['attentions'] = all_attentions

        return result

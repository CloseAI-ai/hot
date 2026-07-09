"""
训练主循环模块
"""

import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any
from tqdm import tqdm

from .optimizer import configure_optimizer
from .scheduler import get_scheduler
from .annealing import ProgressivePhaseAnnealing

logger = logging.getLogger(__name__)


class Trainer:
    """
    训练器

    支持混合精度、梯度累积、Wandb 日志
    """

    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config

        # 从正确的配置层级读取训练参数
        train_cfg = config.get('training', {})

        # 设备配置
        device_str = config.get('device', 'auto')
        if device_str == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device_str)

        self.model.to(self.device)

        # 优化器和调度器（传入 training 子配置）
        self.optimizer = configure_optimizer(model, train_cfg)
        self.scheduler = get_scheduler(self.optimizer, train_cfg)

        # 退火调度
        hot_cfg = config.get('hot', {})
        annealing_cfg = hot_cfg.get('annealing', {})
        self.annealing = ProgressivePhaseAnnealing(
            warmup_steps=annealing_cfg.get('warmup_steps', 2000),
            schedule=annealing_cfg.get('schedule', 'cosine'),
        )

        # 训练配置
        self.max_steps = train_cfg.get('max_steps', 100000)
        self.gradient_accumulation = train_cfg.get('gradient_accumulation', 1)
        self.max_grad_norm = train_cfg.get('max_grad_norm', 1.0)
        self.precision = train_cfg.get('precision', 'bf16')
        self.log_every = train_cfg.get('log_every', 100)

        # 混合精度
        use_amp = self.device.type == 'cuda' and self.precision == 'bf16'
        self.use_amp = use_amp
        self.scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

        # 日志
        self.use_wandb = config.get('use_wandb', False)
        self._wandb = None

        # 检查点
        ckpt_cfg = config.get('checkpoint', {})
        self.save_dir = ckpt_cfg.get('save_dir', 'checkpoints')
        self.save_every = ckpt_cfg.get('save_every', 1000)
        os.makedirs(self.save_dir, exist_ok=True)

        # 评估配置
        eval_cfg = config.get('evaluation', {})
        self.eval_every = eval_cfg.get('eval_every', 500)
        self.eval_steps = eval_cfg.get('eval_steps', 100)

        # 训练状态
        self.global_step = 0
        self.micro_step = 0
        self.best_loss = float('inf')

    def _init_wandb(self):
        """延迟初始化 Wandb"""
        if self._wandb is not None:
            return
        try:
            import wandb
            self._wandb = wandb
            project = self.config.get('wandb_project', 'hot')
            # 将 config 转为可序列化的 flat dict
            flat = self._flatten_config(self.config)
            wandb.init(project=project, config=flat)
        except Exception as e:
            logger.warning(f"Wandb 初始化失败，跳过: {e}")
            self.use_wandb = False

    @staticmethod
    def _flatten_config(d: Any, parent_key: str = '', sep: str = '/') -> dict:
        """将嵌套字典展平为 JSON 可序列化的格式"""
        items = {}
        if isinstance(d, dict):
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.update(Trainer._flatten_config(v, new_key, sep))
                elif isinstance(v, (int, float, str, bool)):
                    items[new_key] = v
        return items

    def train(self, train_dataloader: DataLoader,
              eval_dataloader: Optional[DataLoader] = None):
        """
        训练模型

        Args:
            train_dataloader: 训练数据加载器
            eval_dataloader: 评估数据加载器
        """
        if self.use_wandb:
            self._init_wandb()

        self.model.train()
        self.optimizer.zero_grad()

        logger.info(f"开始训练：max_steps={self.max_steps}, "
                    f"grad_accum={self.gradient_accumulation}, "
                    f"device={self.device}")

        pbar = tqdm(total=self.max_steps, desc="Training")

        while self.global_step < self.max_steps:
            for batch in train_dataloader:
                if self.global_step >= self.max_steps:
                    break

                # 更新退火系数
                self.model.global_step = self.global_step
                self.model.annealing_schedule = self.annealing

                # 前向 + 反向
                loss = self._training_step(batch)
                self.scaler.scale(loss / self.gradient_accumulation).backward()
                self.micro_step += 1

                # 梯度累积达到阈值时执行优化器步进
                if self.micro_step % self.gradient_accumulation == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    self.global_step += 1
                    pbar.update(1)

                    # 日志
                    if self.global_step % self.log_every == 0:
                        lr = self.optimizer.param_groups[0]['lr']
                        beta = self.annealing.get_beta(self.global_step)
                        self._log_metrics({
                            'train_loss': loss.item(),
                            'learning_rate': lr,
                            'annealing_beta': beta,
                        })

                    # 评估
                    if (eval_dataloader is not None
                            and self.global_step % self.eval_every == 0):
                        eval_loss = self._evaluate(eval_dataloader)
                        self._log_metrics({'eval_loss': eval_loss})

                        if eval_loss < self.best_loss:
                            self.best_loss = eval_loss
                            self._save_checkpoint('best_model')

                    # 定期保存
                    if self.global_step % self.save_every == 0:
                        self._save_checkpoint(f'step_{self.global_step}')

        pbar.close()
        self._save_checkpoint('final_model')
        logger.info("训练完成")

    def _training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """单个训练步骤"""
        batch = {k: v.to(self.device) for k, v in batch.items()}

        ctx = torch.amp.autocast('cuda', enabled=self.use_amp)
        with ctx:
            outputs = self.model(**batch)
            loss = outputs['loss']

        return loss

    def _evaluate(self, eval_dataloader: DataLoader) -> float:
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                ctx = torch.amp.autocast('cuda', enabled=self.use_amp)
                with ctx:
                    outputs = self.model(**batch)
                    loss = outputs['loss']

                total_loss += loss.item()
                num_batches += 1

                if num_batches >= self.eval_steps:
                    break

        self.model.train()
        return total_loss / max(num_batches, 1)

    def _log_metrics(self, metrics: Dict[str, float]):
        """记录指标"""
        if self.use_wandb and self._wandb is not None:
            self._wandb.log(metrics, step=self.global_step)
        else:
            # 回退到 logger
            parts = [f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}"
                     for k, v in metrics.items()]
            logger.info(f"[step {self.global_step}] {', '.join(parts)}")

    def _save_checkpoint(self, name: str):
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_loss': self.best_loss,
        }

        path = os.path.join(self.save_dir, f'{name}.pt')
        torch.save(checkpoint, path)
        logger.info(f"检查点已保存: {path}")

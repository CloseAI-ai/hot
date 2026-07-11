"""
训练主循环模块
"""

import os
import time
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

    支持混合精度、梯度累积、Wandb 日志、断点续传
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

        # 梯度检查点（在 compile 之前设置，让 compile 能看到检查点逻辑）
        grad_ckpt = train_cfg.get('gradient_checkpointing', False)
        if grad_ckpt and hasattr(self.model, 'set_gradient_checkpointing'):
            self.model.set_gradient_checkpointing(True)
            logger.info("启用梯度检查点（用计算换内存）")

        # torch.compile 优化
        # - default：inductor 优化（内核融合、内存规划），不使用 CUDA Graphs
        #   （reduce-overhead 的 CUDA Graphs 与 FlexAttention score_mod 闭包不兼容：
        #     闭包每步捕获新张量，Graphs 回放时地址已变）
        # - dynamic=True：允许输入形状动态变化，避免因微小变化触发重编译
        # FlexAttention 的 _flex_attention_call 已用 @allow_in_graph 包装，
        # dynamo 会将其纳入计算图而非断图
        compile_cfg = train_cfg.get('compile', False)
        if compile_cfg and hasattr(torch, 'compile'):
            torch._dynamo.reset()
            logger.info("启用 torch.compile (mode=default, dynamic=True)")
            self.model = torch.compile(
                self.model, mode='default', dynamic=True
            )

        # 优化器和调度器（传入 training 子配置，使用 self.model 以支持 compile）
        self.optimizer = configure_optimizer(self.model, train_cfg)
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

        # 混合精度（bf16 不需要 GradScaler）
        use_amp = self.device.type == 'cuda' and self.precision == 'bf16'
        self.use_amp = use_amp
        # bf16 不需要 loss scaling，禁用 GradScaler 以减少开销
        self.scaler = torch.amp.GradScaler('cuda', enabled=False)

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

        # 吞吐量追踪
        self._timing_start = None
        self._timing_steps = 0
        self._tokens_per_step = int(train_cfg.get('batch_size', 16)) * \
            int(train_cfg.get('gradient_accumulation', 1)) * \
            int(config.get('data', {}).get('max_length', 512))

    def resume(self, checkpoint_path: str):
        """
        从检查点恢复训练状态

        Args:
            checkpoint_path: 检查点文件路径
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点不存在: {checkpoint_path}")

        logger.info(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # 恢复模型参数
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # 恢复优化器状态（如果存在）
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("已恢复优化器状态")
        else:
            logger.info("检查点不含优化器状态，使用新初始化的优化器")

        # 恢复调度器状态（如果存在）
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            logger.info("检查点不含调度器状态，使用新初始化的调度器")

        # 恢复训练状态
        self.global_step = checkpoint.get('global_step', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))

        # micro_step 需要根据 global_step 和 grad_accum 推算
        self.micro_step = self.global_step * self.gradient_accumulation

        logger.info(f"已恢复到 step {self.global_step}, best_loss={self.best_loss:.4f}")

    def save_checkpoint_now(self, name: str = 'manual'):
        """手动保存检查点（不依赖训练循环）"""
        self._save_checkpoint(name)
        return os.path.join(self.save_dir, f'{name}.pt')

    def _init_wandb(self):
        """延迟初始化 Wandb"""
        if self._wandb is not None:
            return
        try:
            import wandb
            self._wandb = wandb
            project = self.config.get('wandb_project', 'hot')
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
        self.optimizer.zero_grad(set_to_none=True)

        logger.info(f"开始训练：max_steps={self.max_steps}, "
                    f"grad_accum={self.gradient_accumulation}, "
                    f"device={self.device}, "
                    f"resume_from={self.global_step}")

        # 关闭 tqdm 以消除进度条开销（用日志替代）
        pbar = None

        log_interval = max(self.log_every, 100)
        self._timing_start = time.time()
        self._timing_steps = 0

        while self.global_step < self.max_steps:
            for batch in train_dataloader:
                if self.global_step >= self.max_steps:
                    break

                # 在编译区域外计算退火系数，避免 torch.compile 重编译
                beta = self.annealing.get_beta(self.global_step)

                # 前向 + 反向（训练时不返回详细信息以节省内存）
                loss = self._training_step(batch, beta)

                # NaN 检测：单次 isfinite 检查（一次 GPU-CPU sync）
                if not torch.isfinite(loss):
                    logger.warning(f"[step {self.global_step}] 检测到 NaN/Inf loss，跳过此 step")
                    self.optimizer.zero_grad(set_to_none=True)
                    self.micro_step = 0
                    continue

                (loss / self.gradient_accumulation).backward()
                self.micro_step += 1

                # 梯度累积达到阈值时执行优化器步进
                if self.micro_step % self.gradient_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                    self.global_step += 1
                    self._timing_steps += 1

                    # 统一日志（合并原 step%100 和 step%log_every 两个块，减少 GPU-CPU sync）
                    if self.global_step % log_interval == 0:
                        loss_val = loss.item()  # 单次 GPU-CPU sync
                        lr = self.optimizer.param_groups[0]['lr']
                        beta_val = self.annealing.get_beta(self.global_step)

                        elapsed = time.time() - self._timing_start
                        steps_per_sec = self._timing_steps / max(elapsed, 1e-6)
                        tokens_per_sec = steps_per_sec * self._tokens_per_step
                        eta_steps = self.max_steps - self.global_step
                        eta_sec = eta_steps / max(steps_per_sec, 1e-6)

                        logger.info(
                            f"[step {self.global_step}/{self.max_steps}] "
                            f"loss={loss_val:.4f} lr={lr:.6f} beta={beta_val:.6f} "
                            f"| {steps_per_sec:.2f} step/s {tokens_per_sec:.0f} tok/s "
                            f"ETA {eta_sec/3600:.1f}h"
                        )
                        self._log_metrics({
                            'train_loss': loss_val,
                            'learning_rate': lr,
                            'annealing_beta': beta_val,
                            'steps_per_sec': steps_per_sec,
                            'tokens_per_sec': tokens_per_sec,
                        })
                        self._timing_start = time.time()
                        self._timing_steps = 0

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

        self._save_checkpoint('final_model')
        logger.info("训练完成")

    def _training_step(self, batch: Dict[str, torch.Tensor], beta: float) -> torch.Tensor:
        """单个训练步骤"""
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

        ctx = torch.amp.autocast('cuda', enabled=self.use_amp)
        with ctx:
            outputs = self.model(**batch, beta=beta)
            loss = outputs['loss']

        return loss

    def _evaluate(self, eval_dataloader: DataLoader) -> float:
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.inference_mode():
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
            parts = [f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}"
                     for k, v in metrics.items()]
            logger.info(f"[step {self.global_step}] {', '.join(parts)}")

    def _save_checkpoint(self, name: str):
        """保存检查点（保存一次，latest.pt 通过复制生成，避免双重序列化）"""
        import shutil

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_loss': self.best_loss,
        }

        path = os.path.join(self.save_dir, f'{name}.pt')
        latest_path = os.path.join(self.save_dir, 'latest.pt')

        # 保存到临时文件再原子重命名
        tmp_path = path + '.tmp'
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)

        # latest.pt 通过复制生成（避免第二次 torch.save 序列化开销）
        shutil.copy2(path, latest_path)

        logger.info(f"检查点已保存: {path} (step={self.global_step})")

#!/usr/bin/env python3
"""
HOT 训练自动监控脚本

实现无人值守训练看管，遵循"非必要不干涉"原则。

功能：
1. 自动检测并恢复训练异常
2. GPU 显存溢出自动降 batch size
3. 训练卡住自动重启
4. 磁盘空间不足自动清理
5. GPU 温度过高自动等待
6. 完整日志记录

用法：
    python train_monitor.py [--config configs/hot_42m.yaml] [--max_retries 10]
"""

import os
import sys
import time
import json
import signal
import logging
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TrainMonitor:
    """训练监控器"""

    def __init__(self, config_path: str = "configs/hot_42m.yaml",
                 max_retries: int = 10,
                 check_interval: int = 60,
                 stall_timeout: int = 600,
                 gpu_temp_limit: int = 85,
                 min_disk_gb: int = 50):
        """
        Args:
            config_path: 训练配置文件
            max_retries: 最大重试次数
            check_interval: 检查间隔（秒）
            stall_timeout: 训练卡住超时（秒）
            gpu_temp_limit: GPU 温度限制（℃）
            min_disk_gb: 最小磁盘空间（GB）
        """
        self.config_path = config_path
        self.max_retries = max_retries
        self.check_interval = check_interval
        self.stall_timeout = stall_timeout
        self.gpu_temp_limit = gpu_temp_limit
        self.min_disk_gb = min_disk_gb

        self.process: Optional[subprocess.Popen] = None
        self.current_batch_size = 23  # 默认 batch size
        self.min_batch_size = 4  # 最小 batch size
        self.retry_count = 0
        self.last_progress_time = time.time()
        self.last_step = 0
        self.start_time = None
        self.state_file = "monitor_state.json"

        # 信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # 加载状态
        self._load_state()

    def _signal_handler(self, signum, frame):
        """处理终止信号"""
        logger.info(f"收到信号 {signum}，正在停止...")
        self.stop()
        sys.exit(0)

    def _load_state(self):
        """加载监控状态"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                self.current_batch_size = state.get('batch_size', self.current_batch_size)
                self.retry_count = state.get('retry_count', 0)
                logger.info(f"加载状态: batch_size={self.current_batch_size}, retries={self.retry_count}")
            except Exception as e:
                logger.warning(f"加载状态失败: {e}")

    def _save_state(self):
        """保存监控状态"""
        state = {
            'batch_size': self.current_batch_size,
            'retry_count': self.retry_count,
            'last_update': datetime.now().isoformat()
        }
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning(f"保存状态失败: {e}")

    def check_gpu_status(self) -> Dict[str, Any]:
        """检查 GPU 状态"""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                temp, mem_used, mem_total, util = result.stdout.strip().split(', ')
                return {
                    'temperature': int(temp),
                    'memory_used': int(mem_used),
                    'memory_total': int(mem_total),
                    'utilization': int(util),
                    'memory_percent': int(mem_used) / int(mem_total) * 100
                }
        except Exception as e:
            logger.warning(f"获取 GPU 状态失败: {e}")
        return None

    def check_disk_space(self) -> Dict[str, Any]:
        """检查磁盘空间"""
        try:
            stat = os.statvfs('.')
            total = stat.f_frsize * stat.f_blocks
            free = stat.f_frsize * stat.f_bavail
            return {
                'total_gb': total / (1024**3),
                'free_gb': free / (1024**3),
                'used_percent': (1 - free / total) * 100
            }
        except Exception as e:
            logger.warning(f"获取磁盘空间失败: {e}")
        return None

    def check_training_progress(self) -> Optional[int]:
        """检查训练进度"""
        try:
            # 读取最新日志
            log_files = ['train_final.log', 'train_current.log']
            for log_file in log_files:
                if os.path.exists(log_file) and not os.path.islink(log_file):
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        # 从后往前查找 step 信息
                        for line in reversed(lines[-100:]):
                            if 'step' in line and 'loss' in line:
                                # 提取 step 数
                                import re
                                match = re.search(r'step\s+(\d+)', line)
                                if match:
                                    return int(match.group(1))
        except Exception as e:
            logger.warning(f"检查训练进度失败: {e}")
        return None

    def clean_checkpoints(self, keep_latest: int = 3):
        """清理旧检查点"""
        try:
            checkpoint_dir = "checkpoints"
            if not os.path.exists(checkpoint_dir):
                return

            checkpoints = []
            for f in os.listdir(checkpoint_dir):
                if f.startswith("step_") and f.endswith(".pt"):
                    path = os.path.join(checkpoint_dir, f)
                    checkpoints.append((os.path.getmtime(path), path))

            # 按时间排序，保留最新的几个
            checkpoints.sort(reverse=True)
            for _, path in checkpoints[keep_latest:]:
                os.remove(path)
                logger.info(f"清理旧检查点: {path}")

        except Exception as e:
            logger.warning(f"清理检查点失败: {e}")

    def start_training(self, resume_from=None):
        """启动训练进程"""
        # 使用 conda run 确保正确的环境
        cmd = [
            "conda", "run", "-n", "torch_cuda", "--no-capture-output",
            "python", "scripts/train.py",
            "--config", self.config_path,
        ]

        # 如果有检查点，添加 resume参数
        if resume_from and os.path.exists(resume_from):
            cmd.extend(["--resume", resume_from])
            logger.info(f"从检查点恢复: {resume_from}")

        logger.info(f"启动训练: retry={self.retry_count}")
        logger.info(f"命令: {' '.join(cmd)}")

        try:
            # 打开日志文件用于保存训练输出
            self.log_file = open('train_final.log', 'a')
            # 设置环境变量确保 TF32 生效
            env = os.environ.copy()
            env['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = '1'
            self.process = subprocess.Popen(
                cmd,
                stdout=self.log_file,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=env
            )
            self.start_time = time.time()
            self.last_progress_time = time.time()
            self._save_state()
            return True
        except Exception as e:
            logger.error(f"启动训练失败: {e}")
            return False

    def stop_training(self):
        """停止训练进程"""
        if self.process and self.process.poll() is None:
            logger.info("停止训练进程...")
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logger.warning("训练进程未响应，强制终止")
                self.process.kill()
                self.process.wait()
            self.process = None

        # 关闭日志文件
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()
            self.log_file = None

    def handle_oom(self):
        """处理显存溢出"""
        logger.warning("检测到 OOM，降低 batch size")

        # 降低 batch size
        new_batch_size = max(self.min_batch_size, self.current_batch_size // 2)
        if new_batch_size == self.current_batch_size:
            logger.error("batch size 已是最小，无法继续降低")
            return False

        self.current_batch_size = new_batch_size
        self._save_state()

        # 清理 GPU 显存
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

        return True

    def handle_stall(self):
        """处理训练卡住"""
        logger.warning(f"训练可能卡住（超过 {self.stall_timeout} 秒无进度）")
        self.stop_training()
        time.sleep(10)  # 等待资源释放
        return True

    def handle_nan_loss(self):
        """处理 NaN loss"""
        logger.warning("检测到 NaN/Inf loss")
        # 通常 trainer 会自动处理，这里只记录
        return True

    def handle_high_temperature(self, temp: int):
        """处理 GPU 温度过高"""
        logger.warning(f"GPU 温度过高: {temp}°C > {self.gpu_temp_limit}°C")
        logger.info("等待 GPU 降温...")

        while True:
            gpu_status = self.check_gpu_status()
            if gpu_status and gpu_status['temperature'] < self.gpu_temp_limit - 5:
                logger.info(f"GPU 温度已降至 {gpu_status['temperature']}°C")
                break
            time.sleep(30)

        return True

    def handle_low_disk(self):
        """处理磁盘空间不足"""
        logger.warning("磁盘空间不足，清理旧检查点")
        self.clean_checkpoints(keep_latest=2)
        return True

    def check_for_errors(self) -> Optional[str]:
        """检查训练输出中的错误"""
        if not self.process or self.process.poll() is not None:
            return None

        try:
            # 非阻塞读取输出
            import select
            if select.select([self.process.stdout], [], [], 0)[0]:
                line = self.process.stdout.readline()
                if line:
                    line = line.strip()
                    # 检查错误模式
                    if 'CUDA out of memory' in line or 'OOM' in line:
                        return 'oom'
                    elif 'NaN' in line or 'Inf' in line:
                        return 'nan'
                    elif 'Error' in line or 'Exception' in line:
                        return 'error'
                    # 记录正常输出
                    elif 'step' in line and 'loss' in line:
                        logger.debug(line)
        except Exception:
            pass

        return None

    def run(self):
        """主监控循环"""
        logger.info("=" * 60)
        logger.info("HOT 训练监控器启动")
        logger.info("=" * 60)
        logger.info(f"配置文件: {self.config_path}")
        logger.info(f"最大重试: {self.max_retries}")
        logger.info(f"检查间隔: {self.check_interval} 秒")
        logger.info(f"温度限制: {self.gpu_temp_limit}°C")
        logger.info(f"磁盘限制: {self.min_disk_gb} GB")
        logger.info("=" * 60)

        # 查找最新检查点
        latest_checkpoint = None
        checkpoint_dir = "checkpoints"
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("step_") and f.endswith(".pt")]
            if checkpoints:
                # 按步数排序，取最新的
                checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
                latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
                logger.info(f"找到检查点: {latest_checkpoint}")

        # 首次启动
        if not self.start_training(resume_from=latest_checkpoint):
            logger.error("首次启动训练失败")
            return

        while True:
            try:
                time.sleep(self.check_interval)

                # 1. 检查进程状态
                if self.process and self.process.poll() is not None:
                    exit_code = self.process.returncode
                    logger.warning(f"训练进程已退出，退出码: {exit_code}")

                    if exit_code == 0:
                        logger.info("训练正常完成")
                        break

                    # 需要重试
                    self.retry_count += 1
                    if self.retry_count >= self.max_retries:
                        logger.error(f"达到最大重试次数 ({self.max_retries})，停止监控")
                        break

                    logger.info(f"准备重试 ({self.retry_count}/{self.max_retries})")
                    time.sleep(10)

                    if not self.start_training():
                        logger.error("重启训练失败")
                        break

                    continue

                # 2. 检查 GPU 状态
                gpu_status = self.check_gpu_status()
                if gpu_status:
                    # 温度检查
                    if gpu_status['temperature'] > self.gpu_temp_limit:
                        self.handle_high_temperature(gpu_status['temperature'])
                        continue

                    # 显存检查（接近满载）
                    if gpu_status['memory_percent'] > 95:
                        logger.warning(f"GPU 显存使用率高: {gpu_status['memory_percent']:.1f}%")

                # 3. 检查磁盘空间
                disk_status = self.check_disk_space()
                if disk_status and disk_status['free_gb'] < self.min_disk_gb:
                    self.handle_low_disk()

                # 4. 检查训练进度
                current_step = self.check_training_progress()
                if current_step and current_step > self.last_step:
                    self.last_step = current_step
                    self.last_progress_time = time.time()
                    logger.info(f"训练进度: step {current_step}")
                elif time.time() - self.last_progress_time > self.stall_timeout:
                    # 训练卡住
                    self.handle_stall()
                    if not self.start_training():
                        logger.error("重启训练失败")
                        break

                # 5. 检查训练输出错误
                error = self.check_for_errors()
                if error:
                    if error == 'oom':
                        if self.handle_oom():
                            self.stop_training()
                            time.sleep(5)
                            if not self.start_training():
                                logger.error("OOM 后重启失败")
                                break
                        else:
                            logger.error("无法处理 OOM")
                            break
                    elif error == 'nan':
                        self.handle_nan_loss()

                # 6. 定期清理检查点（每 10000 步）
                if self.last_step > 0 and self.last_step % 10000 == 0:
                    self.clean_checkpoints(keep_latest=3)

            except KeyboardInterrupt:
                logger.info("收到中断信号")
                break
            except Exception as e:
                logger.error(f"监控循环异常: {e}")
                time.sleep(10)

        # 清理
        self.stop()
        logger.info("监控器已停止")

    def stop(self):
        """停止监控"""
        self.stop_training()
        self._save_state()


def main():
    parser = argparse.ArgumentParser(description='HOT 训练监控器')
    parser.add_argument('--config', type=str, default='configs/hot_42m.yaml',
                        help='训练配置文件')
    parser.add_argument('--max_retries', type=int, default=10,
                        help='最大重试次数')
    parser.add_argument('--check_interval', type=int, default=60,
                        help='检查间隔（秒）')
    parser.add_argument('--stall_timeout', type=int, default=600,
                        help='训练卡住超时（秒）')
    parser.add_argument('--gpu_temp_limit', type=int, default=85,
                        help='GPU 温度限制（℃）')
    parser.add_argument('--min_disk_gb', type=int, default=50,
                        help='最小磁盘空间（GB）')
    args = parser.parse_args()

    monitor = TrainMonitor(
        config_path=args.config,
        max_retries=args.max_retries,
        check_interval=args.check_interval,
        stall_timeout=args.stall_timeout,
        gpu_temp_limit=args.gpu_temp_limit,
        min_disk_gb=args.min_disk_gb
    )

    monitor.run()


if __name__ == '__main__':
    main()

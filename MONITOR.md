# HOT 训练监控器使用指南

## 概述

训练监控器实现无人值守训练看管，遵循"非必要不干涉"原则。自动检测并处理各种异常情况。

## 功能特性

### 自动处理的异常

| 异常类型 | 检测方式 | 处理策略 |
|---------|---------|---------|
| 训练进程崩溃 | 进程退出码非零 | 自动重启（最多 10 次） |
| GPU OOM | 检测 "CUDA out of memory" | 降低 batch size 后重启 |
| NaN/Inf loss | 检测日志输出 | 记录警告（trainer 通常自动处理） |
| 训练卡住 | 600 秒无进度 | 重启训练进程 |
| GPU 温度过高 | 温度 > 85°C | 暂停等待降温 |
| 磁盘空间不足 | 剩余 < 50 GB | 清理旧检查点 |

### 非必要不干涉原则

- 训练正常运行时完全静默
- NaN/Inf loss 由 trainer 内部处理，监控器只记录
- GPU 温度稍高不干预，超过阈值才暂停
- 定期检查而非实时监控，减少系统开销

## 快速开始

### 方式 1：直接运行（推荐测试）

```bash
# 前台运行，可直接看到输出
make monitor-run

# 或直接调用
python train_monitor.py --config configs/hot_42m.yaml
```

### 方式 2：后台运行（推荐生产）

```bash
# 后台启动
make monitor-start

# 查看状态
make monitor-status

# 查看日志
make monitor-logs

# 停止
make monitor-stop
```

### 方式 3：systemd 服务（推荐长期运行）

```bash
# 安装服务（需要 sudo）
sudo bash install_monitor.sh

# 启动
sudo systemctl start hot-train-monitor

# 查看状态
sudo systemctl status hot-train-monitor

# 查看日志
sudo journalctl -u hot-train-monitor -f

# 停止
sudo systemctl stop hot-train-monitor

# 禁用开机启动
sudo systemctl disable hot-train-monitor
```

## 配置参数

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | configs/hot_42m.yaml | 训练配置文件 |
| `--max_retries` | 10 | 最大重试次数 |
| `--check_interval` | 60 | 检查间隔（秒） |
| `--stall_timeout` | 600 | 训练卡住超时（秒） |
| `--gpu_temp_limit` | 85 | GPU 温度限制（℃） |
| `--min_disk_gb` | 50 | 最小磁盘空间（GB） |

### 示例

```bash
# 自定义配置
python train_monitor.py \
    --config configs/hot_42m.yaml \
    --max_retries 5 \
    --check_interval 30 \
    --stall_timeout 300 \
    --gpu_temp_limit 80 \
    --min_disk_gb 100
```

## 状态文件

监控器会自动保存状态到 `monitor_state.json`：

```json
{
  "batch_size": 23,
  "retry_count": 0,
  "last_update": "2026-07-11T16:00:00"
}
```

状态会在以下情况更新：
- 启动训练
- OOM 后降低 batch size
- 重试计数增加

## 日志文件

| 文件 | 说明 |
|------|------|
| `monitor.log` | 监控器日志 |
| `train_final.log` | 训练输出日志 |
| `train_current.log` | 当前训练日志（符号链接） |

## 常见问题

### Q: 训练一直重启怎么办？

检查：
1. GPU 显存是否足够（尝试降低 batch size）
2. 数据集是否存在
3. 配置文件是否正确

```bash
# 手动测试训练
python scripts/train.py --config configs/hot_42m.yaml --batch_size 4
```

### Q: 如何恢复被监控器降低的 batch size？

删除状态文件，监控器会使用默认值：

```bash
rm monitor_state.json
```

### Q: 监控器本身崩溃了怎么办？

systemd 会自动重启监控器。如果使用后台运行方式：

```bash
# 重新启动
make monitor-start
```

### Q: 如何查看历史日志？

```bash
# 监控器日志
cat monitor.log

# systemd 日志
sudo journalctl -u hot-train-monitor --since "1 hour ago"
```

## 架构图

```
┌─────────────────────────────────────────────────────────┐
│                    监控主循环                            │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │
│  │ 进程检查 │  │ GPU检查 │  │ 磁盘检查│  │ 进度检查│   │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘   │
│       │            │            │            │         │
│       ▼            ▼            ▼            ▼         │
│  ┌─────────────────────────────────────────────────┐   │
│  │              异常检测与处理                      │   │
│  │  - OOM → 降低 batch size                       │   │
│  │  - 卡住 → 重启进程                             │   │
│  │  - 温度过高 → 等待降温                         │   │
│  │  - 磁盘不足 → 清理检查点                       │   │
│  └─────────────────────────────────────────────────┘   │
│                         │                              │
│                         ▼                              │
│  ┌─────────────────────────────────────────────────┐   │
│  │              状态保存与日志记录                  │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## 最佳实践

1. **首次运行**：使用前台模式测试，确认训练正常
2. **生产环境**：使用 systemd 服务，确保系统重启后自动恢复
3. **监控资源**：定期检查 GPU 温度和磁盘空间
4. **日志轮转**：定期清理旧日志，避免磁盘占满
5. **备份检查点**：定期备份 `checkpoints/best_model.pt`

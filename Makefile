.PHONY: install train evaluate ablation visualize test lint format type-check clean monitor status create-val-split verify-split monitor-start monitor-stop monitor-status monitor-logs web-monitor

# Conda 环境（CUDA 版 PyTorch）
CONDA = eval "$$(conda shell.bash hook 2>/dev/null)" && conda activate torch_cuda &&

# 安装依赖
install:
	$(CONDA) pip install -e . --index-url https://pypi.org/simple/
	$(CONDA) pip install -r requirements.txt --index-url https://pypi.org/simple/

# 训练模型（GPU）
train:
	$(CONDA) python scripts/train.py --config configs/hot_42m.yaml

# 训练 8M 模型
train-hot-8m:
	$(CONDA) python scripts/train.py --config configs/hot_8m.yaml

# 创建训练/验证集分割
create-val-split:
	$(CONDA) python scripts/create_validation_split.py

# 验证数据分割完整性
verify-split:
	$(CONDA) python scripts/verify_data_split.py

# 评估模型
evaluate:
	$(CONDA) python scripts/evaluate.py --config configs/hot_42m.yaml --checkpoint checkpoints/best_model.pt

# 运行消融实验
ablation:
	$(CONDA) python scripts/ablation.py --experiments no_gating full_coupling no_annealing

# 可视化
visualize:
	$(CONDA) python scripts/visualize.py --config configs/hot_42m.yaml --checkpoint checkpoints/best_model.pt

# 运行测试
test:
	$(CONDA) pytest tests/ -v

# 代码格式化
format:
	$(CONDA) black hot/ tests/ scripts/
	$(CONDA) isort hot/ tests/ scripts/

# 代码检查
lint:
	$(CONDA) flake8 hot/ tests/ scripts/

# 类型检查
type-check:
	$(CONDA) mypy hot/

# 清理生成的文件
clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache
	rm -rf build dist *.egg-info
	rm -rf checkpoints visualizations
	rm -rf wandb data/raw
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# 训练 RoPE 基线
train-rope:
	$(CONDA) python scripts/train.py --config configs/rope_42m.yaml

# 评估 RoPE 基线
evaluate-rope:
	$(CONDA) python scripts/evaluate.py --config configs/rope_42m.yaml --checkpoint checkpoints/rope_best_model.pt

# 所有检查
check: format lint type-check test

# 帮助
help:
	@echo "环境: torch_cuda (PyTorch + CUDA)"
	@echo ""
	@echo "Available commands:"
	@echo "  install       - Install dependencies"
	@echo "  train         - Train HOT 42M model (GPU)"
	@echo "  evaluate      - Evaluate HOT 42M model"
	@echo "  ablation      - Run ablation experiments"
	@echo "  visualize     - Generate visualizations"
	@echo "  test          - Run unit tests"
	@echo "  format        - Format code with black and isort"
	@echo "  lint          - Lint code with flake8"
	@echo "  type-check    - Type check with mypy"
	@echo "  clean         - Clean generated files"
	@echo "  train-rope    - Train RoPE 42M baseline"
	@echo "  evaluate-rope - Evaluate RoPE 42M baseline"
	@echo "  check         - Run all checks"
	@echo "  monitor       - Monitor training progress"
	@echo "  status        - Show project status"

# 监控训练进度
monitor:
	$(CONDA) python monitor.py

# 显示项目状态
status:
	@bash check_status.sh

# 启动训练监控（前台）
monitor-run:
	$(CONDA) python train_monitor.py --config configs/hot_42m.yaml

# 后台启动训练监控
monitor-start:
	@echo "启动训练监控（后台）..."
	@nohup $(CONDA) python train_monitor.py --config configs/hot_42m.yaml > monitor.log 2>&1 &
	@echo "PID: $$!"
	@echo "日志: tail -f monitor.log"

# 停止训练监控
monitor-stop:
	@echo "停止训练监控..."
	@pkill -f "train_monitor.py" || echo "未找到监控进程"
	@pkill -f "scripts/train.py" || echo "未找到训练进程"

# 查看监控状态
monitor-status:
	@echo "=== 监控进程状态 ==="
	@ps aux | grep -E "train_monitor|train.py" | grep -v grep || echo "无监控/训练进程"
	@echo ""
	@echo "=== GPU 状态 ==="
	@nvidia-smi --query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "无法获取 GPU 状态"
	@echo ""
	@echo "=== 最近日志 ==="
	@tail -20 monitor.log 2>/dev/null || echo "无监控日志"

# 查看监控日志
monitor-logs:
	@tail -f monitor.log

# 启动 Web 监控面板
web-monitor:
	@echo "启动 Web 监控面板..."
	@nohup $(CONDA) python monitor.py --port 8081 > /dev/null 2>&1 &
	@sleep 2
	@echo "Web 监控面板已启动: http://localhost:8081"
	@echo "查看日志: tail -f monitor.log"

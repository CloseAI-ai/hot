.PHONY: install train evaluate ablation visualize test lint format type-check clean

# Conda 环境（CUDA 版 PyTorch）
CONDA = eval "$$(conda shell.bash hook 2>/dev/null)" && conda activate torch_cuda &&

# 安装依赖
install:
	$(CONDA) pip install -e . --index-url https://pypi.org/simple/
	$(CONDA) pip install -r requirements.txt --index-url https://pypi.org/simple/

# 训练模型（GPU）
train:
	$(CONDA) python scripts/train.py --config configs/hot_42m.yaml

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

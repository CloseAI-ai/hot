"""模型前向传播和集成测试（频率-相位解耦方案）"""

import pytest
import torch
import yaml

from hot.model import HOTModel


@pytest.fixture
def config():
    with open('configs/hot_42m.yaml', 'r') as f:
        return yaml.safe_load(f)


@pytest.fixture
def model(config):
    return HOTModel(config)


class TestForward:
    """测试前向传播"""

    def test_without_labels(self, model, config):
        B, N = 2, 64
        input_ids = torch.randint(0, config['model']['vocab_size'], (B, N))

        # 默认不返回详细信息（节省内存）
        outputs = model(input_ids=input_ids)
        assert outputs['logits'].shape == (B, N, config['model']['vocab_size'])
        assert outputs['loss'] is None
        assert 'thetas' not in outputs  # 默认不返回

        # 使用 return_details=True 获取详细信息
        outputs = model(input_ids=input_ids, return_details=True)
        assert len(outputs['thetas']) == config['model']['num_layers']
        assert len(outputs['attentions']) == config['model']['num_layers']

        for theta in outputs['thetas']:
            assert theta.shape == (B, config['model']['num_heads'], N)

    def test_with_labels(self, model, config):
        B, N = 2, 64
        input_ids = torch.randint(0, config['model']['vocab_size'], (B, N))
        labels = torch.randint(0, config['model']['vocab_size'], (B, N))
        outputs = model(input_ids=input_ids, labels=labels)

        assert outputs['loss'] is not None
        assert outputs['loss'].dim() == 0
        assert outputs['loss'].item() > 0


class TestParameterCount:
    """测试参数量"""

    def test_approximately_42m(self, model):
        total = sum(p.numel() for p in model.parameters())
        assert 35e6 < total < 55e6, f"参数量 {total / 1e6:.2f}M 不在预期范围"

    def test_weight_tying(self, model):
        assert model.embed.weight is model.lm_head.weight

    def test_theta_is_parameter(self, model):
        """θ 是可训练参数（不是 buffer）"""
        assert isinstance(model.theta_init, torch.nn.Parameter)
        assert model.theta_init.requires_grad


class TestGradientFlow:
    """测试梯度流"""

    def test_all_params_receive_grad(self, model, config):
        B, N = 2, 32
        input_ids = torch.randint(0, config['model']['vocab_size'], (B, N))
        labels = torch.randint(0, config['model']['vocab_size'], (B, N))

        outputs = model(input_ids=input_ids, labels=labels)
        outputs['loss'].backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


class TestNoCircularDependency:
    """测试无循环依赖"""

    def test_forward_without_prev_attention(self, model, config):
        """前向传播不需要 prev_attention 参数"""
        input_ids = torch.randint(0, config['model']['vocab_size'], (1, 16))
        outputs = model(input_ids=input_ids)
        assert outputs['logits'].shape[0] == 1

    def test_phase_propagates_through_layers(self, model, config):
        """相位跨层传递，每层独立更新"""
        input_ids = torch.randint(0, config['model']['vocab_size'], (1, 16))
        outputs = model(input_ids=input_ids, return_details=True)

        # 各层相位应不同（因为每层独立更新）
        thetas = outputs['thetas']
        for i in range(len(thetas) - 1):
            assert not torch.allclose(thetas[i], thetas[i + 1], atol=1e-4)


class TestAnnealingIntegration:
    """测试退火调度集成"""

    def test_annealing_affects_output(self, model, config):
        input_ids = torch.randint(0, config['model']['vocab_size'], (1, 32))

        # beta=0（冻结期）
        out_beta0 = model(input_ids=input_ids, beta=0.0)

        # beta=1（退火完成期）
        out_beta1 = model(input_ids=input_ids, beta=1.0)

        # beta=0 和 beta=1 应产生不同输出
        assert not torch.allclose(out_beta0['logits'], out_beta1['logits'], atol=1e-3)

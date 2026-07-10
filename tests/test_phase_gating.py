"""相位门控单元测试（残差耦合方案）"""

import pytest
import torch
import math

from hot.model.phase_gating import PhaseGating


def make_config(gate_position='pre_softmax', alpha_init=0.1):
    return {'hot': {'gate_position': gate_position, 'alpha_init': alpha_init}}


def _prep(theta):
    """预计算 sin/cos，与模型行为一致"""
    return torch.cos(theta), torch.sin(theta)


@pytest.fixture
def pg():
    return PhaseGating(make_config())


class TestAlphaParameter:
    """测试可学习耦合强度 α"""

    def test_alpha_is_learnable(self, pg):
        assert isinstance(pg.alpha, torch.nn.Parameter)
        assert pg.alpha.requires_grad

    def test_alpha_default_value(self, pg):
        assert torch.allclose(pg.alpha.data, torch.tensor(0.1))


class TestResidualCoupling:
    """测试残差耦合机制"""

    def test_synchronized_phase(self, pg):
        """同相时 cos(0)=1，注意力增强"""
        B, H, N = 1, 1, 4
        scores = torch.zeros(B, H, N, N)
        theta = torch.ones(B, H, N) * 1.5  # 所有 token 同相
        cos_t, sin_t = _prep(theta)

        result = pg(scores, cos_t, sin_t)
        # 残差耦合：scores + softplus(α)·cos(0) = 0 + softplus(0.1)·1 > 0
        assert (result > 0).all()

    def test_opposite_phase(self, pg):
        """反相时跨对角线 logit 为负，但对角线（同相）仍为正"""
        B, H, N = 1, 1, 2
        scores = torch.zeros(B, H, N, N)
        theta = torch.tensor([[[0.0, math.pi]]])
        cos_t, sin_t = _prep(theta)

        result = pg(scores, cos_t, sin_t)
        # 对角线 cos(0)=1 → 正 logit
        # 跨对角线 cos(π)=-1 → 负 logit
        assert result[0, 0, 0, 0] > 0  # cos(0)
        assert result[0, 0, 0, 1] < 0  # cos(π)
        # softmax 后所有权重 > 0（内容注意力不丢失）
        weights = torch.softmax(result, dim=-1)
        assert (weights > 0).all()

    def test_alpha_zero_no_modulation(self):
        """α=0 时退化为标准注意力"""
        pg = PhaseGating(make_config(alpha_init=0.0))
        scores = torch.randn(1, 1, 4, 4)
        theta = torch.rand(1, 1, 4) * 2 * math.pi
        cos_t, sin_t = _prep(theta)

        result = pg(scores, cos_t, sin_t)
        pg.alpha.data.fill_(-10.0)
        result = pg(scores, cos_t, sin_t)
        assert torch.allclose(result, scores, atol=1e-4)

    def test_content_attention_preserved(self):
        """即使 cos(Δθ)=-1，softmax 后内容注意力仍保留"""
        pg = PhaseGating(make_config(alpha_init=1.0))
        B, H, N = 1, 1, 4
        scores = torch.tensor([[[[2.0, 1.0, 0.5, 0.1],
                                  [1.0, 2.0, 0.5, 0.1],
                                  [0.5, 0.5, 2.0, 1.0],
                                  [0.1, 0.1, 1.0, 2.0]]]])
        theta = torch.tensor([[[0.0, 0.0, math.pi, math.pi]]])
        cos_t, sin_t = _prep(theta)

        result = pg(scores, cos_t, sin_t)
        weights = torch.softmax(result, dim=-1)

        assert (weights > 0).all()
        assert torch.allclose(weights.sum(-1), torch.ones(1, 1, 1), atol=1e-5)


class TestForwardShapes:
    """测试前向传播形状"""

    def test_pre_softmax(self, pg):
        B, H, N = 2, 4, 8
        scores = torch.randn(B, H, N, N)
        theta = torch.rand(B, H, N) * 2 * math.pi
        cos_t, sin_t = _prep(theta)
        result = pg(scores, cos_t, sin_t)
        assert result.shape == (B, H, N, N)

    def test_none_gate(self):
        pg = PhaseGating(make_config('none'))
        scores = torch.randn(2, 4, 8, 8)
        theta = torch.rand(2, 4, 8) * 2 * math.pi
        cos_t, sin_t = _prep(theta)
        result = pg(scores, cos_t, sin_t)
        assert torch.equal(result, scores)


class TestGradientFlow:
    """测试梯度流"""

    def test_alpha_receives_gradient(self, pg):
        scores = torch.randn(1, 2, 4, 4)
        theta = torch.rand(1, 2, 4) * 2 * math.pi
        cos_t, sin_t = _prep(theta)
        result = pg(scores, cos_t, sin_t)
        result.sum().backward()
        assert pg.alpha.grad is not None

    def test_theta_receives_gradient(self, pg):
        scores = torch.randn(1, 2, 4, 4)
        theta = torch.rand(1, 2, 4) * 2 * math.pi
        theta.requires_grad_(True)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        result = pg(scores, cos_t, sin_t)
        result.sum().backward()
        assert theta.grad is not None

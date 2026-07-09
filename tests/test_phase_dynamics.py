"""相位动力学单元测试（频率-相位解耦方案）"""

import pytest
import torch
import math

from hot.model.phase_dynamics import PhaseDynamics
from hot.model.hot_layer import causal_order_parameter


def make_config():
    return {'hot': {}}


@pytest.fixture
def pd():
    return PhaseDynamics(make_config())


class TestPhaseWrapping:
    """测试相位归一化到 [0, 2π)"""

    def test_positive_wrap(self):
        theta = torch.tensor([0.0, math.pi, 3 * math.pi])
        result = theta % (2 * math.pi)
        assert (result >= 0).all()
        assert (result < 2 * math.pi).all()
        assert torch.allclose(result, torch.tensor([0.0, math.pi, math.pi]), atol=1e-6)

    def test_negative_wrap(self):
        theta = torch.tensor([-math.pi, -0.5])
        result = theta % (2 * math.pi)
        assert (result >= 0).all()
        assert (result < 2 * math.pi).all()


class TestDtParameter:
    """测试可学习时间步长 Δt"""

    def test_dt_is_learnable(self, pd):
        assert isinstance(pd.dt, torch.nn.Parameter)
        assert pd.dt.requires_grad

    def test_dt_default_value(self, pd):
        assert torch.allclose(pd.dt.data, torch.tensor(1.0))


class TestForward:
    """测试相位更新"""

    def test_shape(self, pd):
        B, H, N = 2, 4, 8
        theta = torch.rand(B, H, N) * 2 * math.pi
        omega = torch.randn(B, H, N) * 0.1

        theta_new = pd(theta, omega)
        assert theta_new.shape == (B, H, N)

    def test_output_range(self, pd):
        B, H, N = 2, 4, 8
        theta = torch.rand(B, H, N) * 2 * math.pi
        omega = torch.randn(B, H, N)

        theta_new = pd(theta, omega)
        assert (theta_new >= 0).all()
        assert (theta_new < 2 * math.pi).all()

    def test_zero_frequency_no_change(self, pd):
        """ω=0 时相位不变（Δt·0=0）"""
        theta = torch.rand(2, 4, 8) * 2 * math.pi
        omega = torch.zeros(2, 4, 8)

        theta_new = pd(theta, omega)
        assert torch.allclose(theta_new, theta % (2 * math.pi), atol=1e-6)

    def test_constant_frequency_accumulates(self, pd):
        """恒定频率下，相位线性累积"""
        theta = torch.zeros(1, 1, 1)
        omega = torch.ones(1, 1, 1) * 0.5

        theta1 = pd(theta, omega)
        theta2 = pd(theta1, omega)

        # θ2 = 0 + 0.5·Δt + 0.5·Δt = Δt
        expected = (pd.dt * 1.0) % (2 * math.pi)
        assert torch.allclose(theta2, expected.expand_as(theta2), atol=1e-5)

    def test_gradient_flow(self, pd):
        theta = torch.rand(1, 2, 4) * 2 * math.pi
        theta.requires_grad_(True)
        omega = torch.randn(1, 2, 4)

        theta_new = pd(theta, omega)
        loss = theta_new.sum()
        loss.backward()

        assert theta.grad is not None
        assert pd.dt.grad is not None

    def test_dt_gradient(self, pd):
        """Δt 应能接收梯度"""
        theta = torch.zeros(1, 1, 1)
        omega = torch.ones(1, 1, 1)
        theta_new = pd(theta, omega)
        theta_new.sum().backward()
        assert pd.dt.grad is not None


class TestCausalOrderParameter:
    """测试因果序参量"""

    def test_output_shape(self):
        theta = torch.rand(2, 4, 8) * 2 * math.pi
        r = causal_order_parameter(theta)
        assert r.shape == (2, 8)

    def test_range(self):
        theta = torch.rand(2, 4, 16) * 2 * math.pi
        r = causal_order_parameter(theta)
        assert (r >= 0).all()
        assert (r <= 1 + 1e-5).all()

    def test_synchronized_phase(self):
        """所有 token 同相时 r = 1"""
        theta = torch.ones(1, 4, 8) * 1.5  # 所有相位相同
        r = causal_order_parameter(theta)
        assert torch.allclose(r, torch.ones_like(r), atol=1e-5)

    def test_first_token_is_one(self):
        """第一个 token 的序参量恒为 1（只有一个 token）"""
        theta = torch.rand(2, 4, 1) * 2 * math.pi
        r = causal_order_parameter(theta)
        assert torch.allclose(r, torch.ones_like(r), atol=1e-5)

    def test_causal_property(self):
        """因果性：r_i 只依赖 j≤i 的相位"""
        B, H, N = 1, 1, 4
        theta = torch.rand(B, H, N) * 2 * math.pi

        r_full = causal_order_parameter(theta)

        # 修改最后一个 token 的相位
        theta_modified = theta.clone()
        theta_modified[0, 0, -1] = theta[0, 0, -1] + math.pi

        r_modified = causal_order_parameter(theta_modified)

        # 前 N-1 个 token 的 r 应该相同
        assert torch.allclose(r_full[0, :N-1], r_modified[0, :N-1], atol=1e-5)
        # 最后一个 token 的 r 应该不同
        assert not torch.allclose(r_full[0, -1], r_modified[0, -1], atol=1e-3)

    def test_desynchronized_lower(self):
        """随机相位的序参量应小于 1"""
        torch.manual_seed(42)
        theta = torch.rand(1, 1, 100) * 2 * math.pi
        r = causal_order_parameter(theta)
        # 大量随机相位时，序参量应显著小于 1
        assert r[0, -1] < 0.5

    def test_gradient_flow(self):
        theta = torch.rand(1, 2, 4) * 2 * math.pi
        theta.requires_grad_(True)
        r = causal_order_parameter(theta)
        r.sum().backward()
        assert theta.grad is not None

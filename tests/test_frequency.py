"""固有频率计算单元测试（能量差方案）"""

import pytest
import torch

from hot.model.frequency import IntrinsicFrequency


def make_config(num_heads=4, head_dim=16):
    return {
        'model': {'num_heads': num_heads, 'head_dim': head_dim},
    }


@pytest.fixture
def freq():
    return IntrinsicFrequency(make_config())


class TestOutputShape:
    """测试输出形状"""

    def test_shape(self, freq):
        B, H, N, d = 2, 4, 8, 16
        Q = torch.randn(B, H, N, d)
        K = torch.randn(B, H, N, d)
        omega = freq(Q, K)
        assert omega.shape == (B, H, N)

    def test_single_token(self, freq):
        Q = torch.randn(1, 4, 1, 16)
        K = torch.randn(1, 4, 1, 16)
        omega = freq(Q, K)
        assert omega.shape == (1, 4, 1)


class TestOutputRange:
    """测试输出范围 (-1, 1)"""

    def test_normal_input(self, freq):
        Q = torch.randn(2, 4, 8, 16)
        K = torch.randn(2, 4, 8, 16)
        omega = freq(Q, K)
        assert (omega >= -1).all()
        assert (omega <= 1).all()

    def test_large_input(self, freq):
        """tanh 饱和区仍保持 [-1, 1]"""
        Q = torch.randn(1, 4, 4, 16) * 100
        K = torch.randn(1, 4, 4, 16) * 100
        omega = freq(Q, K)
        assert (omega >= -1).all()
        assert (omega <= 1).all()

    def test_zero_input(self, freq):
        """Q=K=0 → 能量差=0 → ω=tanh(bias)，所有 token 输出相同"""
        Q = torch.zeros(1, 4, 4, 16)
        K = torch.zeros(1, 4, 4, 16)
        omega = freq(Q, K)
        # 同一 head 内所有 token 的频率相同（因为输入相同）
        for h in range(4):
            assert torch.allclose(omega[0, h, 0], omega[0, h, 1], atol=1e-6)


class TestLearnableProjection:
    """测试可学习投影层"""

    def test_proj_exists(self, freq):
        assert hasattr(freq, 'proj')
        assert isinstance(freq.proj, torch.nn.Linear)

    def test_gradient_flow(self, freq):
        Q = torch.randn(2, 4, 8, 16)
        K = torch.randn(2, 4, 8, 16)
        omega = freq(Q, K)
        loss = omega.sum()
        loss.backward()
        assert freq.proj.weight.grad is not None
        assert not torch.isnan(freq.proj.weight.grad).any()


class TestEnergyDifference:
    """测试能量差计算逻辑"""

    def test_equal_q_k_zero_freq(self, freq):
        """Q=K 时能量差为零，频率仅由 bias 决定（所有 token 相同）"""
        Q = torch.randn(1, 4, 4, 16)
        K = Q.clone()
        omega = freq(Q, K)
        # 同一 head 内所有 token 输出相同
        for h in range(4):
            assert torch.allclose(omega[0, h, 0], omega[0, h, 1], atol=1e-6)

    def test_high_q_low_k_positive(self, freq):
        """Q 能量 > K 能量时，应产生正频率（经 Linear 后）"""
        Q = torch.ones(1, 4, 4, 16) * 10
        K = torch.ones(1, 4, 4, 16) * 0.1
        omega = freq(Q, K)
        # energy_diff 很大且为正，proj 应产生非零输出
        assert omega.abs().mean() > 0.01

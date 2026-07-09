"""退火调度测试"""

import pytest
import math

from hot.training.annealing import ProgressivePhaseAnnealing


class TestLinearAnnealing:
    def test_frozen_period(self):
        a = ProgressivePhaseAnnealing(1000, 'linear')
        assert a.get_beta(0) == 0.0
        assert a.get_beta(500) == 0.0
        assert a.get_beta(999) == 0.0

    def test_annealing_period(self):
        a = ProgressivePhaseAnnealing(1000, 'linear')
        assert a.get_beta(1000) == 0.0
        assert abs(a.get_beta(1500) - 0.5) < 1e-6
        assert a.get_beta(2000) == 1.0

    def test_completed_period(self):
        a = ProgressivePhaseAnnealing(1000, 'linear')
        assert a.get_beta(2500) == 1.0
        assert a.get_beta(10000) == 1.0


class TestCosineAnnealing:
    def test_frozen_period(self):
        a = ProgressivePhaseAnnealing(1000, 'cosine')
        assert a.get_beta(0) == 0.0
        assert a.get_beta(999) == 0.0

    def test_midpoint(self):
        a = ProgressivePhaseAnnealing(1000, 'cosine')
        # 余弦退火在 t=K 时 beta=0.5（即 step=2K）
        mid = a.get_beta(2000)
        assert abs(mid - 1.0) < 1e-6  # t=1000, beta = 0.5*(1-cos(pi)) = 1.0

    def test_completed(self):
        a = ProgressivePhaseAnnealing(1000, 'cosine')
        assert a.get_beta(3000) == 1.0


class TestSigmoidAnnealing:
    def test_frozen_period(self):
        a = ProgressivePhaseAnnealing(1000, 'sigmoid')
        assert a.get_beta(0) == 0.0
        assert a.get_beta(999) == 0.0

    def test_midpoint(self):
        a = ProgressivePhaseAnnealing(1000, 'sigmoid')
        # t=1.5K 时 sigmoid(0)=0.5, 即 step=2.5K
        mid = a.get_beta(2500)
        assert abs(mid - 0.5) < 0.01

    def test_converges_to_one(self):
        a = ProgressivePhaseAnnealing(1000, 'sigmoid')
        assert a.get_beta(5000) > 0.99


class TestNoneSchedule:
    def test_always_one(self):
        a = ProgressivePhaseAnnealing(1000, 'none')
        assert a.get_beta(0) == 1.0
        assert a.get_beta(500) == 1.0
        assert a.get_beta(1000) == 1.0


class TestMonotonicity:
    def test_monotonic_increase(self):
        a = ProgressivePhaseAnnealing(1000, 'cosine')
        prev = 0.0
        for step in range(1000, 3000, 10):
            beta = a.get_beta(step)
            assert beta >= prev - 1e-9
            prev = beta


class TestRange:
    @pytest.mark.parametrize('schedule', ['linear', 'cosine', 'sigmoid', 'none'])
    def test_beta_in_range(self, schedule):
        a = ProgressivePhaseAnnealing(1000, schedule)
        for step in range(0, 5000, 100):
            beta = a.get_beta(step)
            assert 0.0 <= beta <= 1.0


class TestInvalidSchedule:
    def test_raises(self):
        a = ProgressivePhaseAnnealing(1000, 'invalid')
        with pytest.raises(ValueError):
            a.get_beta(1500)

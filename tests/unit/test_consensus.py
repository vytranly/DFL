"""Test cho src/dfl/metrics/consensus.py: khoảng cách RMS tới model trung bình (T09b; KH §3.6)."""

import numpy as np
import pytest
import torch

from dfl.metrics.consensus import consensus_distance


def test_consensus_distance_matches_hand_computed_rms() -> None:
    # w̄ = (1, 1 | 1); ‖w_0 − w̄‖² = 3, ‖w_1 − w̄‖² = 3, ‖w_2 − w̄‖² = 0 + 4 + 4 = 8, nên Ξ = sqrt(14 / 3).
    states = [
        {"a": torch.tensor([0.0, 0.0]), "b": torch.tensor([0.0])},
        {"a": torch.tensor([2.0, 0.0]), "b": torch.tensor([0.0])},
        {"a": torch.tensor([1.0, 3.0]), "b": torch.tensor([3.0])},
    ]

    assert consensus_distance(states) == pytest.approx(np.sqrt(14 / 3), rel=1e-15)


def test_consensus_distance_is_exactly_zero_for_identical_models() -> None:
    state = {"a": torch.tensor([0.1, 0.7, -3.3]), "b": torch.tensor([[1e-3]])}

    assert consensus_distance([state] * 20) == 0.0


def test_integer_tensors_do_not_count() -> None:
    float_only = [{"a": torch.tensor([0.0])}, {"a": torch.tensor([2.0])}]
    with_counters = [
        {"a": torch.tensor([0.0]), "steps": torch.tensor(5)},
        {"a": torch.tensor([2.0]), "steps": torch.tensor(500)},
    ]

    assert consensus_distance(with_counters) == consensus_distance(float_only) == 1.0


def test_consensus_distance_accumulates_in_float64() -> None:
    # Khoảng cách giữa hai số float32 liền kề quanh 10⁴ là 2⁻¹⁰. Trung bình của chúng không biểu diễn được
    # bằng float32, nên tính bằng float32 sẽ ra 0 hoặc 2⁻¹⁰·√½ thay vì đúng 2⁻¹¹.
    states = [
        {"a": torch.tensor([10_000.0], dtype=torch.float32)},
        {"a": torch.tensor([10_000.0009765625], dtype=torch.float32)},
    ]

    assert consensus_distance(states) == 2.0**-11


def test_consensus_distance_needs_at_least_one_state() -> None:
    with pytest.raises(ValueError):
        consensus_distance([])

"""Test cho src/dfl/data/heterogeneity.py: độ đo mức Non-IID (KH §3.2; ADR-005 cho lệch mục tiêu)."""

import numpy as np
import pytest

from dfl.data import heterogeneity as het


def test_js_is_zero_for_identical_and_one_bit_for_disjoint_distributions() -> None:
    p = np.array([0.5, 0.5, 0.0])
    q = np.array([0.0, 0.0, 1.0])

    assert het.js_divergence_bits(p, p) == pytest.approx(0.0, abs=1e-12)
    assert het.js_divergence_bits(p, q) == pytest.approx(1.0)


def test_js_is_symmetric_and_bounded() -> None:
    rng = np.random.default_rng(0)
    p, q = rng.dirichlet(np.ones(10)), rng.dirichlet(np.ones(10))

    js = het.js_divergence_bits(p, q)

    assert js == pytest.approx(het.js_divergence_bits(q, p))
    assert 0.0 < js < 1.0


def test_tv_is_zero_for_identical_and_one_for_disjoint_distributions() -> None:
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])

    assert het.total_variation(p, p) == 0.0
    assert het.total_variation(p, q) == pytest.approx(1.0)


def test_class_counts_builds_the_node_by_class_matrix() -> None:
    labels = np.array([0, 0, 1, 2, 2, 2])
    parts = [np.array([0, 2]), np.array([1, 3, 4, 5])]

    counts = het.class_counts(labels, parts, num_classes=3)

    np.testing.assert_array_equal(counts, [[1, 1, 0], [1, 0, 3]])


def test_summary_on_small_example_matches_hand_computation() -> None:
    # Node 0: 4 mẫu lớp 0; node 1: 2 mẫu lớp 0, 2 mẫu lớp 1; node 2: 4 mẫu lớp 1.
    labels = np.array([0] * 6 + [1] * 6)
    parts = [np.arange(0, 4), np.array([4, 5, 6, 7]), np.arange(8, 12)]

    stats = het.summarize(labels, parts, num_classes=2)

    assert stats.n_min == 4 and stats.n_max == 4
    assert stats.cv_n == 0.0
    assert stats.classes_ge_5pct_mean == pytest.approx((1 + 2 + 1) / 3)
    # Phân bố toàn cục (0.5, 0.5); TV của từng node: 0.5, 0, 0.5.
    assert stats.tv_mean == pytest.approx(1 / 3)
    # Trộn đều theo node cho đúng (0.5, 0.5), nên lệch mục tiêu bằng 0.
    assert stats.objective_shift_tv == pytest.approx(0.0, abs=1e-12)
    assert stats.node_class_counts == ((4, 0), (2, 2), (0, 4))


def test_cv_uses_population_std_of_node_sizes() -> None:
    labels = np.zeros(6, dtype=int)
    parts = [np.arange(0, 1), np.arange(1, 6)]

    stats = het.summarize(labels, parts, num_classes=1)

    sizes = np.array([1, 5])
    assert stats.cv_n == pytest.approx(sizes.std() / sizes.mean())


def test_objective_shift_is_zero_when_every_node_is_iid() -> None:
    labels = np.tile(np.arange(10), 40)
    parts = np.array_split(np.arange(len(labels)), 4)  # mỗi phần có đúng 1 mẫu mỗi lớp

    stats = het.summarize(labels, parts)

    assert stats.objective_shift_tv == pytest.approx(0.0, abs=1e-12)
    assert stats.js_mean == pytest.approx(0.0, abs=1e-12)


def test_objective_shift_uses_the_objective_weights() -> None:
    # Node lớn chỉ có lớp 0, node nhỏ chỉ có lớp 1: trộn đều 1/N lệch khỏi phân bố toàn cục,
    # còn trọng số theo kích thước (n_j/n, như FedAvg) thì không.
    labels = np.array([0] * 30 + [1] * 10)
    parts = [np.arange(0, 30), np.arange(30, 40)]

    uniform = het.summarize(labels, parts, num_classes=2)
    size_weighted = het.summarize(labels, parts, num_classes=2, objective_weights=np.array([0.75, 0.25]))

    assert uniform.objective_shift_tv == pytest.approx(0.25)
    assert size_weighted.objective_shift_tv == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("weights", [np.array([0.5]), np.array([0.7, 0.7]), np.array([1.5, -0.5])])
def test_invalid_objective_weights_raise_value_error(weights: np.ndarray) -> None:
    labels = np.array([0, 1])
    parts = [np.array([0]), np.array([1])]

    with pytest.raises(ValueError, match="objective_weights"):
        het.summarize(labels, parts, num_classes=2, objective_weights=weights)


def test_empty_part_raises_value_error() -> None:
    with pytest.raises(ValueError, match="rỗng"):
        het.summarize(np.array([0, 1]), [np.array([0, 1]), np.array([], dtype=int)], num_classes=2)

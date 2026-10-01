"""Test cho src/dfl/data/partition.py: P0–P3 theo NIID-Bench (KH §3.2; T-INV-8; H P3.1).

Dùng mảng nhãn tổng hợp từ số đếm lớp của MNIST (55k sau khi tách validation), không tải dữ liệu (ENG-8).
"""

from collections.abc import Callable

import numpy as np
import pytest

from dfl.contracts import Partitioner, PartitionResult
from dfl.data import partition as part
from dfl.data.heterogeneity import summarize
from tests.reference import niid_bench_reference as ref

LABELS = ref.mnist_train_labels()
N = 20
ALL_PARTITIONERS = [
    part.IIDPartitioner(),
    part.DirichletPartitioner(alpha=0.1),
    part.LabelKPartitioner(k=2),
    part.LabelKPartitioner(k=1),
    part.QuantityPartitioner(beta=0.5),
]


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


@pytest.mark.parametrize("partitioner", ALL_PARTITIONERS, ids=lambda p: repr(p))
def test_t_inv_8_parts_are_disjoint_cover_all_samples_and_respect_min_size(partitioner: Partitioner) -> None:
    # Act
    result = partitioner(LABELS, N, _rng(0))

    # Assert
    assert isinstance(result, PartitionResult)
    assert len(result.parts) == N
    joined = np.concatenate(result.parts)
    assert len(joined) == len(LABELS)
    np.testing.assert_array_equal(np.sort(joined), np.arange(len(LABELS)))
    assert min(len(p) for p in result.parts) >= 64


@pytest.mark.parametrize("partitioner", ALL_PARTITIONERS, ids=lambda p: repr(p))
def test_t_inv_8_same_generator_state_gives_the_same_partition(partitioner: Partitioner) -> None:
    first = partitioner(LABELS, N, _rng(3))
    second = partitioner(LABELS, N, _rng(3))
    other = partitioner(LABELS, N, _rng(4))

    assert all(np.array_equal(a, b) for a, b in zip(first.parts, second.parts, strict=True))
    assert not all(np.array_equal(a, b) for a, b in zip(first.parts, other.parts, strict=True))


@pytest.mark.parametrize("k", [1, 2, 5])
def test_t_inv_8_label_k_gives_each_node_exactly_k_labels_and_every_label_an_owner(k: int) -> None:
    result = part.LabelKPartitioner(k=k)(LABELS, N, _rng(1))

    node_labels = [set(np.unique(LABELS[p]).tolist()) for p in result.parts]
    assert all(len(s) == k for s in node_labels)
    assert set().union(*node_labels) == set(range(10))


def test_t_inv_8_label_k_with_fewer_than_ten_nodes_is_a_config_error() -> None:
    with pytest.raises(ValueError, match="N ≥ 10"):
        part.LabelKPartitioner(k=2)(LABELS, 9, _rng(0))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_t_inv_8_golden_matches_reference_exactly(seed: int) -> None:
    # Cùng trạng thái Generator thì trùng tuyệt đối với hàm tham chiếu (thứ tự tiêu thụ số ngẫu nhiên).
    cases = [
        (part.IIDPartitioner(), lambda rng: (ref.niid_bench_iid(LABELS, rng, N), 1)),
        (part.DirichletPartitioner(alpha=0.1), lambda rng: ref.niid_bench_dirichlet(LABELS, 0.1, rng, N, 64)),
        (
            part.DirichletPartitioner(alpha=0.05),
            lambda rng: ref.niid_bench_dirichlet(LABELS, 0.05, rng, N, 64),
        ),
        (part.LabelKPartitioner(k=2), lambda rng: (ref.niid_bench_label_k(LABELS, 2, rng, N), 1)),
        (
            part.QuantityPartitioner(beta=0.5),
            lambda rng: (ref.niid_bench_quantity(LABELS, 0.5, rng, N, 64), None),
        ),
    ]
    for partitioner, reference in cases:
        result = partitioner(LABELS, N, _rng(seed))
        expected_parts, attempts = reference(_rng(seed))
        assert all(np.array_equal(a, b) for a, b in zip(result.parts, expected_parts, strict=True)), (
            partitioner
        )
        if attempts is not None:
            assert result.retries == attempts - 1


def test_t_inv_8_near_iid_dirichlet_has_tiny_js() -> None:
    result = part.DirichletPartitioner(alpha=1000)(LABELS, N, _rng(0))

    assert summarize(LABELS, result.parts).js_mean < 0.01


def _mean_js(partitioner: Partitioner) -> float:
    return float(
        np.mean([summarize(LABELS, partitioner(LABELS, N, _rng(s)).parts).js_mean for s in range(3)])
    )


def test_t_inv_8_js_increases_monotonically_as_alpha_decreases() -> None:
    js = [_mean_js(part.DirichletPartitioner(alpha=a)) for a in (100, 1, 0.5, 0.1, 0.05)]

    assert all(a < b for a, b in zip(js, js[1:], strict=False)), js


def test_dirichlet_alpha_point_one_has_js_in_the_expected_range() -> None:
    assert 0.45 <= _mean_js(part.DirichletPartitioner(alpha=0.1)) <= 0.65


def test_quantity_skew_keeps_labels_iid_but_sizes_uneven() -> None:
    stats = summarize(LABELS, part.QuantityPartitioner(beta=0.5)(LABELS, N, _rng(0)).parts)

    assert stats.js_mean < 0.02
    assert stats.cv_n > 0.5


@pytest.mark.parametrize(
    "partitioner",
    [part.DirichletPartitioner(alpha=0.5, min_size=60), part.QuantityPartitioner(beta=0.5, min_size=60)],
    ids=["dirichlet", "quantity"],
)
def test_more_than_max_retries_raises_value_error(partitioner: Partitioner) -> None:
    # 4 node chia 200 mẫu thì trung bình 50 mẫu/node, nên min_size = 60 không bao giờ đạt.
    small = np.repeat(np.arange(10), 20)

    with pytest.raises(ValueError, match="1,000 lần"):
        partitioner(small, 4, _rng(0))


@pytest.mark.parametrize(
    "factory",
    [
        lambda: part.DirichletPartitioner(alpha=0.0),
        lambda: part.QuantityPartitioner(beta=-1.0),
        lambda: part.LabelKPartitioner(k=0),
        lambda: part.LabelKPartitioner(k=11),
        lambda: part.DirichletPartitioner(alpha=1.0, min_size=0),
    ],
)
def test_invalid_parameters_raise_value_error(factory: Callable[[], object]) -> None:
    with pytest.raises(ValueError):
        factory()


def test_labels_outside_zero_to_nine_raise_value_error() -> None:
    with pytest.raises(ValueError, match="nhãn"):
        part.IIDPartitioner()(np.array([0, 1, 10]), 1, _rng(0))


def test_iid_split_smaller_than_min_size_raises_value_error() -> None:
    # 100 mẫu chia 4 node thì mỗi node 25 mẫu, nhỏ hơn min_size = 64; IID không lấy mẫu lại được.
    with pytest.raises(ValueError, match="min_size"):
        part.IIDPartitioner()(np.repeat(np.arange(10), 10), 4, _rng(0))


@pytest.mark.parametrize("labels", [np.zeros((2, 2), dtype=int), np.array([0.0, 1.0])])
def test_labels_must_be_a_one_dimensional_integer_array(labels: np.ndarray) -> None:
    with pytest.raises(ValueError, match="labels"):
        part.IIDPartitioner(min_size=1)(labels, 1, _rng(0))


def test_num_nodes_must_be_positive() -> None:
    with pytest.raises(ValueError, match="num_nodes"):
        part.IIDPartitioner(min_size=1)(np.array([0, 1]), 0, _rng(0))

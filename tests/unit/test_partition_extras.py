"""Test cho hoán vị vị trí node và P2' (shards) trong src/dfl/data/partition.py (KH §3.2; SPEC QS-11)."""

import numpy as np
import pytest

from dfl.contracts import PartitionResult
from dfl.data import partition as part
from tests.reference import niid_bench_reference as ref

LABELS = ref.mnist_train_labels()
N = 20


def test_assign_positions_is_deterministic_and_a_valid_permutation() -> None:
    # Arrange
    result = part.DirichletPartitioner(alpha=0.5)(LABELS, N, np.random.default_rng(0))

    # Act
    placed = part.assign_positions(result, np.random.default_rng(7))
    again = part.assign_positions(result, np.random.default_rng(7))

    # Assert
    assert isinstance(placed, PartitionResult)
    assert placed.retries == result.retries
    assert all(np.array_equal(a, b) for a, b in zip(placed.parts, again.parts, strict=True))
    original = sorted(tuple(p.tolist()) for p in result.parts)
    assert sorted(tuple(p.tolist()) for p in placed.parts) == original


def test_assign_positions_uses_one_permutation_draw() -> None:
    result = part.LabelKPartitioner(k=2)(LABELS, N, np.random.default_rng(1))

    placed = part.assign_positions(result, np.random.default_rng(3))

    perm = np.random.default_rng(3).permutation(N)
    for position, source in enumerate(perm):
        np.testing.assert_array_equal(placed.parts[position], result.parts[source])


def test_shards_give_each_node_two_equal_sorted_shards_disjoint_and_covering() -> None:
    result = part.ShardsPartitioner()(LABELS, N, np.random.default_rng(0))

    joined = np.concatenate(result.parts)
    np.testing.assert_array_equal(np.sort(joined), np.arange(len(LABELS)))
    sizes = [len(p) for p in result.parts]
    assert max(sizes) - min(sizes) <= 2  # 2N mảnh gần bằng nhau, mỗi node 2 mảnh
    # Mỗi mảnh là một đoạn liên tiếp của mảng đã sắp theo nhãn, nên một node có 1 tới vài lớp, không phải 10.
    assert max(len(np.unique(LABELS[p])) for p in result.parts) <= 4


def test_shards_golden_follows_the_documented_draw_order() -> None:
    result = part.ShardsPartitioner()(LABELS, N, np.random.default_rng(5))

    shards = np.array_split(np.argsort(LABELS, kind="stable"), 2 * N)
    order = np.random.default_rng(5).permutation(2 * N)
    for node in range(N):
        expected = np.concatenate([shards[order[2 * node]], shards[order[2 * node + 1]]])
        np.testing.assert_array_equal(result.parts[node], expected)


def test_shards_smaller_than_min_size_raise_value_error() -> None:
    with pytest.raises(ValueError, match="min_size"):
        part.ShardsPartitioner()(np.repeat(np.arange(10), 10), 4, np.random.default_rng(0))


def test_shards_per_node_must_be_positive() -> None:
    with pytest.raises(ValueError, match="shards_per_node"):
        part.ShardsPartitioner(shards_per_node=0)

"""Test cho P5 `rare_classes` và vị trí theo bậc trong src/dfl/data/partition.py (E7; ADR-014)."""

import numpy as np
import pytest

from dfl.contracts import PartitionResult
from dfl.data import partition as part
from tests.reference import niid_bench_reference as ref

LABELS = ref.mnist_train_labels()
N = 20
K = 2


def classes_of(result: PartitionResult) -> list[set[int]]:
    return [set(LABELS[p].tolist()) for p in result.parts]


def rare_split(seed: int = 0) -> PartitionResult:
    return part.RareClassPartitioner(holders=K)(LABELS, N, np.random.default_rng(seed))


def test_rare_classes_go_only_to_the_first_k_parts() -> None:
    result = rare_split()

    classes = classes_of(result)

    assert all(set(part.RARE_CLASSES) <= c for c in classes[:K])
    assert all(c.isdisjoint(part.RARE_CLASSES) for c in classes[K:])


def test_common_classes_are_split_evenly_over_every_part() -> None:
    result = rare_split()

    common = [int(np.isin(LABELS[p], part.RARE_CLASSES, invert=True).sum()) for p in result.parts]
    rare = [int(np.isin(LABELS[p], part.RARE_CLASSES).sum()) for p in result.parts[:K]]

    assert all({0, 1, 2, 3, 4} <= c for c in classes_of(result))
    assert max(common) - min(common) <= 1
    assert max(rare) - min(rare) <= 1


def test_rare_classes_use_every_index_exactly_once() -> None:
    result = rare_split()

    everything = np.concatenate(result.parts)

    assert sorted(everything.tolist()) == list(range(len(LABELS)))
    assert result.retries == 0


def test_rare_classes_are_fixed_by_the_generator() -> None:
    first, again, other = rare_split(0), rare_split(0), rare_split(1)

    assert all(np.array_equal(a, b) for a, b in zip(first.parts, again.parts, strict=True))
    assert not all(np.array_equal(a, b) for a, b in zip(first.parts, other.parts, strict=True))


@pytest.mark.parametrize("holders", [0, N, N + 1])
def test_rare_classes_reject_holders_outside_one_to_n_minus_one(holders: int) -> None:
    with pytest.raises(ValueError, match="holders"):
        part.RareClassPartitioner(holders=holders)(LABELS, N, np.random.default_rng(0))


def test_rare_classes_still_enforce_min_size() -> None:
    with pytest.raises(ValueError, match="min_size"):
        part.RareClassPartitioner(holders=K, min_size=5_000)(LABELS, N, np.random.default_rng(0))


# ------------------------------------------------------------------------------------ vị trí theo bậc

DEGREES = np.array([2, 9, 2, 3, 12, 2, 1, 5, 2, 2, 3, 4, 2, 2, 8, 3, 2, 2, 3, 2])


def holder_nodes(placed: PartitionResult) -> set[int]:
    return {i for i, c in enumerate(classes_of(placed)) if not c.isdisjoint(part.RARE_CLASSES)}


def test_hub_places_the_rare_parts_on_the_highest_degree_nodes() -> None:
    placed = part.assign_by_degree(rare_split(), DEGREES, K, "hub", np.random.default_rng(3))

    assert holder_nodes(placed) == {4, 1}


def test_edge_places_the_rare_parts_on_the_lowest_degree_nodes() -> None:
    placed = part.assign_by_degree(rare_split(), DEGREES, K, "edge", np.random.default_rng(3))

    holders = holder_nodes(placed)

    # Bậc 1 chỉ có node 6; node thứ hai là một trong các node bậc 2, chọn bằng hoán vị phá hòa.
    assert 6 in holders and len(holders) == K
    assert max(DEGREES[sorted(holders)]) == 2


def test_degree_ties_are_broken_by_one_permutation_of_the_placement_stream() -> None:
    # Mọi bậc bằng nhau: thứ tự chỉ còn là hoán vị, nên phần j vào node perm[j].
    result = rare_split()

    placed = part.assign_by_degree(result, np.full(N, 3), K, "hub", np.random.default_rng(5))

    perm = np.random.default_rng(5).permutation(N)
    for j, node in enumerate(perm):
        np.testing.assert_array_equal(placed.parts[node], result.parts[j])


@pytest.mark.parametrize("side", ["hub", "edge"])
def test_placement_by_degree_is_a_permutation_of_the_parts(side: str) -> None:
    result = rare_split()

    placed = part.assign_by_degree(result, DEGREES, K, side, np.random.default_rng(0))

    assert placed.retries == result.retries
    assert sorted(tuple(p.tolist()) for p in placed.parts) == sorted(tuple(p.tolist()) for p in result.parts)


@pytest.mark.parametrize(
    ("degrees", "holders", "side", "message"),
    [
        (DEGREES[:-1], K, "hub", "bậc"),
        (DEGREES, K, "center", "side"),
        (DEGREES, 0, "hub", "holders"),
        (DEGREES, N, "edge", "holders"),
    ],
    ids=["degrees-length", "bad-side", "holders-zero", "holders-n"],
)
def test_placement_by_degree_rejects_bad_inputs(
    degrees: np.ndarray, holders: int, side: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        part.assign_by_degree(rare_split(), degrees, holders, side, np.random.default_rng(0))

"""Test cho src/dfl/seeding.py: luồng ngẫu nhiên tường minh, tất định, không trùng nhau (SPEC §6, ADR-009)."""

import numpy as np
import pytest
import torch

from dfl import seeding
from dfl.seeding import Stream


def _draw(rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2**63 - 1, size=8)


def test_same_seed_stream_and_key_give_the_same_numbers() -> None:
    # Arrange / Act
    first = _draw(seeding.numpy_rng(7, Stream.BATCH, 3))
    second = _draw(seeding.numpy_rng(7, Stream.BATCH, 3))

    # Assert
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ((Stream.BATCH, 0), (Stream.BATCH, 1)),
        ((Stream.GOSSIP, 5), (Stream.GOSSIP_ER, 5)),
        ((Stream.PARTITION,), (Stream.PLACEMENT,)),
        ((Stream.INIT,), (Stream.B1_BATCH,)),
    ],
)
def test_different_streams_or_keys_do_not_share_entropy(a: tuple, b: tuple) -> None:
    # Arrange
    seq_a = seeding.seed_sequence(7, *a)
    seq_b = seeding.seed_sequence(7, *b)

    # Act / Assert
    assert seq_a.spawn_key != seq_b.spawn_key
    assert not np.array_equal(seq_a.generate_state(4), seq_b.generate_state(4))


def test_different_run_seeds_give_different_numbers() -> None:
    assert not np.array_equal(
        _draw(seeding.numpy_rng(0, Stream.PARTITION)), _draw(seeding.numpy_rng(1, Stream.PARTITION))
    )


def test_data_streams_do_not_depend_on_the_run_seed() -> None:
    # Luồng dữ liệu cố định dùng DATA_SEED, nên hàm không nhận seed của lượt.
    first = _draw(seeding.data_rng(Stream.VAL_SPLIT))
    second = _draw(seeding.data_rng(Stream.VAL_SPLIT))

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, _draw(seeding.data_rng(Stream.EVAL_SUBSET)))


def test_run_and_data_streams_cannot_be_mixed() -> None:
    with pytest.raises(ValueError, match="luồng dữ liệu"):
        seeding.numpy_rng(0, Stream.VAL_SPLIT)
    with pytest.raises(ValueError, match="luồng theo seed"):
        seeding.data_rng(Stream.BATCH, 0)


@pytest.mark.parametrize("bad_seed", [-1, 1.5, True, "0"])
def test_invalid_seed_is_rejected_with_value_error(bad_seed: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        seeding.numpy_rng(bad_seed, Stream.PARTITION)


def test_negative_key_is_rejected_with_value_error() -> None:
    with pytest.raises(ValueError, match="khóa"):
        seeding.numpy_rng(0, Stream.BATCH, -1)


def test_torch_generator_is_deterministic_and_leaves_global_rng_untouched() -> None:
    # Arrange
    global_state = torch.get_rng_state()

    # Act
    first = torch.randint(0, 1000, (16,), generator=seeding.torch_generator(3, Stream.BATCH, 2))
    second = torch.randint(0, 1000, (16,), generator=seeding.torch_generator(3, Stream.BATCH, 2))
    other = torch.randint(0, 1000, (16,), generator=seeding.torch_generator(3, Stream.BATCH, 1))

    # Assert
    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    assert torch.equal(torch.get_rng_state(), global_state)


def test_child_seeds_cover_every_run_stream_and_are_deterministic() -> None:
    seeds = seeding.child_seeds(4)

    assert set(seeds) == {s.name.lower() for s in seeding.RUN_STREAMS}
    assert seeds == seeding.child_seeds(4)
    assert seeds != seeding.child_seeds(5)
    assert all(isinstance(v, int) and v >= 0 for v in seeds.values())


def test_stream_ids_are_fixed_constants() -> None:
    # Đổi số của một luồng làm đổi mọi kết quả cũ, nên các số được ghim ở đây.
    assert {s.name: int(s) for s in Stream} == {
        "PARTITION": 1,
        "PLACEMENT": 2,
        "GRAPH": 3,
        "INIT": 4,
        "BATCH": 5,
        "B1_BATCH": 6,
        "GOSSIP": 7,
        "GOSSIP_ER": 8,
        "VAL_SPLIT": 101,
        "EVAL_SUBSET": 102,
        "SMOKE_DATA": 103,
    }

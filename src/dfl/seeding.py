"""Nơi duy nhất tạo SeedSequence / Generator của dự án (SPEC §6, ADR-009, CONSTRAINTS F7).

Mỗi luồng ngẫu nhiên là SeedSequence(entropy=…, spawn_key=(ID luồng, khóa…)). ID luồng là hằng số cố định,
không dùng hash() của Python (vốn đổi theo tiến trình). Nhờ vậy không luồng nào trùng entropy với luồng khác,
và cùng seed thì cùng partition, cùng hoán vị, cùng w⁰ ở mọi topology và thuật toán (G9).

Module này import torch để dựng torch.Generator. Vì vậy các module thuần (network, data.partition,
data.heterogeneity) không import nó: chúng nhận Generator qua tham số, do sim truyền vào (contract
pure-no-torch, SPEC §3.2).

numpy.random.SeedSequence (entropy phải là số nguyên không âm; spawn_key là entropy bổ sung):
https://numpy.org/doc/2.5/reference/random/bit_generators/generated/numpy.random.SeedSequence.html
numpy.random.default_rng nhận trực tiếp một SeedSequence:
https://numpy.org/doc/2.5/reference/random/generator.html
torch.Generator.manual_seed nhận số trong [-0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff]:
https://docs.pytorch.org/docs/2.14/generated/torch.Generator.html
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
import torch


class Stream(IntEnum):
    """ID cố định của từng luồng. Đổi một số ở đây làm đổi mọi kết quả cũ, nên không bao giờ đổi."""

    # Luồng theo seed của lượt.
    PARTITION = 1
    PLACEMENT = 2
    GRAPH = 3
    INIT = 4
    BATCH = 5  # khóa: id node
    B1_BATCH = 6
    GOSSIP = 7  # khóa: round (T3)
    GOSSIP_ER = 8  # khóa: round (T3b)
    # Luồng dữ liệu cố định, giống nhau ở mọi lượt (entropy = DATA_SEED).
    VAL_SPLIT = 101
    EVAL_SUBSET = 102
    SMOKE_DATA = 103


DATA_STREAMS = frozenset({Stream.VAL_SPLIT, Stream.EVAL_SUBSET, Stream.SMOKE_DATA})
RUN_STREAMS = frozenset(set(Stream) - DATA_STREAMS)
# Entropy của luồng dữ liệu cố định: mọi lượt, ở mọi seed, dùng cùng validation và cùng ảnh đánh giá.
DATA_SEED = 0


def _check_non_negative_int(value: object, what: str) -> int:
    # bool là lớp con của int nhưng không phải một seed hợp lệ.
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{what} phải là số nguyên không âm, nhận được {value!r}")
    return value


def seed_sequence(seed: int, stream: Stream, *key: int) -> np.random.SeedSequence:
    """SeedSequence của một luồng: entropy = seed, spawn_key = (ID luồng, *khóa)."""
    _check_non_negative_int(seed, "seed")
    for k in key:
        _check_non_negative_int(k, "khóa của luồng")
    return np.random.SeedSequence(entropy=seed, spawn_key=(int(stream), *key))


def _require_run_stream(stream: Stream) -> None:
    if stream not in RUN_STREAMS:
        raise ValueError(f"{stream.name} là luồng dữ liệu cố định; dùng data_rng(), không truyền seed lượt")


def numpy_rng(seed: int, stream: Stream, *key: int) -> np.random.Generator:
    """Generator numpy của một luồng theo seed của lượt (partition, placement, đồ thị, gossip…)."""
    _require_run_stream(stream)
    return np.random.default_rng(seed_sequence(seed, stream, *key))


def data_rng(stream: Stream, *key: int) -> np.random.Generator:
    """Generator numpy của một luồng dữ liệu cố định; không phụ thuộc seed của lượt."""
    if stream not in DATA_STREAMS:
        raise ValueError(f"{stream.name} là luồng theo seed của lượt; dùng numpy_rng(seed, …)")
    return np.random.default_rng(seed_sequence(DATA_SEED, stream, *key))


def torch_generator(seed: int, stream: Stream, *key: int) -> torch.Generator:
    """torch.Generator trên CPU của một luồng theo seed của lượt (init, batch của node, batch của B1).

    Seed 64 bit lấy từ SeedSequence của luồng, nằm trong dải mà manual_seed chấp nhận. RNG toàn cục của
    torch không bị động tới (T-INV-5).
    """
    _require_run_stream(stream)
    state = seed_sequence(seed, stream, *key).generate_state(1, np.uint64)
    return torch.Generator(device="cpu").manual_seed(int(state[0]))


def child_seeds(seed: int) -> dict[str, int]:
    """Một số nguyên đại diện cho mỗi luồng theo seed của lượt, để ghi vào manifest (ManifestRecord)."""
    return {
        s.name.lower(): int(seed_sequence(seed, s).generate_state(1, np.uint64)[0])
        for s in sorted(RUN_STREAMS)
    }

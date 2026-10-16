"""Lưới E0a đúng như đã đăng ký (T14b; ADR-007; SPEC EXP-2): 24 lượt, seed 100, chỉ validation, và đủ các ô
mà analysis.select_lr cần.
"""

from collections import Counter
from pathlib import Path

from dfl.sim import sweep

ROOT = Path(__file__).resolve().parents[2]


def cells() -> list[sweep.Cell]:
    return list(sweep.load_experiment(ROOT / "configs" / "experiments" / "e0a_lr.yaml").cells)


def test_e0a_has_the_24_registered_runs() -> None:
    shape = Counter(
        (c.config.topology.kind, c.config.partition.scheme, c.config.algorithm.tau) for c in cells()
    )

    assert shape == {
        ("fully_connected", "iid", 20): 3,
        ("ring", "dirichlet", 20): 3,
        ("ring", "label_k", 20): 3,
        ("fully_connected", "iid", 1): 3,
        ("ring", "dirichlet", 1): 3,
        ("fully_connected", "iid", 50): 3,
        ("ring", "dirichlet", 50): 3,
        ("central", "iid", 20): 3,
    }


def test_every_cell_sweeps_the_three_etas_once() -> None:
    by_cell = Counter(
        (c.config.topology.kind, c.config.partition.scheme, c.config.algorithm.tau, c.config.algorithm.lr)
        for c in cells()
    )

    assert set(by_cell.values()) == {1}
    assert {lr for *_, lr in by_cell} == {0.01, 0.05, 0.1}


def test_e0a_never_touches_the_test_set_and_uses_seed_100_only() -> None:
    assert {c.config.eval.split for c in cells()} == {"validation"}
    assert {c.seed for c in cells()} == {c.config.seed for c in cells()} == {100}
    assert {c.config.algorithm.mu for c in cells()} == {0.0}  # μ tính từ η theo QS-2, không quét


def test_the_dfl_cells_keep_the_frozen_protocol_and_b1_runs_ten_epochs() -> None:
    dfl = [c.config for c in cells() if c.config.topology.kind != "central"]
    b1 = [c.config for c in cells() if c.config.topology.kind == "central"]

    assert {
        (c.num_nodes, c.local_steps, c.partition.alpha) for c in dfl if c.partition.scheme == "dirichlet"
    } == {(20, 4_000, 0.1)}
    assert {c.partition.k for c in dfl if c.partition.scheme == "label_k"} == {2}
    assert {(c.eval.subset_size, c.eval.every_steps, c.threads) for c in dfl + b1} == {(2_000, 100, 1)}
    assert {(c.num_nodes, c.local_steps) for c in b1} == {(1, 17_200)}  # 17,200 · 32 ≈ 10 epoch của 55k (F19)

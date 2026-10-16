"""Lưới E0b đúng như đã đăng ký (T14c; ADR-007; SPEC EXP-2): 43 lượt trên tập test, η chung và η của B1 lấy
từ E0a, và lượt độ nhạy η/2, 2η chỉ ở seed 0.
"""

from collections import Counter
from pathlib import Path

from dfl.sim import sweep

ROOT = Path(__file__).resolve().parents[2]
ETA = 0.1  # ADR-007, E0a 2026-10-16
B1_ETA = 0.1


def cells() -> list[sweep.Cell]:
    return list(sweep.load_experiment(ROOT / "configs" / "experiments" / "e0b_sanity.yaml").cells)


def key(c: sweep.Cell) -> tuple[object, ...]:
    cfg = c.config
    return (cfg.topology.kind, cfg.partition.scheme, cfg.local_steps, cfg.algorithm.lr)


def test_e0b_has_the_43_registered_runs_with_their_seeds() -> None:
    seeds: dict[tuple[object, ...], list[int]] = {}
    for c in cells():
        seeds.setdefault(key(c), []).append(c.seed)

    assert {k: sorted(v) for k, v in seeds.items()} == {
        ("central", "iid", 17_200, B1_ETA): [0, 1, 2],
        ("central", "iid", 4_000, ETA): [0, 1, 2],
        ("local_only", "iid", 4_000, ETA): [0, 1, 2],
        ("local_only", "dirichlet", 4_000, ETA): [0, 1, 2],
        ("server", "iid", 4_000, ETA): [0, 1, 2],
        ("ring", "iid", 4_000, ETA): [0, 1, 2, 3, 4],
        ("erdos_renyi", "iid", 4_000, ETA): [0, 1, 2, 3, 4],
        ("gossip", "iid", 4_000, ETA): [0, 1, 2, 3, 4],
        ("fully_connected", "iid", 4_000, ETA): [0, 1, 2, 3, 4],
        **{
            (kind, scheme, 4_000, lr): [0]
            for lr in (ETA / 2, 2 * ETA)
            for kind in ("ring", "fully_connected")
            for scheme in ("iid", "dirichlet")
        },
    }
    assert len(cells()) == 43


def test_every_run_is_evaluated_on_test_with_the_frozen_protocol() -> None:
    configs = [c.config for c in cells()]

    assert {c.eval.split for c in configs} == {"test"}
    assert {(c.eval.subset_size, c.eval.every_steps, c.threads) for c in configs} == {(2_000, 100, 1)}
    assert {(c.algorithm.tau, c.algorithm.mu, c.algorithm.batch_size) for c in configs} == {(20, 0.0, 32)}
    assert {c.partition.alpha for c in configs if c.partition.scheme == "dirichlet"} == {0.1}


def test_only_b3_uses_size_weighted_mixing_and_the_dfl_cells_have_20_nodes() -> None:
    mixing = Counter((c.config.topology.kind == "server", c.config.mixing) for c in cells())

    assert mixing == {(True, "size_weighted"): 3, (False, "metropolis_hastings"): 40}
    assert {c.config.num_nodes for c in cells() if c.config.topology.kind != "central"} == {20}

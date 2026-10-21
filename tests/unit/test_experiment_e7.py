"""Lưới E7 đúng như đã đăng ký (T30e; ADR-014; thẻ docs/experiments/E7.md): BA m = 2 × rare_classes K = 2 ×
{hub, edge, random} × seed {0, 1, 2} = 9 lượt, cùng giao thức cố định của lưới lõi.
"""

import dataclasses
from collections import Counter
from pathlib import Path

from dfl.sim import sweep

ROOT = Path(__file__).resolve().parents[2]
ETA = 0.1  # ADR-007, E0a 2026-10-16


def cells() -> list[sweep.Cell]:
    return list(sweep.load_experiment(ROOT / "configs" / "experiments" / "e7_centrality.yaml").cells)


def test_e7_has_the_9_registered_runs_with_their_seeds() -> None:
    seeds: dict[str, list[int]] = {}
    for c in cells():
        seeds.setdefault(c.config.partition.placement, []).append(c.seed)

    assert {k: sorted(v) for k, v in seeds.items()} == {p: [0, 1, 2] for p in ("hub", "edge", "random")}
    assert len(cells()) == 9


def test_every_e7_run_uses_ba_m2_with_two_rare_class_holders_out_of_20_nodes() -> None:
    configs = [c.config for c in cells()]

    assert {(c.topology.kind, c.topology.m) for c in configs} == {("barabasi_albert", 2)}
    assert {(c.partition.scheme, c.partition.holders, c.num_nodes) for c in configs} == {
        ("rare_classes", 2, 20)
    }
    assert {c.mixing for c in configs} == {"metropolis_hastings"}


def test_every_e7_run_follows_the_frozen_protocol_of_the_core_grid() -> None:
    configs = [c.config for c in cells()]

    assert {c.eval.split for c in configs} == {"test"}
    assert {(c.eval.subset_size, c.eval.every_steps, c.threads) for c in configs} == {(2_000, 100, 1)}
    assert {(c.local_steps, c.algorithm.tau, c.algorithm.mu, c.algorithm.batch_size) for c in configs} == {
        (4_000, 20, 0.0, 32)
    }
    assert {c.algorithm.lr for c in configs} == {ETA}
    assert {(c.dataset, c.model, c.runtime) for c in configs} == {("mnist", "smallcnn", "inprocess")}


def test_the_three_placements_of_a_seed_differ_only_in_placement() -> None:
    by_seed: dict[int, list[sweep.Cell]] = {}
    for c in cells():
        by_seed.setdefault(c.seed, []).append(c)

    for group in by_seed.values():
        as_random = {
            dataclasses.replace(
                c.config, partition=dataclasses.replace(c.config.partition, placement="random")
            )
            for c in group
        }
        assert len(as_random) == 1
    assert Counter(len(group) for group in by_seed.values()) == {3: 3}

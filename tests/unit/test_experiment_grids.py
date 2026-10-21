"""Lưới của E1–E4, E8 đúng như đã đăng ký (SPEC EXP-2, EXP-3; ADR-007, ADR-011; thẻ docs/experiments/E<k>.md):
số lượt, ô, seed, η chung, μ theo QS-2 và giao thức đánh giá đã chốt.
"""

from collections import Counter
from pathlib import Path

import pytest

from dfl.config import RunConfig
from dfl.sim import sweep

ROOT = Path(__file__).resolve().parents[2]
ETA = 0.1  # ADR-007, E0a
MU_WEAK, MU_STRONG = 0.05, 0.5  # QS-2: μ danh nghĩa 0.1 và 1 nhân 0.05/η
CORE_SEEDS, ABLATION_SEEDS = [0, 1, 2, 3, 4], [0, 1, 2]
DIRICHLET = (100.0, 1.0, 0.5, 0.1, 0.05)


def configs(name: str) -> list[tuple[RunConfig, int]]:
    cells = sweep.load_experiment(ROOT / "configs" / "experiments" / f"{name}.yaml").cells
    return [(c.config, c.seed) for c in cells]


def level(cfg: RunConfig) -> str:
    p = cfg.partition
    return {"iid": "iid", "dirichlet": f"a{p.alpha}", "label_k": f"c{p.k}", "quantity": f"b{p.beta}"}[
        p.scheme
    ]


GETTERS = {
    "kind": lambda c: c.topology.kind,
    "level": level,
    "tau": lambda c: c.algorithm.tau,
    "mu": lambda c: c.algorithm.mu,
    "mixing": lambda c: c.mixing,
    "model": lambda c: c.model,
}


def seeds_by(name: str, *fields: str) -> dict[tuple[object, ...], list[int]]:
    seeds: dict[tuple[object, ...], list[int]] = {}
    for cfg, seed in configs(name):
        seeds.setdefault(tuple(GETTERS[f](cfg) for f in fields), []).append(seed)
    return {k: sorted(v) for k, v in seeds.items()}


@pytest.mark.parametrize(
    ("name", "runs"),
    [
        ("e1_noniid", 70),
        ("e2_topology", 30),
        ("e3a_algorithms", 24),
        ("e4a_quantity", 6),
        ("e8_model_complexity", 72),
        ("e1c_objective", 15),
        ("e2b_gossip", 9),
        ("e3b_tau_mu", 42),
        ("e4_quantity", 15),
    ],
)
def test_every_group_has_its_registered_run_count_and_the_frozen_protocol(name: str, runs: int) -> None:
    cells = configs(name)

    assert len(cells) == runs
    assert len({(cfg, seed) for cfg, seed in cells}) == runs  # không ô nào lặp
    assert {cfg.algorithm.lr for cfg, _ in cells} == {ETA}
    protocol = {
        (c.algorithm.batch_size, c.eval.split, c.eval.subset_size, c.eval.every_steps, c.threads)
        for c, _ in cells
    }
    assert protocol == {(32, "test", 2_000, 100, 1)}


def test_e1_crosses_ring_and_fc_with_every_non_iid_level_over_five_seeds() -> None:
    levels = [f"a{a}" for a in DIRICHLET] + ["c2", "c1"]

    assert seeds_by("e1_noniid", "kind", "level") == {
        (kind, lv): CORE_SEEDS for kind in ("ring", "fully_connected") for lv in levels
    }
    shape = {(c.algorithm.tau, c.algorithm.mu, c.num_nodes, c.local_steps) for c, _ in configs("e1_noniid")}
    assert shape == {(20, 0.0, 20, 4_000)}


def test_e2_crosses_er_and_gossip_with_three_levels_over_five_seeds() -> None:
    assert seeds_by("e2_topology", "kind", "level") == {
        (kind, lv): CORE_SEEDS for kind in ("erdos_renyi", "gossip") for lv in ("a1.0", "a0.1", "c2")
    }


def test_e3a_runs_d_psgd_and_strong_dfedprox_on_three_seeds() -> None:
    assert seeds_by("e3a_algorithms", "tau", "mu", "kind", "level") == {
        (tau, mu, kind, lv): ABLATION_SEEDS
        for tau, mu in ((1, 0.0), (20, MU_STRONG))
        for kind in ("ring", "fully_connected")
        for lv in ("iid", "a0.1")
    }


def test_e4a_is_quantity_skew_on_er_and_fc() -> None:
    assert seeds_by("e4a_quantity", "kind", "level", "mixing") == {
        ("erdos_renyi", "b0.5", "metropolis_hastings"): ABLATION_SEEDS,
        ("fully_connected", "b0.5", "metropolis_hastings"): ABLATION_SEEDS,
    }


def test_e8_runs_four_models_on_eight_processes_for_a_thousand_steps() -> None:
    assert seeds_by("e8_model_complexity", "model", "kind", "level") == {
        (model, kind, lv): ABLATION_SEEDS
        for model in ("logreg", "smallcnn", "mlp2nn", "cnn_mcmahan")
        for kind in ("ring", "gossip", "fully_connected")
        for lv in ("iid", "a0.1")
    }
    shape = {
        (c.num_nodes, c.local_steps, c.algorithm.tau, c.runtime, c.deploy)
        for c, _ in configs("e8_model_complexity")
    }
    assert shape == {(8, 1_000, 20, "processes", "local")}


def test_e1c_is_the_server_path_with_size_weights_on_the_dirichlet_ladder() -> None:
    assert seeds_by("e1c_objective", "kind", "mixing", "level") == {
        ("server", "size_weighted", f"a{a}"): ABLATION_SEEDS for a in DIRICHLET
    }


def test_e2b_is_gossip_restricted_to_er_edges() -> None:
    assert seeds_by("e2b_gossip", "kind", "level") == {
        ("er_gossip", lv): ABLATION_SEEDS for lv in ("iid", "a0.1", "c2")
    }


def test_e3b_scans_tau_and_mu_with_qs2_strengths_and_keeps_the_step_budget() -> None:
    expected = {
        (tau, mu, kind, lv): ABLATION_SEEDS
        for tau, mu in ((5, 0.0), (50, 0.0), (20, MU_WEAK))
        for kind in ("ring", "fully_connected")
        for lv in ("iid", "a0.1")
    }
    expected |= {(50, MU_STRONG, kind, "a0.1"): ABLATION_SEEDS for kind in ("ring", "fully_connected")}

    assert seeds_by("e3b_tau_mu", "tau", "mu", "kind", "level") == expected
    assert {c.local_steps for c, _ in configs("e3b_tau_mu")} == {4_000}


def test_e4_varies_mixing_weights_on_er_and_size_weights_on_the_server_path() -> None:
    mixing = Counter((c.topology.kind, c.mixing, level(c)) for c, _ in configs("e4_quantity"))

    assert mixing == {
        ("erdos_renyi", "uniform", "b0.5"): 3,
        ("erdos_renyi", "size_corrected", "b0.5"): 3,
        ("erdos_renyi", "uniform", "a0.1"): 3,
        ("erdos_renyi", "size_corrected", "a0.1"): 3,
        ("server", "size_weighted", "b0.5"): 3,
    }

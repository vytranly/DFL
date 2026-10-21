"""Lưới E9 đúng như đã đăng ký (T31g; ADR-015; thẻ docs/experiments/E9.md): {ring, lattice, ER, BA, FC} ×
{IID, α = 0.1} × seed {0, 1, 2} = 30 lượt, cùng giao thức cố định của lưới lõi.
"""

from pathlib import Path

from dfl.sim import sweep

ROOT = Path(__file__).resolve().parents[2]
ETA = 0.1  # ADR-007, E0a 2026-10-16
KINDS = ("ring", "lattice", "erdos_renyi", "barabasi_albert", "fully_connected")


def cells() -> list[sweep.Cell]:
    return list(sweep.load_experiment(ROOT / "configs" / "experiments" / "e9_topology_shapes.yaml").cells)


def test_e9_has_the_30_registered_runs_with_their_seeds() -> None:
    seeds: dict[tuple[str, str], list[int]] = {}
    for c in cells():
        seeds.setdefault((c.config.topology.kind, c.config.partition.scheme), []).append(c.seed)

    assert {k: sorted(v) for k, v in seeds.items()} == {
        (kind, scheme): [0, 1, 2] for kind in KINDS for scheme in ("iid", "dirichlet")
    }
    assert len(cells()) == 30


def test_e9_uses_the_registered_graph_parameters_and_non_iid_level() -> None:
    configs = [c.config for c in cells()]

    assert {c.topology.p for c in configs if c.topology.kind == "erdos_renyi"} == {0.3}
    assert {c.topology.m for c in configs if c.topology.kind == "barabasi_albert"} == {2}
    assert {c.partition.alpha for c in configs if c.partition.scheme == "dirichlet"} == {0.1}
    assert {c.num_nodes for c in configs} == {20} and {c.mixing for c in configs} == {"metropolis_hastings"}


def test_every_e9_run_follows_the_frozen_protocol_of_the_core_grid() -> None:
    configs = [c.config for c in cells()]

    assert {c.eval.split for c in configs} == {"test"}
    assert {(c.eval.subset_size, c.eval.every_steps, c.threads) for c in configs} == {(2_000, 100, 1)}
    assert {(c.local_steps, c.algorithm.tau, c.algorithm.mu, c.algorithm.batch_size) for c in configs} == {
        (4_000, 20, 0.0, 32)
    }
    assert {c.algorithm.lr for c in configs} == {ETA}
    assert {(c.dataset, c.model, c.runtime) for c in configs} == {("mnist", "smallcnn", "inprocess")}

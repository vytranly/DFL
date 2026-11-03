"""E7 qua sim.run: lớp hiếm đặt ở hub hay ở node biên của đồ thị BA, và đồ thị trong manifest (ADR-014).

Dùng dữ liệu tổng hợp của profile smoke (1,800 ảnh train) với N = 8, nên phần lớp chung của mỗi node
(khoảng 110 ảnh) vẫn trên min_size = 64.
"""

import dataclasses
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from dfl.config import EvalConfig, PartitionConfig, RunConfig, TopologyConfig, load_config
from dfl.data import datasets
from dfl.data.partition import RARE_CLASSES
from dfl.io import manifest
from dfl.sim import run

ROOT = Path(__file__).resolve().parents[2]
SMOKE = load_config(ROOT / "configs" / "profiles" / "smoke.yaml")
HOLDERS = 2


def e7(placement: str, seed: int = 0) -> RunConfig:
    return dataclasses.replace(
        SMOKE,
        num_nodes=8,
        seed=seed,
        topology=TopologyConfig(kind="barabasi_albert", m=2),
        partition=PartitionConfig(scheme="rare_classes", holders=HOLDERS, placement=placement),
    )


@pytest.fixture(scope="module")
def data() -> datasets.Datasets:
    return datasets.load("synthetic")


def holders_and_degrees(assembly: run.Assembly) -> tuple[set[int], dict[int, int]]:
    counts = np.array(assembly.heterogeneity.node_class_counts)
    holders = {int(i) for i in np.flatnonzero(counts[:, list(RARE_CLASSES)].sum(axis=1) > 0)}
    assert assembly.schedule is not None
    return holders, dict(assembly.schedule.graph_at(0).degree())


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_hub_puts_the_rare_classes_on_nodes_of_degree_no_lower_than_any_other(
    data: datasets.Datasets, seed: int
) -> None:
    holders, degree = holders_and_degrees(run.assemble(e7("hub", seed), data))

    others = set(degree) - holders
    assert len(holders) == HOLDERS
    assert min(degree[i] for i in holders) >= max(degree[i] for i in others)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_edge_puts_the_rare_classes_on_nodes_of_degree_no_higher_than_any_other(
    data: datasets.Datasets, seed: int
) -> None:
    holders, degree = holders_and_degrees(run.assemble(e7("edge", seed), data))

    others = set(degree) - holders
    assert len(holders) == HOLDERS
    assert max(degree[i] for i in holders) <= min(degree[i] for i in others)


def test_the_three_placements_of_a_seed_share_the_graph_and_the_parts(data: datasets.Datasets) -> None:
    # G9: hub, edge và random chỉ khác chỗ đặt các phần.
    assemblies = [run.assemble(e7(placement), data) for placement in ("hub", "edge", "random")]

    assert all(a.graph_edges is not None for a in assemblies)
    graphs = {tuple(map(tuple, a.graph_edges)) for a in assemblies}
    parts = {tuple(sorted(tuple(p.tolist()) for p in a.partition.parts)) for a in assemblies}

    assert len(graphs) == 1 and len(parts) == 1
    assert len({tuple(a.partition.parts[0].tolist()) for a in assemblies}) > 1


@pytest.mark.parametrize(
    ("topology", "expected"),
    [
        (TopologyConfig(kind="ring"), [[0, 1], [0, 3], [1, 2], [2, 3]]),
        (TopologyConfig(kind="local_only"), []),
        (TopologyConfig(kind="gossip"), None),
        (TopologyConfig(kind="er_gossip"), None),
    ],
    ids=["ring", "local-only", "gossip", "er-gossip"],
)
def test_assembly_lists_the_edges_of_static_graphs_only(
    data: datasets.Datasets, topology: TopologyConfig, expected: list[list[int]] | None
) -> None:
    assembly = run.assemble(dataclasses.replace(SMOKE, topology=topology), data)

    assert assembly.graph_edges == expected


def test_a_run_writes_its_ba_graph_into_the_manifest(tmp_path: Path) -> None:
    config = dataclasses.replace(e7("hub"), local_steps=4, eval=EvalConfig(subset_size=100, every_steps=None))
    provenance = dataclasses.replace(manifest.current_provenance(), dirty=False)

    attempt = run.run(config, out=tmp_path, provenance=provenance)

    record = json.loads((attempt / "manifest.json").read_text(encoding="utf-8"))
    graph = nx.Graph(record["graph_edges"])
    assert graph.number_of_nodes() == 8 and graph.number_of_edges() == 2 * (8 - 2)
    assert record["config"]["partition"]["placement"] == "hub"


def test_the_ba_graphs_of_e9_have_the_hub_and_leaf_counts_its_card_registered() -> None:
    # Thẻ docs/experiments/E9.md: seed 0, 1, 2 có 3, 3, 4 hub và 11, 12, 11 leaf (N = 20, m = 2); đổi luồng
    # graph hay networkx mà đổi số này thì phải sửa thẻ trước khi chạy E9.
    from dfl import seeding
    from dfl.network import topology
    from dfl.seeding import Stream

    counts = []
    for seed in (0, 1, 2):
        degrees = [
            d
            for _, d in topology.barabasi_albert(20, 2, seeding.numpy_rng(seed, Stream.GRAPH))
            .graph_at(0)
            .degree()
        ]
        counts.append((sum(d >= 6 for d in degrees), sum(1 <= d <= 2 for d in degrees)))

    assert counts == [(3, 11), (3, 12), (4, 11)]

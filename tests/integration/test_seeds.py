"""T-INV-5 qua một lượt chạy thật trên profile smoke (T11b; SPEC §6, §9.2; G9; INV-6).

Cùng seed và `code_version` thì `rounds.jsonl` trùng từng byte; khác seed thì khác cách chia và khác w⁰ ở mọi
tensor; cùng seed thì cùng cách chia, cùng vị trí node, cùng w⁰ và cùng luồng batch ở mọi topology và thuật
toán; RNG toàn cục của torch không đổi sau một lượt. Mỗi luồng theo seed lấy đúng seed của cấu hình, và P_i
(`node_class_counts`) xếp theo đúng id node (REVIEW-C3-P3.7 P0-2, P0-4).
"""

import dataclasses
import json
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch

from dfl import seeding
from dfl.config import AlgorithmConfig, EvalConfig, PartitionConfig, RunConfig, TopologyConfig, load_config
from dfl.data import datasets
from dfl.data.partition import IIDPartitioner, assign_positions
from dfl.learning.models import build_model
from dfl.network import topology
from dfl.seeding import Stream
from dfl.sim import run

SMOKE = load_config(Path(__file__).resolve().parents[2] / "configs" / "profiles" / "smoke.yaml")


@pytest.fixture(scope="module")
def data() -> datasets.Datasets:
    return datasets.load("synthetic")


def w0(setup: run.RunSetup) -> list[dict[str, torch.Tensor]]:
    return [node.model.state_dict() for node in setup.nodes]


def parts_multiset(parts: tuple[np.ndarray, ...]) -> list[tuple[int, ...]]:
    return sorted(tuple(sorted(part.tolist())) for part in parts)


# Lượt chạy thật trong test: 3 round, tập con đánh giá 100 ảnh (đủ 10 lớp), để test nhanh.
SHORT = dataclasses.replace(SMOKE, local_steps=6, eval=EvalConfig(subset_size=100, every_steps=None))


def test_t_inv_5_same_seed_and_code_give_byte_identical_rounds_files(tmp_path: Path) -> None:
    first = run.run(SHORT, out=tmp_path / "a")
    second = run.run(SHORT, out=tmp_path / "b")

    assert (first / "rounds.jsonl").read_bytes() == (second / "rounds.jsonl").read_bytes()


def test_t_inv_5_a_run_leaves_the_global_torch_rng_untouched(tmp_path: Path) -> None:
    before = torch.get_rng_state()

    run.run(SHORT, out=tmp_path)

    assert torch.equal(torch.get_rng_state(), before)


def test_t_inv_5_another_seed_changes_the_partition_and_every_w0_tensor(data: datasets.Datasets) -> None:
    a = run.build(SMOKE, data)
    b = run.build(dataclasses.replace(SMOKE, seed=1), data)

    # So tập các phần, không so theo vị trí: riêng hoán vị vị trí node đã làm các phần nằm ở chỗ khác.
    assert parts_multiset(a.partition.parts) != parts_multiset(b.partition.parts)
    first, second = w0(a)[0], w0(b)[0]
    assert all(not torch.equal(first[name], second[name]) for name in first)


def test_t_inv_5_nodes_get_the_partition_permuted_by_the_placement_stream(data: datasets.Datasets) -> None:
    labels = data.train.y.numpy()
    raw = IIDPartitioner(SMOKE.partition.min_size)(labels, 4, seeding.numpy_rng(SMOKE.seed, Stream.PARTITION))
    expected = assign_positions(raw, seeding.numpy_rng(SMOKE.seed, Stream.PLACEMENT))

    setup = run.build(SMOKE, data)

    assert all(np.array_equal(x, y) for x, y in zip(setup.partition.parts, expected.parts, strict=True))
    assert not all(np.array_equal(x, y) for x, y in zip(raw.parts, expected.parts, strict=True))


def test_every_node_starts_from_the_same_w0_in_its_own_memory(data: datasets.Datasets) -> None:
    # Giả định A3 (KH §3.1): w⁰ chung theo seed; mỗi node vẫn có bộ nhớ tham số riêng (engine kiểm), và rút
    # minibatch từ luồng batch của chính nó, spawn_key (ID_BATCH, i) (SPEC §6).
    setup = run.build(SMOKE, data)
    states = w0(setup)

    for node in setup.nodes:
        expected = seeding.torch_generator(SMOKE.seed, Stream.BATCH, node.node_id)
        assert torch.equal(node.generator.get_state(), expected.get_state())

    for state in states[1:]:
        assert all(torch.equal(state[name], states[0][name]) for name in state)
        assert all(state[name].data_ptr() != states[0][name].data_ptr() for name in state)


VARIANTS: dict[str, Callable[[RunConfig], RunConfig]] = {
    "fully-connected": lambda c: dataclasses.replace(c, topology=TopologyConfig(kind="fully_connected")),
    "erdos-renyi": lambda c: dataclasses.replace(c, topology=TopologyConfig(kind="erdos_renyi", p=0.6)),
    "gossip": lambda c: dataclasses.replace(c, topology=TopologyConfig(kind="gossip")),
    "er-gossip": lambda c: dataclasses.replace(c, topology=TopologyConfig(kind="er_gossip", p=0.6)),
    "star": lambda c: dataclasses.replace(c, topology=TopologyConfig(kind="star")),
    "local-only": lambda c: dataclasses.replace(c, topology=TopologyConfig(kind="local_only")),
    "server": lambda c: dataclasses.replace(
        c, topology=TopologyConfig(kind="server"), mixing="size_weighted"
    ),
    "uniform": lambda c: dataclasses.replace(c, mixing="uniform"),
    "size-corrected": lambda c: dataclasses.replace(c, mixing="size_corrected"),
    "prox": lambda c: dataclasses.replace(c, algorithm=AlgorithmConfig(tau=2, mu=0.1)),
    "tau-1": lambda c: dataclasses.replace(c, algorithm=AlgorithmConfig(tau=1)),
}


@pytest.mark.parametrize("variant", list(VARIANTS), ids=list(VARIANTS))
def test_t_inv_5_same_seed_shares_partition_placement_w0_and_batch_streams_across_variants(
    data: datasets.Datasets, variant: str
) -> None:
    base = run.build(SMOKE, data)
    other = run.build(VARIANTS[variant](SMOKE), data)

    assert all(np.array_equal(x, y) for x, y in zip(base.partition.parts, other.partition.parts, strict=True))
    for mine, theirs in zip(w0(base), w0(other), strict=True):
        assert all(torch.equal(mine[name], theirs[name]) for name in mine)
    for mine, theirs in zip(base.nodes, other.nodes, strict=True):
        assert torch.equal(mine.generator.get_state(), theirs.generator.get_state())
        assert torch.equal(mine.data.x, theirs.data.x) and torch.equal(mine.data.y, theirs.data.y)


SCHEMES = {
    "iid": PartitionConfig(scheme="iid"),
    "dirichlet": PartitionConfig(scheme="dirichlet", alpha=0.5),
    "quantity": PartitionConfig(scheme="quantity", beta=2.0),
    "label-k": PartitionConfig(scheme="label_k", k=2),
}


@pytest.mark.parametrize("scheme", list(SCHEMES), ids=list(SCHEMES))
def test_t_inv_5_every_partition_scheme_is_fixed_by_the_seed(data: datasets.Datasets, scheme: str) -> None:
    # #C = k cần N ≥ 10 (G23); 1,800 ảnh chia cho 10 node vẫn trên min_size = 64.
    config = dataclasses.replace(SMOKE, partition=SCHEMES[scheme], num_nodes=10 if scheme == "label-k" else 4)

    first = run.build(config, data).partition
    again = run.build(config, data).partition

    assert all(np.array_equal(x, y) for x, y in zip(first.parts, again.parts, strict=True))
    assert sorted(np.concatenate(first.parts).tolist()) == list(range(len(data.train.y)))


def test_shards_are_rejected_outside_describe(data: datasets.Datasets) -> None:
    config = dataclasses.replace(SMOKE, partition=PartitionConfig(scheme="shards"))

    with pytest.raises(ValueError, match="QS-11"):
        run.build(config, data)


# ------------------------------------------------------------- dây seed của từng luồng (REVIEW-C3-P3.7 P0-2)

GRAPH_P = 0.6
ROUNDS = range(8)


def edges(graph: nx.Graph) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((min(u, v), max(u, v)) for u, v in graph.edges))


def generator_state(generator: torch.Generator) -> bytes:
    return generator.get_state().numpy().tobytes()


def drawn_by_the_run(seed: int, data: datasets.Datasets) -> dict[str, object]:
    """Những gì sim.run rút từ các luồng theo seed khi lắp lượt smoke có `seed` (cách chia, vị trí node, w⁰,
    batch của node và của B1, đồ thị ER, matching của gossip và ER-gossip ở 8 round đầu)."""

    def assembled(**changes: object) -> run.Assembly:
        return run.assemble(dataclasses.replace(SMOKE, seed=seed, **changes), data)

    ring = assembled()
    er = assembled(topology=TopologyConfig(kind="erdos_renyi", p=GRAPH_P)).schedule
    gossip = assembled(topology=TopologyConfig(kind="gossip")).schedule
    er_gossip = assembled(topology=TopologyConfig(kind="er_gossip", p=GRAPH_P)).schedule
    central = assembled(num_nodes=1, topology=TopologyConfig(kind="central"))
    assert er is not None and gossip is not None and er_gossip is not None
    nodes = [ring.node(i) for i in range(SMOKE.num_nodes)]
    return {
        "parts": tuple(tuple(part.tolist()) for part in ring.partition.parts),
        "w0": tuple(t.numpy().tobytes() for t in nodes[0].model.state_dict().values()),
        "batch": tuple(generator_state(node.generator) for node in nodes),
        "b1_batch": generator_state(central.node(0).generator),
        "graph": edges(er.graph_at(0)),
        "gossip": tuple(edges(gossip.graph_at(t)) for t in ROUNDS),
        "er_gossip": tuple(edges(er_gossip.graph_at(t)) for t in ROUNDS),
    }


def drawn_from_the_streams(seeds: Mapping[Stream, int], data: datasets.Datasets) -> dict[str, object]:
    """Cùng các giá trị, dựng thẳng từ các luồng theo SPEC §6, không qua sim.run; `seeds` là seed của từng
    luồng."""
    n = SMOKE.num_nodes
    raw = IIDPartitioner(SMOKE.partition.min_size)(
        data.train.y.numpy(), n, seeding.numpy_rng(seeds[Stream.PARTITION], Stream.PARTITION)
    )
    placed = assign_positions(raw, seeding.numpy_rng(seeds[Stream.PLACEMENT], Stream.PLACEMENT))
    w0 = build_model(SMOKE.model, seeding.torch_generator(seeds[Stream.INIT], Stream.INIT)).state_dict()
    base = topology.erdos_renyi(n, GRAPH_P, seeding.numpy_rng(seeds[Stream.GRAPH], Stream.GRAPH))
    gossip = topology.gossip(n, partial(seeding.numpy_rng, seeds[Stream.GOSSIP], Stream.GOSSIP))
    er_gossip = topology.er_gossip(
        base.graph, partial(seeding.numpy_rng, seeds[Stream.GOSSIP_ER], Stream.GOSSIP_ER)
    )
    return {
        "parts": tuple(tuple(part.tolist()) for part in placed.parts),
        "w0": tuple(t.numpy().tobytes() for t in w0.values()),
        "batch": tuple(
            generator_state(seeding.torch_generator(seeds[Stream.BATCH], Stream.BATCH, i)) for i in range(n)
        ),
        "b1_batch": generator_state(seeding.torch_generator(seeds[Stream.B1_BATCH], Stream.B1_BATCH)),
        "graph": edges(base.graph_at(0)),
        "gossip": tuple(edges(gossip.graph_at(t)) for t in ROUNDS),
        "er_gossip": tuple(edges(er_gossip.graph_at(t)) for t in ROUNDS),
    }


@pytest.mark.parametrize("seed", [0, 3], ids=["seed=0", "seed=3"])
def test_t_inv_5_every_seeded_stream_of_a_run_follows_config_seed(data: datasets.Datasets, seed: int) -> None:
    # Ở seed 0, một luồng bị ghim vào seed 0 vẫn cho đúng giá trị; ca seed 3 mới bắt được lỗi đó.
    expected = drawn_from_the_streams(dict.fromkeys(seeding.RUN_STREAMS, seed), data)

    assert drawn_by_the_run(seed, data) == expected


@pytest.mark.parametrize("stream", sorted(seeding.RUN_STREAMS), ids=lambda stream: stream.name)
def test_seed_3_tells_apart_a_run_that_pins_any_one_stream_to_seed_0(
    data: datasets.Datasets, stream: Stream
) -> None:
    # Ca seed 3 ở test trên chỉ có nghĩa nếu ghim riêng một luồng vào seed 0 làm đổi những gì lượt rút ra.
    honest = drawn_from_the_streams(dict.fromkeys(seeding.RUN_STREAMS, 3), data)

    pinned = drawn_from_the_streams({**dict.fromkeys(seeding.RUN_STREAMS, 3), stream: 0}, data)

    assert pinned != honest


def test_node_class_counts_follow_node_ids_so_acc_local_uses_each_nodes_own_labels(
    data: datasets.Datasets, tmp_path: Path
) -> None:
    # REVIEW-C3-P3.7 P0-4: với IID mọi hàng P_i như nhau, nên dùng Dirichlet α = 0.1 để mỗi node có P_i riêng.
    config = dataclasses.replace(SHORT, partition=PartitionConfig(scheme="dirichlet", alpha=0.1, min_size=8))

    setup = run.build(config, data)
    attempt = run.run(config, out=tmp_path)

    expected = [np.bincount(node.data.y.numpy(), minlength=10).tolist() for node in setup.nodes]
    assert len({tuple(row) for row in expected}) == len(expected)
    assert [list(row) for row in setup.heterogeneity.node_class_counts] == expected
    manifest = json.loads((attempt / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["node_class_counts"] == expected

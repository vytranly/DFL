"""Test cho src/dfl/network/topology.py: T0–T4, T3b và lịch W = I (KH §3.3; ADR-009; INV-4; P3.2)."""

from collections.abc import Callable

import networkx as nx
import numpy as np
import pytest

from dfl import seeding
from dfl.contracts import TopologySchedule
from dfl.network import topology as topo
from dfl.seeding import Stream


def _gossip_rng(seed: int, stream: Stream = Stream.GOSSIP) -> Callable[[int], np.random.Generator]:
    # Đúng cách sim sẽ dựng rng_for: một SeedSequence riêng cho mỗi (seed, luồng, round) (ADR-009).
    return lambda r: seeding.numpy_rng(seed, stream, r)


def _er(seed: int, n: int = 20) -> nx.Graph:
    return topo.erdos_renyi(n, 0.3, seeding.numpy_rng(seed, Stream.GRAPH)).graph_at(0)


def test_ring_has_degree_two_and_is_connected() -> None:
    schedule = topo.ring(20)

    graph = schedule.graph_at(0)

    assert isinstance(schedule, TopologySchedule) and schedule.is_static
    assert schedule.num_nodes == 20
    assert {d for _, d in graph.degree()} == {2}
    assert nx.is_connected(graph)


def test_fully_connected_has_degree_n_minus_one() -> None:
    assert {d for _, d in topo.fully_connected(20).graph_at(5).degree()} == {19}


def test_star_has_one_hub() -> None:
    degrees = sorted(d for _, d in topo.star(20).graph_at(0).degree())

    assert degrees == [1] * 19 + [19]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_erdos_renyi_is_connected_and_deterministic_per_seed(seed: int) -> None:
    graph = _er(seed)

    assert nx.is_connected(graph)
    assert sorted(graph.edges()) == sorted(_er(seed).edges())


def test_erdos_renyi_differs_across_seeds() -> None:
    assert sorted(_er(0).edges()) != sorted(_er(1).edges())


def test_static_schedule_returns_a_frozen_graph() -> None:
    graph = topo.ring(5).graph_at(0)

    with pytest.raises(nx.NetworkXError):
        graph.add_edge(0, 2)


def test_local_only_has_no_edges() -> None:
    schedule = topo.local_only(4)

    assert schedule.graph_at(3).number_of_edges() == 0
    assert schedule.graph_at(3).number_of_nodes() == 4


@pytest.mark.parametrize("n", [20, 21])
def test_gossip_gives_each_node_at_most_one_partner_and_one_left_out_when_n_is_odd(n: int) -> None:
    schedule = topo.gossip(n, _gossip_rng(0))

    for r in range(50):
        graph = schedule.graph_at(r)
        degrees = [d for _, d in graph.degree()]
        assert max(degrees) == 1
        assert degrees.count(0) == n % 2
    assert not schedule.is_static


def test_gossip_graph_at_is_a_pure_function_of_seed_and_round() -> None:
    first = topo.gossip(20, _gossip_rng(4))
    second = topo.gossip(20, _gossip_rng(4))

    a = sorted(first.graph_at(7).edges())
    first.graph_at(3)  # gọi xen một round khác
    assert sorted(first.graph_at(7).edges()) == a
    # Hai lịch tạo độc lập (như hai node tự tính) cho cùng đồ thị, gọi theo thứ tự khác nhau.
    assert sorted(second.graph_at(7).edges()) == a
    assert [sorted(first.graph_at(r).edges()) for r in (2, 1, 0)] == [
        sorted(second.graph_at(r).edges()) for r in (2, 1, 0)
    ]


def test_gossip_expected_mixing_matrix_matches_closed_form() -> None:
    # E[W] = ½ I + (J − I)/(2(N − 1)) với N chẵn, khi mỗi cặp dùng W = ½ (ADR-009).
    n, rounds = 20, 10_000
    schedule = topo.gossip(n, _gossip_rng(1))
    total = np.zeros((n, n))
    for r in range(rounds):
        w = np.eye(n)
        for i, j in schedule.graph_at(r).edges():
            w[i, i] = w[j, j] = w[i, j] = w[j, i] = 0.5
        total += w
    mean = total / rounds
    expected = 0.5 * np.eye(n) + (np.ones((n, n)) - np.eye(n)) / (2 * (n - 1))
    p = 1 / (n - 1)
    se = 0.5 * np.sqrt(p * (1 - p) / rounds)

    assert np.abs(mean - expected).max() <= 4 * se


def test_er_gossip_uses_only_er_edges_and_is_a_maximal_matching() -> None:
    base = _er(2)
    schedule = topo.er_gossip(base, _gossip_rng(2, Stream.GOSSIP_ER))

    for r in range(50):
        graph = schedule.graph_at(r)
        assert set(map(frozenset, graph.edges())) <= set(map(frozenset, base.edges()))
        assert max(d for _, d in graph.degree()) <= 1
        unmatched = {v for v, d in graph.degree() if d == 0}
        assert not any(u in unmatched and v in unmatched for u, v in base.edges())


@pytest.mark.parametrize("kind", ["gossip", "er_gossip"])
def test_union_of_gossip_rounds_is_connected(kind: str) -> None:
    schedule = (
        topo.gossip(20, _gossip_rng(3))
        if kind == "gossip"
        else topo.er_gossip(_er(3), _gossip_rng(3, Stream.GOSSIP_ER))
    )

    union = nx.compose_all([schedule.graph_at(r) for r in range(200)])

    assert nx.is_connected(union)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: topo.ring(2),
        lambda: topo.fully_connected(1),
        lambda: topo.star(1),
        lambda: topo.gossip(1, _gossip_rng(0)),
    ],
)
def test_too_few_nodes_raise_value_error(factory: Callable[[], object]) -> None:
    with pytest.raises(ValueError):
        factory()


def test_er_that_cannot_be_connected_raises_value_error() -> None:
    with pytest.raises(ValueError, match="liên thông"):
        topo.erdos_renyi(20, 0.0, seeding.numpy_rng(0, Stream.GRAPH))


def test_er_gossip_rejects_a_disconnected_base_graph() -> None:
    with pytest.raises(ValueError, match="liên thông"):
        topo.er_gossip(nx.empty_graph(4), _gossip_rng(0, Stream.GOSSIP_ER))


def test_er_probability_outside_unit_interval_raises_value_error() -> None:
    with pytest.raises(ValueError, match="p phải"):
        topo.erdos_renyi(20, 1.5, seeding.numpy_rng(0, Stream.GRAPH))


def test_er_gossip_rejects_nodes_not_numbered_from_zero() -> None:
    with pytest.raises(ValueError, match="0..N-1"):
        topo.er_gossip(nx.path_graph([1, 2, 3]), _gossip_rng(0, Stream.GOSSIP_ER))

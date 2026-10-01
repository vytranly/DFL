"""Test cho src/dfl/network/mixing.py: ma trận trộn và spectral gap (KH §3.3; ADR-005; INV-4; T-INV-9)."""

import networkx as nx
import numpy as np
import pytest

from dfl.network import mixing


def _er(seed: int) -> nx.Graph:
    for k in range(1000):
        graph = nx.erdos_renyi_graph(20, 0.3, seed=1000 * seed + k)
        if nx.is_connected(graph):
            return graph
    raise AssertionError("không dựng được ER liên thông cho test")


def _matching(n: int, pairs: list[tuple[int, int]]) -> nx.Graph:
    graph = nx.empty_graph(n)
    graph.add_edges_from(pairs)
    return graph


def test_t_inv_9_ring_spectral_gap_matches_closed_form() -> None:
    gap = mixing.spectral_gap(mixing.metropolis_hastings(nx.cycle_graph(20)))

    assert gap == pytest.approx(2 / 3 * (1 - np.cos(2 * np.pi / 20)), abs=1e-12)
    assert gap == pytest.approx(0.033, abs=0.001)


def test_t_inv_9_fully_connected_spectral_gap_is_one() -> None:
    assert mixing.spectral_gap(mixing.metropolis_hastings(nx.complete_graph(20))) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "graph", [nx.cycle_graph(20), nx.complete_graph(20), _er(0), _er(1), nx.star_graph(19)], ids=str
)
def test_metropolis_hastings_is_symmetric_doubly_stochastic_and_follows_edges(graph: nx.Graph) -> None:
    # Act
    w = mixing.metropolis_hastings(graph)

    # Assert
    np.testing.assert_allclose(w, w.T)
    np.testing.assert_allclose(w.sum(axis=0), 1.0)
    np.testing.assert_allclose(w.sum(axis=1), 1.0)
    assert (w >= 0).all()
    off_edge = ~nx.to_numpy_array(graph, nodelist=range(len(w)), dtype=bool) & ~np.eye(len(w), dtype=bool)
    assert (w[off_edge] == 0).all()


def test_uniform_is_row_stochastic_but_not_doubly_stochastic_on_irregular_graph() -> None:
    graph = _er(0)
    assert len({d for _, d in graph.degree()}) > 1  # đồ thị không đều

    w = mixing.uniform(graph)

    np.testing.assert_allclose(w.sum(axis=1), 1.0)
    assert not np.allclose(w.sum(axis=0), 1.0)
    assert 0 < mixing.spectral_gap(w) <= 1  # W không đối xứng: gap tính bằng eigvals


def test_uniform_equals_metropolis_hastings_on_regular_graphs() -> None:
    # Vì vậy ablation uniform chỉ chạy trên ER (ADR-005).
    ring = nx.cycle_graph(20)

    np.testing.assert_allclose(mixing.uniform(ring), mixing.metropolis_hastings(ring))


def test_size_weighted_on_fully_connected_graph_gives_rows_equal_to_size_fractions() -> None:
    sizes = np.array([10, 30, 60])

    w = mixing.size_weighted(nx.complete_graph(3), sizes)

    np.testing.assert_allclose(w, np.tile(sizes / sizes.sum(), (3, 1)))


def test_size_weighted_rejects_graphs_that_are_not_fully_connected() -> None:
    with pytest.raises(ValueError, match="đầy đủ"):
        mixing.size_weighted(nx.cycle_graph(4), np.ones(4))


def test_gossip_matching_gives_symmetric_idempotent_matrix() -> None:
    # Cặp (0,1), (2,3); node 4 không được ghép (N lẻ) thì giữ nguyên model: W_44 = 1.
    w = mixing.metropolis_hastings(_matching(5, [(0, 1), (2, 3)]))

    np.testing.assert_allclose(w, w.T)
    np.testing.assert_allclose(w @ w, w)
    assert w[0, 1] == w[0, 0] == 0.5
    assert w[4, 4] == 1.0


def test_identity_is_the_local_only_matrix() -> None:
    np.testing.assert_array_equal(mixing.identity(4), np.eye(4))


@pytest.mark.parametrize(
    ("scheme", "expected"),
    [
        ("metropolis_hastings", np.full(3, 1 / 3)),
        ("uniform", np.array([2, 3, 2]) / 7),  # π_j ∝ d_j + 1 trên đường 0–1–2
        ("size_weighted", np.array([0.1, 0.3, 0.6])),
        ("size_corrected", np.array([0.1, 0.3, 0.6])),
    ],
)
def test_objective_weights_follow_adr_005(scheme: str, expected: np.ndarray) -> None:
    pi = mixing.objective_weights(scheme, nx.path_graph(3), np.array([10, 30, 60]))

    np.testing.assert_allclose(pi, expected)


def test_uniform_consensus_point_matches_its_objective_weights() -> None:
    # Lặp W uniform hội tụ về hàng π (trung bình có trọng số d_j + 1), đúng như objective_weights ghi.
    graph = _er(2)
    w = mixing.uniform(graph)

    limit = np.linalg.matrix_power(w, 2000)

    np.testing.assert_allclose(limit[0], mixing.objective_weights("uniform", graph, np.ones(20)), atol=1e-9)


@pytest.mark.parametrize(
    "graph",
    [nx.DiGraph([(0, 1)]), nx.MultiGraph([(0, 1), (0, 1)]), nx.Graph([(0, 0), (0, 1)]), nx.Graph([(1, 2)])],
    ids=["directed", "multigraph", "self-loop", "bad-labels"],
)
def test_invalid_graphs_are_rejected(graph: nx.Graph) -> None:
    with pytest.raises(ValueError):
        mixing.metropolis_hastings(graph)


def test_unknown_objective_scheme_raises_value_error() -> None:
    with pytest.raises(ValueError, match="scheme"):
        mixing.objective_weights("fedavg", nx.path_graph(3), np.ones(3))

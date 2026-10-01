"""Lịch topology T0–T4, T3b và W = I cho B2 (module network; KH §3.3; ADR-009; INV-4).

Chỉ phụ thuộc numpy/networkx (contract pure-no-torch), nên không import `seeding`: lịch ngẫu nhiên nhận
`rng_for(round) -> numpy.random.Generator` do sim dựng từ `seeding.numpy_rng(seed, Stream.GOSSIP, round)`
(T3) hoặc `Stream.GOSSIP_ER` (T3b). Mỗi round có SeedSequence riêng, nên `graph_at` là hàm thuần của
(seed, round) dù gọi bao nhiêu lần và theo thứ tự nào. Không tạo Generator từ cặp [seed, r] như H P3.2
gợi ý (plan K10).
Mọi đồ thị trả về đều bị freeze để người gọi không sửa được lịch.

networkx.cycle_graph:
https://networkx.org/documentation/networkx-3.7/reference/generated/networkx.generators.classic.cycle_graph.html
networkx.complete_graph:
https://networkx.org/documentation/networkx-3.7/reference/generated/networkx.generators.classic.complete_graph.html
networkx.star_graph (star_graph(n) có n + 1 node, hub là 0):
https://networkx.org/documentation/networkx-3.7/reference/generated/networkx.generators.classic.star_graph.html
networkx.empty_graph:
https://networkx.org/documentation/networkx-3.7/reference/generated/networkx.generators.classic.empty_graph.html
networkx.erdos_renyi_graph (seed nhận số nguyên):
https://networkx.org/documentation/networkx-3.7/reference/generated/networkx.generators.random_graphs.erdos_renyi_graph.html
networkx.is_connected:
https://networkx.org/documentation/networkx-3.7/reference/algorithms/generated/networkx.algorithms.components.is_connected.html
networkx.freeze:
https://networkx.org/documentation/networkx-3.7/reference/generated/networkx.classes.function.freeze.html
numpy.random.Generator.permutation:
https://numpy.org/doc/2.5/reference/random/generated/numpy.random.Generator.permutation.html
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

# Hàm trả về một Generator MỚI cho mỗi lần gọi với cùng round; dùng lại một Generator sẽ phá tính thuần.
RngForRound = Callable[[int], np.random.Generator]

ER_MAX_ATTEMPTS = 1_000


def _check_connected(graph: nx.Graph) -> None:
    if set(graph.nodes) != set(range(graph.number_of_nodes())):
        raise ValueError("node phải được đánh số 0..N-1 để khớp chỉ số hàng của W")
    if nx.number_of_selfloops(graph) > 0 or not nx.is_connected(graph):
        raise ValueError("đồ thị tĩnh phải liên thông và không có self-loop (INV-4)")


def _check_size(num_nodes: int, minimum: int, name: str) -> None:
    if num_nodes < minimum:
        raise ValueError(f"{name} cần N ≥ {minimum}, nhận {num_nodes}")


def _matching(num_nodes: int, pairs: list[tuple[int, int]]) -> nx.Graph:
    graph = nx.empty_graph(num_nodes)
    graph.add_edges_from(pairs)
    return nx.freeze(graph)


@dataclass(frozen=True)
class StaticSchedule:
    """Cùng một đồ thị cho mọi round (T0 star, T1 ring, T2 ER, T4 FC, B2)."""

    graph: nx.Graph
    is_static: bool = field(default=True, init=False)

    @property
    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()

    def graph_at(self, round_idx: int) -> nx.Graph:
        return self.graph


@dataclass(frozen=True)
class GossipSchedule:
    """T3: perfect matching ngẫu nhiên trên K_N mỗi round; N lẻ thì đúng một node đứng ngoài (ADR-009).

    Ghép (p0, p1), (p2, p3), … của một hoán vị, nên mọi matching cực đại của K_N có cùng xác suất.
    """

    num_nodes: int
    rng_for: RngForRound = field(repr=False)
    is_static: bool = field(default=False, init=False)

    def graph_at(self, round_idx: int) -> nx.Graph:
        perm = self.rng_for(round_idx).permutation(self.num_nodes)
        pairs = perm[: self.num_nodes - self.num_nodes % 2].reshape(-1, 2).tolist()
        return _matching(self.num_nodes, [(u, v) for u, v in pairs])


@dataclass(frozen=True)
class ERGossipSchedule:
    """T3b: maximal matching ngẫu nhiên trên cạnh của đồ thị T2 mỗi round (đối chứng công bằng cho H4).

    Xáo thứ tự cạnh rồi nhận tham lam; `edges` đã sắp theo thứ tự chuẩn để kết quả không phụ thuộc thứ tự
    chèn cạnh của networkx.
    """

    num_nodes: int
    edges: tuple[tuple[int, int], ...]
    rng_for: RngForRound = field(repr=False)
    is_static: bool = field(default=False, init=False)

    def graph_at(self, round_idx: int) -> nx.Graph:
        order = self.rng_for(round_idx).permutation(len(self.edges))
        matched: set[int] = set()
        pairs: list[tuple[int, int]] = []
        for k in order:
            u, v = self.edges[k]
            if u not in matched and v not in matched:
                pairs.append((u, v))
                matched.update((u, v))
        return _matching(self.num_nodes, pairs)


def ring(num_nodes: int) -> StaticSchedule:
    """T1. N ≥ 3, vì cycle_graph(1) có self-loop và cycle_graph(2) chỉ là một cạnh."""
    _check_size(num_nodes, 3, "ring")
    return StaticSchedule(nx.freeze(nx.cycle_graph(num_nodes)))


def fully_connected(num_nodes: int) -> StaticSchedule:
    """T4."""
    _check_size(num_nodes, 2, "fully_connected")
    return StaticSchedule(nx.freeze(nx.complete_graph(num_nodes)))


def star(num_nodes: int) -> StaticSchedule:
    """T0: hub là node 0; chỉ để tính chi phí của B3 (FedAvg có server)."""
    _check_size(num_nodes, 2, "star")
    return StaticSchedule(nx.freeze(nx.star_graph(num_nodes - 1)))


def local_only(num_nodes: int) -> StaticSchedule:
    """B2: không có cạnh nên W = I; ngoại lệ duy nhất của INV-4 về liên thông."""
    _check_size(num_nodes, 1, "local_only")
    return StaticSchedule(nx.freeze(nx.empty_graph(num_nodes)))


def erdos_renyi(num_nodes: int, p: float, rng: np.random.Generator) -> StaticSchedule:
    """T2: lấy mẫu lại G(N, p) tới khi liên thông, tối đa ER_MAX_ATTEMPTS lần.

    Mỗi lần thử lấy seed mới từ `rng` (luồng Stream.GRAPH), nên đồ thị tất định theo seed của lượt.
    """
    _check_size(num_nodes, 2, "erdos_renyi")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p phải nằm trong [0, 1], nhận {p}")
    for _ in range(ER_MAX_ATTEMPTS):
        graph = nx.erdos_renyi_graph(num_nodes, p, seed=int(rng.integers(2**31)))
        if nx.is_connected(graph):
            return StaticSchedule(nx.freeze(graph))
    raise ValueError(
        f"không dựng được ER liên thông sau {ER_MAX_ATTEMPTS:,} lần với N = {num_nodes}, p = {p}"
    )


def gossip(num_nodes: int, rng_for: RngForRound) -> GossipSchedule:
    """T3; `rng_for` dựng từ Stream.GOSSIP."""
    _check_size(num_nodes, 2, "gossip")
    return GossipSchedule(num_nodes, rng_for)


def er_gossip(base_graph: nx.Graph, rng_for: RngForRound) -> ERGossipSchedule:
    """T3b trên cạnh của `base_graph` (đồ thị T2 của cùng lượt); `rng_for` dựng từ Stream.GOSSIP_ER."""
    _check_connected(base_graph)
    edges = tuple(sorted((min(u, v), max(u, v)) for u, v in base_graph.edges()))
    return ERGossipSchedule(base_graph.number_of_nodes(), edges, rng_for)

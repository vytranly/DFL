"""T-INV-6 qua engine, node gossip không được ghép, INV-4 do engine kiểm (T10b; KH §3.6; SPEC REQ-05, §9.2).

N = 20 như cấu hình chuẩn: ring 40, FC 380, gossip trên K_N 20, ER 2·|E|, đường server 2N message mỗi round.
Byte = số message × `nbytes` của snapshot; không có message tự gửi (G7).
"""

import networkx as nx
import numpy as np
import pytest
import torch

from dfl import seeding
from dfl.contracts import Snapshot
from dfl.network import mixing
from dfl.network import topology as topo
from dfl.network.transport import InMemoryTransport
from dfl.seeding import Stream
from dfl.sim.engine import Engine

N20 = 20


class PairRecorder(InMemoryTransport):
    """InMemoryTransport ghi lại (round, người gửi, người nhận) của mọi message."""

    def __init__(self, num_nodes: int, *, with_server: bool = False) -> None:
        super().__init__(num_nodes, with_server=with_server)
        self.pairs: list[tuple[int, int, int]] = []

    def send(self, dst: int, message: Snapshot) -> None:
        super().send(dst, message)
        self.pairs.append((message.round_idx, message.sender, dst))


def mh(graph: nx.Graph, node_sizes: np.ndarray) -> np.ndarray:
    return mixing.metropolis_hastings(graph)


def gossip_schedule(num_nodes: int, seed: int) -> topo.GossipSchedule:
    return topo.gossip(num_nodes, lambda r: seeding.numpy_rng(seed, Stream.GOSSIP, r))


def er_schedule() -> topo.StaticSchedule:
    return topo.erdos_renyi(N20, 0.3, seeding.numpy_rng(0, Stream.GRAPH))


CASES = {
    "ring": (lambda: topo.ring(N20), 40),
    "fc": (lambda: topo.fully_connected(N20), 380),
    "gossip": (lambda: gossip_schedule(N20, 0), 20),
    "er": (er_schedule, None),  # 2·|E| của đồ thị đã lấy mẫu
    "server": (None, 2 * N20),
}


@pytest.mark.parametrize("case", list(CASES))
def test_t_inv_6_messages_and_bytes_per_round_through_engine(case, make_nodes, sgd_trainer) -> None:
    # Arrange
    make_schedule, per_round = CASES[case]
    nodes = make_nodes((64,) * N20)
    nbytes = sum(t.numel() * t.element_size() for t in nodes[0].model.state_dict().values())
    transport = PairRecorder(N20, with_server=make_schedule is None)
    if make_schedule is None:
        engine = Engine(
            nodes=nodes, trainer=sgd_trainer(tau=1, lr=0.1), transport=transport, server_path=True
        )
    else:
        engine = Engine(
            nodes=nodes,
            trainer=sgd_trainer(tau=1, lr=0.1),
            schedule=make_schedule(),
            mixing=mh,
            transport=transport,
        )
        independent = make_schedule()
    # Gossip chỉ chạy trong phạm vi đã kiểm INV-4. Hai matching trên K_20 chưa thể liên thông, nên khai báo
    # phạm vi của một lượt thật (50 round) rồi chỉ chạy hai round đầu.
    engine.check_schedule(50)

    before = transport.counters()
    for r in range(2):
        # Act
        engine.run_round(r)
        after = transport.counters()

        # Assert: mỗi cặp có hướng đúng một message, và chỉ trên cạnh của round (hay giữa node và server).
        pairs = [(sender, dst) for round_idx, sender, dst in transport.pairs if round_idx == r]
        if make_schedule is None:
            expected_pairs = {(i, N20) for i in range(N20)} | {(N20, i) for i in range(N20)}
        else:
            edges = independent.graph_at(r).edges()
            expected_pairs = {(i, j) for i, j in edges} | {(j, i) for i, j in edges}
        assert len(pairs) == len(set(pairs))
        assert set(pairs) == expected_pairs
        assert all(sender != dst for sender, dst in pairs)
        messages = after.messages_cum - before.messages_cum
        assert messages == (
            per_round if per_round is not None else 2 * independent.graph_at(r).number_of_edges()
        )
        assert messages == len(pairs)
        assert after.bytes_cum - before.bytes_cum == messages * nbytes
        before = after
    if make_schedule is None:
        # Server ảo nhận N và gửi N snapshot mỗi round, nên gánh tải lớn nhất (SPEC REQ-05).
        assert (
            transport.node_bytes()[N20] == 2 * (2 * N20 * nbytes) == transport.counters().max_node_bytes_cum
        )


def test_unmatched_gossip_node_still_trains_and_keeps_its_own_model(make_nodes, sgd_trainer) -> None:
    # Arrange: N lẻ, nên mỗi round đúng một node không được ghép.
    trainer = sgd_trainer(tau=2, lr=0.1)
    transport = PairRecorder(5)
    engine = Engine(
        nodes=make_nodes((64,) * 5),
        trainer=trainer,
        schedule=gossip_schedule(5, 3),
        mixing=mh,
        transport=transport,
    )
    independent = gossip_schedule(5, 3)
    engine.check_schedule(50)  # phạm vi của một lượt thật; chỉ chạy ba round đầu

    for r in range(3):
        # Act
        result = engine.run_round(r)
        states = engine.states()

        # Assert: node đứng ngoài vẫn chạy Adapt, giữ W_ii = 1 và đúng model sau Adapt của nó.
        (unmatched,) = [v for v, degree in independent.graph_at(r).degree() if degree == 0]
        assert sorted(call.node_id for call in trainer.calls[5 * r : 5 * (r + 1)]) == list(range(5))
        assert result.weights[unmatched, unmatched] == 1.0
        for name, tensor in result.snapshots[unmatched].state.items():
            assert torch.equal(states[unmatched][name], tensor)
        round_pairs = [(sender, dst) for round_idx, sender, dst in transport.pairs if round_idx == r]
        assert len(round_pairs) == 4
        assert all(unmatched not in pair for pair in round_pairs)


def fc_weights_on_any_graph(graph: nx.Graph, node_sizes: np.ndarray) -> np.ndarray:
    n = graph.number_of_nodes()
    return np.full((n, n), 1 / n)


def half_mh(graph: nx.Graph, node_sizes: np.ndarray) -> np.ndarray:
    return 0.5 * mixing.metropolis_hastings(graph)


def mh_with_negative_self_weight(graph: nx.Graph, node_sizes: np.ndarray) -> np.ndarray:
    w = mixing.metropolis_hastings(graph)
    w[0, 0] -= 0.5
    w[0, 1] += 0.5  # tổng hàng vẫn bằng 1, giá vẫn trên cạnh, nhưng có phần tử âm
    return w


def mh_with_nan(graph: nx.Graph, node_sizes: np.ndarray) -> np.ndarray:
    w = mixing.metropolis_hastings(graph)
    w[1, 1] = np.nan
    return w


def wrong_shape(graph: nx.Graph, node_sizes: np.ndarray) -> np.ndarray:
    return np.eye(graph.number_of_nodes() + 1)


@pytest.mark.parametrize(
    ("policy", "message"),
    [
        (fc_weights_on_any_graph, "ngoài cạnh"),
        (half_mh, "tổng mỗi hàng"),
        (mh_with_negative_self_weight, "không âm"),
        (mh_with_nan, "hữu hạn"),
        (wrong_shape, "shape"),
    ],
    ids=["off-graph", "row-sum", "negative", "nan", "shape"],
)
def test_engine_rejects_mixing_weights_that_break_inv_4(policy, message, make_nodes, sgd_trainer) -> None:
    engine = Engine(
        nodes=make_nodes(),
        trainer=sgd_trainer(tau=1, lr=0.1),
        schedule=topo.ring(4),
        mixing=policy,
        transport=InMemoryTransport(4),
    )

    with pytest.raises(ValueError, match=message):
        engine.weights_at(0)
    with pytest.raises(ValueError, match=message):
        engine.run_round(0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"schedule": topo.ring(4), "mixing": mh, "server_path": True}, "đường server"),
        ({"mixing": mh}, "schedule và mixing"),
        ({"schedule": topo.ring(4)}, "schedule và mixing"),
    ],
    ids=["server-with-schedule", "no-schedule", "no-mixing"],
)
def test_engine_requires_schedule_and_mixing_unless_server_path(
    kwargs, message, make_nodes, sgd_trainer
) -> None:
    with pytest.raises(ValueError, match=message):
        Engine(
            nodes=make_nodes(),
            trainer=sgd_trainer(tau=1, lr=0.1),
            transport=InMemoryTransport(4, with_server=True),
            **kwargs,
        )


class NothingReachesTheServer(InMemoryTransport):
    def send(self, dst: int, message: Snapshot) -> None:
        if dst != N20:
            super().send(dst, message)


def test_server_path_fails_loudly_when_the_server_misses_a_snapshot(make_nodes, sgd_trainer) -> None:
    engine = Engine(
        nodes=make_nodes((64,) * N20),
        trainer=sgd_trainer(tau=1, lr=0.1),
        transport=NothingReachesTheServer(N20, with_server=True),
        server_path=True,
    )

    with pytest.raises(RuntimeError, match="server ảo thiếu snapshot"):
        engine.run_round(0)

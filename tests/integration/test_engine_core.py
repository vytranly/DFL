"""Test tích hợp cho vòng ATC lõi của engine (T10a; KH §3.4, §4.5; SPEC §9.2: T-INV-2, T-INV-4; INV-1, INV-3).

MLP tí hon, dữ liệu tổng hợp 4 node (mỗi node 3 lớp), trainer theo Protocol LocalTrainer (tests/integration/
conftest.py). So khớp "trùng từng bit" dùng torch.equal (SPEC §7):
https://docs.pytorch.org/docs/2.14/generated/torch.equal.html
"""

import networkx as nx
import numpy as np
import pytest
import torch

from dfl import seeding
from dfl.contracts import CommCounters, Snapshot, StateDict
from dfl.network import mixing
from dfl.network import topology as topo
from dfl.network.transport import InMemoryTransport
from dfl.seeding import Stream
from dfl.sim.engine import Engine
from dfl.sim.node import Node

N = 4


def mh(graph: nx.Graph, node_sizes: np.ndarray) -> np.ndarray:
    return mixing.metropolis_hastings(graph)


def er_paw() -> topo.StaticSchedule:
    # Cạnh 01, 02, 03, 13: đồ thị không đều, nên trọng số MH khác nhau giữa các node (ρ = 0.75).
    return topo.erdos_renyi(N, 0.5, seeding.numpy_rng(0, Stream.GRAPH))


def assert_states_equal(a: StateDict, b: StateDict) -> None:
    assert a.keys() == b.keys()
    for name in a:
        assert torch.equal(a[name], b[name]), name


def test_t_inv_2_identity_mixing_matches_local_only_training(make_nodes, sgd_trainer) -> None:
    # Arrange: oracle là cùng 4 node (cùng w⁰, dữ liệu, luồng batch) tự huấn luyện, không giao tiếp.
    nodes, oracle = make_nodes(), make_nodes()
    w0 = [{name: t.clone() for name, t in node.model.state_dict().items()} for node in nodes]
    transport = InMemoryTransport(N)
    engine = Engine(
        nodes=nodes,
        trainer=sgd_trainer(tau=3, lr=0.1, mu=0.1),
        schedule=topo.local_only(N),
        mixing=mh,
        transport=transport,
    )
    oracle_trainer = sgd_trainer(tau=3, lr=0.1, mu=0.1)

    # Act
    results = [engine.run_round(r) for r in range(3)]
    oracle_losses = [
        tuple(
            oracle_trainer(node.model, node.data, loss_scale=node.loss_scale, generator=node.generator)
            for node in oracle
        )
        for _ in range(3)
    ]

    # Assert
    for result in results:
        np.testing.assert_array_equal(result.weights, np.eye(N))
    states = engine.states()
    for state, node in zip(states, oracle, strict=True):
        assert_states_equal(state, node.model.state_dict())
    assert [result.train_losses for result in results] == oracle_losses
    assert transport.counters() == CommCounters(messages_cum=0, bytes_cum=0, max_node_bytes_cum=0)
    # Không suông: các node đã học, và học khác nhau.
    assert not torch.equal(states[0]["fc1.weight"], w0[0]["fc1.weight"])
    assert not torch.equal(states[0]["fc1.weight"], states[1]["fc1.weight"])


@pytest.mark.parametrize("schedule", [topo.ring(N), er_paw()], ids=["ring", "er-paw"])
def test_t_inv_4_node_visiting_order_does_not_change_result(
    schedule, make_nodes, sgd_trainer, recording_transport
) -> None:
    # Arrange: ba thứ tự duyệt, và một lượt mà inbox trả về theo thứ tự chèn ngược.
    variants = [((0, 1, 2, 3), False), ((3, 2, 1, 0), False), ((2, 0, 3, 1), False), ((0, 1, 2, 3), True)]
    trajectories = []
    for order, reverse_inbox in variants:
        nodes = make_nodes()
        trainer = sgd_trainer(tau=2, lr=0.1, mu=0.1)
        transport = recording_transport(N, reverse_inbox=reverse_inbox)
        engine = Engine(
            nodes=[nodes[i] for i in order],
            trainer=trainer,
            schedule=schedule,
            mixing=mh,
            transport=transport,
        )

        # Act
        trajectory = []
        for r in range(3):
            result = engine.run_round(r)
            trajectory.append((result, engine.states()))
        trajectories.append(trajectory)

        # Assert: engine duyệt đúng thứ tự được giao (nếu tự sắp xếp thì test này suông) ...
        assert [call.node_id for call in trainer.calls[:N]] == list(order)
        assert [dst for dst, _ in transport.inbox_reads[:N]] == list(order)
        # ... và không phép tính nào sửa snapshot sau khi gửi (checksum lúc gửi và sau tổng hợp).
        for sent in transport.sent:
            assert_states_equal(sent.message.state, sent.state_at_send)

    # Assert: mọi biến thể trùng từng bit với thứ tự 0, 1, 2, 3 ở mọi round.
    for trajectory in trajectories[1:]:
        for (result, states), (ref_result, ref_states) in zip(trajectory, trajectories[0], strict=True):
            assert result.train_losses == ref_result.train_losses
            for state, ref_state in zip(states, ref_states, strict=True):
                assert_states_equal(state, ref_state)
    # Không suông: với dữ liệu này, cộng theo thứ tự khác cho ra float khác, nên test thấy được thứ tự cộng.
    last = trajectories[0][-1][0]
    ascending = sum(last.weights[0, j] * last.snapshots[j].state["fc1.weight"] for j in range(N))
    descending = sum(last.weights[0, j] * last.snapshots[j].state["fc1.weight"] for j in reversed(range(N)))
    assert not torch.equal(ascending, descending)


def test_combine_reads_only_own_snapshot_and_own_inbox(make_nodes, sgd_trainer, recording_transport) -> None:
    # Arrange: inbox giao hằng số sender + 1 thay cho model thật của láng giềng.
    transport = recording_transport(N, tamper=True)
    engine = Engine(
        nodes=make_nodes(),
        trainer=sgd_trainer(tau=2, lr=0.1),
        schedule=topo.ring(N),
        mixing=mh,
        transport=transport,
    )

    for r in range(2):
        # Act
        result = engine.run_round(r)
        states = engine.states()

        # Assert: model mới chỉ gồm snapshot của chính node và nội dung inbox (INV-1).
        w = result.weights
        for i in range(N):
            for name, own in result.snapshots[i].state.items():
                expected = w[i, i] * own + sum(
                    w[i, j] * (j + 1.0) for j in range(N) if j != i and w[i, j] > 0
                )
                torch.testing.assert_close(states[i][name], expected, rtol=0, atol=1e-6)
        # Mỗi node đọc inbox của chính nó đúng một lần mỗi round.
        assert sorted(read for read in transport.inbox_reads if read[1] == r) == [(i, r) for i in range(N)]
        # Mỗi node tạo đúng một snapshot, gửi chính nó tới mọi láng giềng; đó cũng là snapshot trong kết quả.
        for i in range(N):
            messages = [
                s.message for s in transport.sent if s.message.sender == i and s.message.round_idx == r
            ]
            assert len(messages) == 2
            assert all(message is result.snapshots[i] for message in messages)


def test_combine_equals_weighted_sum_of_snapshots(make_nodes, sgd_trainer) -> None:
    schedule = er_paw()
    engine = Engine(
        nodes=make_nodes(dtype=torch.float64),
        trainer=sgd_trainer(tau=2, lr=0.1),
        schedule=schedule,
        mixing=mh,
        transport=InMemoryTransport(N),
    )

    for r in range(2):
        result = engine.run_round(r)
        states = engine.states()

        np.testing.assert_array_equal(result.weights, mixing.metropolis_hastings(schedule.graph_at(r)))
        for i in range(N):
            for name in states[i]:
                expected = sum(result.weights[i, j] * result.snapshots[j].state[name] for j in range(N))
                torch.testing.assert_close(states[i][name], expected, rtol=0, atol=1e-12)


def test_fully_connected_mh_nodes_are_bitwise_identical_after_combine(make_nodes, sgd_trainer) -> None:
    engine = Engine(
        nodes=make_nodes(),
        trainer=sgd_trainer(tau=2, lr=0.1),
        schedule=topo.fully_connected(N),
        mixing=mh,
        transport=InMemoryTransport(N),
    )

    for r in range(2):
        result = engine.run_round(r)
        states = engine.states()

        # Trước tổng hợp các model khác nhau (drift > 0) ...
        assert not torch.equal(
            result.snapshots[0].state["fc1.weight"], result.snapshots[1].state["fc1.weight"]
        )
        # ... sau tổng hợp mọi node trùng từng bit: mọi node cộng cùng các số hạng theo cùng thứ tự id.
        for state in states[1:]:
            assert_states_equal(state, states[0])


def test_integer_buffers_stay_per_node_and_float_buffers_are_mixed(make_nodes, sgd_trainer) -> None:
    # Arrange: buffer float = 1 + i, buffer int64 = 100 i; trainer cộng τ = 3 vào buffer int.
    engine = Engine(
        nodes=make_nodes(buffers=True),
        trainer=sgd_trainer(tau=3, lr=0.1),
        schedule=topo.fully_connected(N),
        mixing=mh,
        transport=InMemoryTransport(N),
    )

    # Act
    engine.run_round(0)

    # Assert: trên FC với MH, W = 1/4 mọi chỗ, nên buffer float thành (1 + 2 + 3 + 4)/4 = 2.5 ở mọi node (G3).
    for i, state in enumerate(engine.states()):
        assert state["int_buf"].dtype == torch.int64
        assert state["int_buf"].item() == 100 * i + 3
        assert torch.equal(state["float_buf"], torch.full((3,), 2.5))


def test_trainer_gets_the_node_model_at_round_start_and_its_loss_scale(make_nodes, sgd_trainer) -> None:
    # Arrange
    loss_scales = (0.5, 1.0, 1.5, 2.0)
    trainer = sgd_trainer(tau=2, lr=0.1, mu=0.5)
    engine = Engine(
        nodes=make_nodes(loss_scales=loss_scales),
        trainer=trainer,
        schedule=topo.ring(N),
        mixing=mh,
        transport=InMemoryTransport(N),
    )

    # Act
    round_starts = [engine.states()]
    for r in range(3):
        engine.run_round(r)
        round_starts.append(engine.states())

    # Assert: anchor của round t là w_i^t, tức model của node ngay sau round t − 1 (G6).
    assert len(trainer.calls) == 3 * N
    for r in range(3):
        for call in trainer.calls[r * N : (r + 1) * N]:
            assert call.loss_scale == loss_scales[call.node_id]
            assert_states_equal(call.anchor, {k: round_starts[r][call.node_id][k] for k in call.anchor})
    anchors_of_node_0 = [call.anchor["fc1.weight"] for call in trainer.calls if call.node_id == 0]
    assert not torch.equal(anchors_of_node_0[0], anchors_of_node_0[1])
    assert not torch.equal(anchors_of_node_0[1], anchors_of_node_0[2])


def test_run_round_rejects_out_of_order_rounds(make_nodes, sgd_trainer) -> None:
    engine = Engine(
        nodes=make_nodes(),
        trainer=sgd_trainer(tau=1, lr=0.1),
        schedule=topo.ring(N),
        mixing=mh,
        transport=InMemoryTransport(N),
    )

    with pytest.raises(RuntimeError, match="thứ tự"):
        engine.run_round(1)
    engine.run_round(0)
    with pytest.raises(RuntimeError, match="thứ tự"):
        engine.run_round(0)


def test_states_are_owned_copies(make_nodes, sgd_trainer) -> None:
    # Arrange: hai engine giống hệt nhau; chỉ ở engine a, bản state trả về bị sửa.
    engines = [
        Engine(
            nodes=make_nodes(),
            trainer=sgd_trainer(tau=2, lr=0.1),
            schedule=topo.ring(N),
            mixing=mh,
            transport=InMemoryTransport(N),
        )
        for _ in range(2)
    ]
    for engine in engines:
        engine.run_round(0)

    # Act
    for tensor in engines[0].states()[0].values():
        tensor.add_(1)
    for engine in engines:
        engine.run_round(1)

    # Assert
    for state_a, state_b in zip(engines[0].states(), engines[1].states(), strict=True):
        assert_states_equal(state_a, state_b)


@pytest.mark.parametrize(
    ("ids", "schedule_size"), [((0, 1, 2, 2), 4), ((1, 2, 3, 4), 4), ((0, 1, 2, 3), 5)], ids=str
)
def test_engine_rejects_node_ids_or_schedule_size_that_do_not_match(
    ids, schedule_size, make_nodes, sgd_trainer
) -> None:
    nodes = make_nodes()
    nodes = [Node(i, node.model, node.data, node.generator) for i, node in zip(ids, nodes, strict=True)]

    with pytest.raises(ValueError):
        Engine(
            nodes=nodes,
            trainer=sgd_trainer(tau=1, lr=0.1),
            schedule=topo.ring(schedule_size),
            mixing=mh,
            transport=InMemoryTransport(N),
        )


class _LossyTransport(InMemoryTransport):
    """Làm rơi mọi message: node không nhận được snapshot mà W của nó cần."""

    def send(self, dst: int, message: Snapshot) -> None:
        return None


def test_combine_fails_loudly_when_a_needed_snapshot_is_missing(make_nodes, sgd_trainer) -> None:
    engine = Engine(
        nodes=make_nodes(),
        trainer=sgd_trainer(tau=1, lr=0.1),
        schedule=topo.ring(N),
        mixing=mh,
        transport=_LossyTransport(N),
    )

    with pytest.raises(RuntimeError, match="thiếu snapshot"):
        engine.run_round(0)

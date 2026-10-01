"""Test hồi quy cho các phát hiện của phản biện doubt-driven về engine (T10c; H P3.3 bước 4; SESSION_LOG).

Mỗi test khóa một phát hiện đã xác nhận: cấu hình sai bị từ chối trước khi huấn luyện; engine không chạy tiếp
sau lỗi giữa round; RunDiverged qua được ranh giới tiến trình; lý do phân kỳ không phụ thuộc thứ tự node.
"""

import copy
import math
from collections.abc import Mapping

import networkx as nx
import numpy as np
import pytest
import torch

from dfl import seeding
from dfl.contracts import LocalTrainer, MixingPolicy, NodeDataset, Snapshot, TopologySchedule, Transport
from dfl.network import mixing
from dfl.network import topology as topo
from dfl.network.transport import InMemoryTransport
from dfl.seeding import Stream
from dfl.sim.engine import Engine, RunDiverged
from dfl.sim.node import Node
from dfl.sim.observer import run_rounds

N = 4


def mh(graph: nx.Graph, node_sizes: np.ndarray) -> np.ndarray:
    return mixing.metropolis_hastings(graph)


def engine_for(
    nodes: list[Node],
    trainer: LocalTrainer,
    schedule: TopologySchedule | None = None,
    policy: MixingPolicy = mh,
    transport: Transport | None = None,
) -> Engine:
    return Engine(
        nodes=nodes,
        trainer=trainer,
        schedule=topo.ring(len(nodes)) if schedule is None else schedule,
        mixing=policy,
        transport=InMemoryTransport(len(nodes)) if transport is None else transport,
    )


def test_run_diverged_survives_pickling() -> None:
    # Lượt có thể chạy trong tiến trình con của sweep; ngoại lệ không pickle được làm hỏng pool (A3).
    # deepcopy dựng lại ngoại lệ qua đúng __reduce__ mà pickle dùng, nên bắt cùng lỗi.
    error = RunDiverged(3, "mất mát của node 1 không hữu hạn")

    restored = copy.deepcopy(error)

    assert (restored.round_idx, restored.reason, restored.diverged_at_round) == (3, error.reason, 4)
    assert str(restored) == str(error)


def _replace(node: Node, **changes: object) -> Node:
    fields = {"node_id": node.node_id, "model": node.model, "data": node.data, "generator": node.generator}
    fields["loss_scale"] = node.loss_scale
    return Node(**{**fields, **changes})


def shared_generator(nodes: list[Node]) -> list[Node]:
    return [nodes[0]] + [_replace(node, generator=nodes[0].generator) for node in nodes[1:]]


def shared_model(nodes: list[Node]) -> list[Node]:
    return [nodes[0], _replace(nodes[1], model=nodes[0].model), *nodes[2:]]


def shared_parameter(nodes: list[Node]) -> list[Node]:
    nodes[1].model.fc1.weight = nodes[0].model.fc1.weight  # hai module khác nhau, chung một tensor
    return nodes


def foreign_data(nodes: list[Node]) -> list[Node]:
    return [_replace(nodes[0], data=nodes[1].data), *nodes[1:]]


def empty_data(nodes: list[Node]) -> list[Node]:
    data = nodes[2].data
    return [*nodes[:2], _replace(nodes[2], data=NodeDataset(2, data.x[:0], data.y[:0])), nodes[3]]


def batch_norm(nodes: list[Node]) -> list[Node]:
    nodes[3].model.norm = torch.nn.BatchNorm1d(3)
    return nodes


def other_schema(nodes: list[Node]) -> list[Node]:
    nodes[2].model.fc2 = torch.nn.utils.skip_init(torch.nn.Linear, 4, 7)
    return nodes


def other_dtype(nodes: list[Node]) -> list[Node]:
    nodes[1].model.double()
    return nodes


def non_finite_init(nodes: list[Node]) -> list[Node]:
    with torch.no_grad():
        nodes[1].model.fc2.bias[0] = math.nan
    return nodes


def instance_norm_with_running_stats(nodes: list[Node]) -> list[Node]:
    nodes[0].model.norm = torch.nn.InstanceNorm1d(3, track_running_stats=True)
    return nodes


def mismatched_xy(nodes: list[Node]) -> list[Node]:
    data = nodes[3].data
    return [*nodes[:3], _replace(nodes[3], data=NodeDataset(3, data.x[:10], data.y))]


def equal_seeds(nodes: list[Node]) -> list[Node]:
    return [_replace(node, generator=torch.Generator().manual_seed(5)) for node in nodes]


def seeds_equal_in_low_32_bits(nodes: list[Node]) -> list[Node]:
    # Generator CPU của torch (MT19937) chỉ dùng 32 bit thấp của seed: 7 và 7 + 2³² cho cùng một dãy.
    return [
        _replace(node, generator=torch.Generator().manual_seed(7 + i * 2**32)) for i, node in enumerate(nodes)
    ]


def default_generator(nodes: list[Node]) -> list[Node]:
    return [_replace(nodes[0], generator=torch.default_generator), *nodes[1:]]


def dropout(nodes: list[Node]) -> list[Node]:
    nodes[2].model.drop = torch.nn.Dropout(0.5)
    return nodes


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        (shared_generator, "Generator riêng"),
        (shared_model, "bộ nhớ"),
        (shared_parameter, "bộ nhớ"),
        (foreign_data, "dữ liệu của node"),
        (empty_data, "không có mẫu"),
        (batch_norm, "BatchNorm"),
        (other_schema, "cấu trúc state"),
        (other_dtype, "cấu trúc state"),
        (non_finite_init, "w⁰"),
        (instance_norm_with_running_stats, "thống kê chạy"),
        (mismatched_xy, "x và y"),
        (equal_seeds, "seed"),
        (seeds_equal_in_low_32_bits, "seed"),
        (default_generator, "default_generator"),
        (dropout, "Dropout"),
    ],
    ids=[
        "shared-generator",
        "shared-model",
        "shared-parameter",
        "foreign-data",
        "empty-data",
        "batch-norm",
        "other-schema",
        "other-dtype",
        "non-finite-init",
        "instance-norm",
        "xy-length",
        "equal-seeds",
        "seeds-low-32-bits",
        "default-generator",
        "dropout",
    ],
)
def test_engine_rejects_nodes_that_are_not_independent_or_well_formed(
    corrupt, message, make_nodes, sgd_trainer
) -> None:
    # Phát hiện A1, A5, A7, A10, B10, B11: lỗi lắp ráp này không làm hỏng chương trình mà làm sai kết quả
    # (phụ thuộc thứ tự node, trộn thống kê BatchNorm, hay báo "phân kỳ" cho một lỗi khởi tạo).
    nodes = corrupt(make_nodes())

    with pytest.raises(ValueError, match=message):
        engine_for(nodes, sgd_trainer(tau=1, lr=0.1))


@pytest.mark.parametrize("loss_scale", [0.0, -1.0, math.nan, math.inf])
def test_engine_rejects_loss_scales_that_are_not_finite_and_positive(
    loss_scale, make_nodes, sgd_trainer
) -> None:
    nodes = make_nodes()
    nodes[1] = _replace(nodes[1], loss_scale=loss_scale)

    with pytest.raises(ValueError, match="loss_scale"):
        engine_for(nodes, sgd_trainer(tau=1, lr=0.1))


def test_engine_needs_at_least_one_node(sgd_trainer) -> None:
    with pytest.raises(ValueError, match="ít nhất một node"):
        Engine(nodes=[], trainer=sgd_trainer(tau=1, lr=0.1), transport=InMemoryTransport(1), server_path=True)


class FixedGraph:
    """Lịch tĩnh trả đúng một đồ thị dựng tay, kể cả đồ thị sai (để kiểm engine tự từ chối)."""

    is_static = True

    def __init__(self, graph: nx.Graph, num_nodes: int = N) -> None:
        self.graph = graph
        self.num_nodes = num_nodes

    def graph_at(self, round_idx: int) -> nx.Graph:
        return self.graph


def identity(graph: nx.Graph, node_sizes: np.ndarray) -> np.ndarray:
    return np.eye(N)


def ring_with_self_loop() -> nx.Graph:
    graph = nx.cycle_graph(N)
    graph.add_edge(0, 0)
    return graph


@pytest.mark.parametrize(
    ("graph", "policy", "message"),
    [
        (ring_with_self_loop(), identity, "self-loop"),
        (nx.empty_graph([1, 2, 3, 4]), identity, "đúng các node"),
        (nx.path_graph(N + 1), identity, "đúng các node"),
        (
            nx.cycle_graph(N),
            identity,
            "bằng 0 trên một cạnh",
        ),  # B2 đặt trên ring: tính phí message không dùng
        (nx.Graph([(0, 1), (2, 3)]), mh, "liên thông"),
    ],
    ids=["self-loop", "labels", "extra-node", "zero-weight-edge", "disconnected"],
)
def test_engine_rejects_round_graphs_that_break_inv_4(
    graph, policy, message, make_nodes, sgd_trainer
) -> None:
    # Phát hiện A6, B3, B7: lỗi phải lộ ra trước Adapt, không phải lúc gửi message giữa round.
    trainer = sgd_trainer(tau=1, lr=0.1)
    engine = engine_for(make_nodes(), trainer, schedule=FixedGraph(graph), policy=policy)

    with pytest.raises(ValueError, match=message):
        engine.run_round(0)
    assert trainer.calls == []  # chưa node nào bị huấn luyện


class FailingTransport(InMemoryTransport):
    """Lần gửi đầu tiên lỗi, như một lỗi mạng thoáng qua."""

    def __init__(self, num_nodes: int) -> None:
        super().__init__(num_nodes)
        self.failed = False

    def send(self, dst: int, message: Snapshot) -> None:
        if not self.failed:
            self.failed = True
            raise OSError("mất kết nối")
        super().send(dst, message)


def test_engine_cannot_be_resumed_after_a_failure_mid_round(make_nodes, sgd_trainer) -> None:
    # Phát hiện A2, B4: model đã qua Adapt; chạy lại round sẽ huấn luyện hai lần mà không báo gì.
    engine = engine_for(make_nodes(), sgd_trainer(tau=1, lr=0.1), transport=FailingTransport(N))
    with pytest.raises(OSError):
        engine.run_round(0)

    with pytest.raises(RuntimeError, match="giữa chừng"):
        engine.run_round(0)


class NanForNodes:
    """Trainer giả: mất mát NaN cho các node chỉ định, không đổi model."""

    def __init__(self, nodes: set[int]) -> None:
        self.nodes = nodes

    def __call__(
        self, model: torch.nn.Module, data: NodeDataset, *, loss_scale: float, generator: torch.Generator
    ) -> float:
        return math.nan if data.node_id in self.nodes else 0.5


def test_divergence_reason_does_not_depend_on_node_order(make_nodes) -> None:
    # Phát hiện A4: summary.json phải tất định như rounds.jsonl (REP-1), nên lý do nêu node có id nhỏ nhất.
    reasons = []
    for order in ((0, 1, 2, 3), (3, 2, 1, 0)):
        nodes = make_nodes()
        engine = engine_for([nodes[i] for i in order], NanForNodes({0, 2}))
        with pytest.raises(RunDiverged) as caught:
            engine.run_round(0)
        reasons.append(caught.value.reason)

    assert reasons[0] == reasons[1]
    assert "node 0" in reasons[0]


class CountingPolicy:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, graph: nx.Graph, node_sizes: np.ndarray) -> np.ndarray:
        self.calls += 1
        return mixing.metropolis_hastings(graph)


def test_static_topology_builds_and_checks_w_once(make_nodes, sgd_trainer) -> None:
    # Phát hiện A14: W của đồ thị tĩnh không đổi, nên không dựng lại và kiểm lại mỗi round.
    policy = CountingPolicy()
    engine = engine_for(make_nodes(), sgd_trainer(tau=1, lr=0.1), policy=policy)

    for r in range(3):
        engine.run_round(r)
        engine.weights_at(r + 5)

    assert policy.calls == 1


def test_round_results_compare_by_identity(make_nodes, sgd_trainer) -> None:
    # Phát hiện A11: hai kết quả cùng round_idx thì == phải so tới mảng W; không được ném lỗi.
    first = engine_for(make_nodes(), sgd_trainer(tau=1, lr=0.1)).run_round(0)
    twin = engine_for(make_nodes(), sgd_trainer(tau=1, lr=0.1)).run_round(0)

    assert first == first
    assert first != twin


def test_empty_tensors_are_not_mistaken_for_shared_memory(make_nodes, sgd_trainer) -> None:
    # Mọi tensor rỗng có data_ptr 0, nên phép kiểm bộ nhớ dùng chung phải bỏ qua chúng.
    nodes = make_nodes()
    for node in nodes:
        node.model.register_buffer("empty", torch.zeros(0))

    engine_for(nodes, sgd_trainer(tau=1, lr=0.1)).run_round(0)


def test_gossip_whose_union_of_rounds_never_connects_is_rejected_before_training(
    make_nodes, sgd_trainer
) -> None:
    # Phản biện vòng 2, phát hiện 1: rng_for bỏ qua chỉ số round vẫn thuần, nhưng mọi round cùng một matching,
    # nên mạng bị chia cắt mãi (INV-4: gossip phải liên thông theo hợp các round).
    trainer = sgd_trainer(tau=1, lr=0.1)
    constant = topo.gossip(N, lambda r: seeding.numpy_rng(0, Stream.GOSSIP))
    engine = engine_for(make_nodes(), trainer, schedule=constant)

    with pytest.raises(ValueError, match="hợp các round"):
        list(run_rounds(engine, 10, None))
    assert trainer.calls == []


class GlobalDrawTrainer:
    """Trainer dùng thêm RNG toàn cục, như một model có Dropout."""

    def __init__(self, inner: LocalTrainer) -> None:
        self.inner = inner

    def __call__(
        self, model: torch.nn.Module, data: NodeDataset, *, loss_scale: float, generator: torch.Generator
    ) -> float:
        torch.rand(1)
        return self.inner(model, data, loss_scale=loss_scale, generator=generator)


def test_adapt_that_uses_the_global_torch_rng_halts_the_engine(make_nodes, sgd_trainer) -> None:
    # Phát hiện 2 (vòng 2): dùng RNG toàn cục làm quỹ đạo phụ thuộc thứ tự node và mọi lời gọi khác (INV-5).
    engine = engine_for(make_nodes(), GlobalDrawTrainer(sgd_trainer(tau=1, lr=0.1)))

    with torch.random.fork_rng(devices=[]):  # trả lại RNG toàn cục cho fixture autouse
        with pytest.raises(ValueError, match="RNG toàn cục"):
            engine.run_round(0)
    with pytest.raises(RuntimeError, match="không chạy tiếp"):
        engine.run_round(0)


def test_round_weights_cannot_be_made_writable(make_nodes, sgd_trainer) -> None:
    # Phát hiện 3: W tĩnh được giữ lại cho mọi round; người nhận không được bật quyền ghi rồi sửa nó.
    engine = engine_for(make_nodes(), sgd_trainer(tau=1, lr=0.1))

    result = engine.run_round(0)

    for weights in (result.weights, engine.weights_at(1)):
        with pytest.raises(ValueError):
            weights.setflags(write=True)


def test_edge_attributes_named_weight_do_not_hide_edges(make_nodes, sgd_trainer) -> None:
    # Phát hiện 4: to_numpy_array đọc thuộc tính "weight" của cạnh; cạnh có weight = 0 vẫn là cạnh để gửi.
    graph = nx.cycle_graph(N)
    graph[0][1]["weight"] = 0
    engine = engine_for(make_nodes(), sgd_trainer(tau=1, lr=0.1), schedule=FixedGraph(graph))

    result = engine.run_round(0)

    np.testing.assert_array_equal(result.weights, mixing.metropolis_hastings(nx.cycle_graph(N)))


def test_states_cannot_be_read_after_the_engine_halts(make_nodes) -> None:
    # Phát hiện 7: sau khi dừng giữa round, state là hỗn hợp w^t và w^(t+½); không được đem đi đánh giá.
    engine = engine_for(make_nodes(), NanForNodes({1}))
    with pytest.raises(RunDiverged):
        engine.run_round(0)

    with pytest.raises(RuntimeError, match="đã dừng"):
        engine.states()


class Alternating:
    """Lịch gossip SAI có chủ đích: round chẵn có cạnh 0–1, round lẻ có cạnh 2–3; hợp không liên thông."""

    num_nodes = N
    is_static = False

    def graph_at(self, round_idx: int) -> nx.Graph:
        graph = nx.empty_graph(N)
        graph.add_edge(*((0, 1) if round_idx % 2 == 0 else (2, 3)))
        return graph


def test_engine_refuses_gossip_rounds_outside_the_checked_horizon(make_nodes, sgd_trainer) -> None:
    # Vòng 3, phát hiện 2: INV-4 của gossip phải do chính engine buộc, không tùy người gọi có dùng run_rounds.
    trainer = sgd_trainer(tau=1, lr=0.1)
    engine = engine_for(make_nodes(), trainer, schedule=Alternating())

    with pytest.raises(RuntimeError, match="check_schedule"):
        engine.run_round(0)
    with pytest.raises(ValueError, match="hợp các round"):
        engine.check_schedule(6)
    assert trainer.calls == []

    gossip = topo.gossip(N, lambda r: seeding.numpy_rng(4, Stream.GOSSIP, r))
    valid = engine_for(make_nodes(), sgd_trainer(tau=1, lr=0.1), schedule=gossip)
    valid.check_schedule(10)
    for r in range(10):
        valid.run_round(r)
    with pytest.raises(RuntimeError, match="check_schedule"):
        valid.run_round(10)


@pytest.mark.parametrize("total_rounds", [0, -1])
@pytest.mark.parametrize("kind", ["gossip", "static"])
def test_check_schedule_needs_a_positive_number_of_rounds(
    kind, total_rounds, make_nodes, sgd_trainer
) -> None:
    # Vòng 3, phát hiện 11.
    schedule = topo.gossip(N, lambda r: seeding.numpy_rng(4, Stream.GOSSIP, r)) if kind == "gossip" else None
    engine = engine_for(make_nodes(), sgd_trainer(tau=1, lr=0.1), schedule=schedule)

    with pytest.raises(ValueError, match="total_rounds"):
        engine.check_schedule(total_rounds)


class StaleTransport(InMemoryTransport):
    """Từ round 1, inbox giao snapshot của round trước, như một transport giữ nhầm bộ đệm cũ."""

    def __init__(self, num_nodes: int) -> None:
        super().__init__(num_nodes)
        self.history: dict[tuple[int, int], dict[int, Snapshot]] = {}

    def send(self, dst: int, message: Snapshot) -> None:
        super().send(dst, message)
        self.history.setdefault((message.round_idx, dst), {})[message.sender] = message

    def inbox(self, dst: int, round_idx: int) -> Mapping[int, Snapshot]:
        current = super().inbox(dst, round_idx)
        return self.history.get((round_idx - 1, dst), current) if round_idx > 0 else current


class SwappedSenders(InMemoryTransport):
    """Inbox gắn nhầm khóa: snapshot của một láng giềng nằm dưới id của láng giềng kia."""

    def inbox(self, dst: int, round_idx: int) -> Mapping[int, Snapshot]:
        items = list(super().inbox(dst, round_idx).items())
        keys = [key for key, _ in items][::-1]
        return {key: message for key, (_, message) in zip(keys, items, strict=True)}


@pytest.mark.parametrize(
    "transport_cls", [StaleTransport, SwappedSenders], ids=["stale-round", "swapped-senders"]
)
def test_combine_rejects_inbox_snapshots_from_the_wrong_round_or_sender(
    transport_cls, make_nodes, sgd_trainer
) -> None:
    # Vòng 3, phát hiện 3: engine không được trộn snapshot của round khác hay gắn sai người gửi.
    engine = engine_for(make_nodes(), sgd_trainer(tau=1, lr=0.1), transport=transport_cls(N))

    with pytest.raises(RuntimeError, match="inbox"):
        for r in range(2):
            engine.run_round(r)


def test_no_array_behind_round_weights_can_be_made_writable(make_nodes, sgd_trainer) -> None:
    # Vòng 3, phát hiện 4a: view chỉ đọc là chưa đủ, vì `.base` của nó có thể bật lại quyền ghi.
    engine = engine_for(make_nodes(), sgd_trainer(tau=1, lr=0.1))

    for weights in (engine.run_round(0).weights, engine.weights_at(1)):
        chain = weights
        while isinstance(chain, np.ndarray):
            with pytest.raises(ValueError):
                chain.setflags(write=True)
            chain = chain.base


def test_static_graph_mutated_after_round_zero_does_not_change_the_run(make_nodes, sgd_trainer) -> None:
    # Vòng 3, phát hiện 4b: engine giữ bản sao đóng băng của đồ thị tĩnh, đồng bộ với W đã kiểm.
    graph = nx.cycle_graph(N)
    transport = InMemoryTransport(N)
    engine = engine_for(
        make_nodes(), sgd_trainer(tau=1, lr=0.1), schedule=FixedGraph(graph), transport=transport
    )
    twin = engine_for(make_nodes(), sgd_trainer(tau=1, lr=0.1), schedule=FixedGraph(nx.cycle_graph(N)))
    engine.run_round(0)
    twin.run_round(0)

    graph.add_edge(0, 2)
    engine.run_round(1)
    twin.run_round(1)

    assert transport.counters().messages_cum == 2 * 8
    for a, b in zip(engine.states(), twin.states(), strict=True):
        for name in a:
            assert torch.equal(a[name], b[name])


def test_impure_schedule_is_detected_when_the_round_runs(make_nodes, sgd_trainer) -> None:
    # Vòng 3, phát hiện 9: rng_for dùng chung một Generator làm graph_at phụ thuộc số lần gọi (G18).
    # Với N = 4 chỉ có 3 matching, nên hai lần rút liền nhau đôi khi trùng; chạy nhiều round để chắc có lệch.
    shared = seeding.numpy_rng(9, Stream.GOSSIP)
    engine = engine_for(make_nodes(), sgd_trainer(tau=1, lr=0.1), schedule=topo.gossip(N, lambda r: shared))
    engine.check_schedule(20)

    with pytest.raises(RuntimeError, match="thuần"):
        for r in range(20):
            engine.run_round(r)

"""Test tích hợp cho Observer và kiểm hữu hạn của engine (T10c; KH §3.4 bước 4, §3.5, §4.5; ADR-007;
SPEC §9.2: T-INV-11, T-INV-12; INV-5).
"""

import math
from collections.abc import Callable

import networkx as nx
import numpy as np
import pytest
import torch
from sklearn.metrics import recall_score

from dfl import seeding
from dfl.contracts import (
    CommCounters,
    EvalResult,
    Evaluator,
    LocalTrainer,
    NodeDataset,
    RoundRecord,
    Snapshot,
    StateDict,
    TopologySchedule,
)
from dfl.network import mixing
from dfl.network import topology as topo
from dfl.network.transport import InMemoryTransport
from dfl.seeding import Stream
from dfl.sim.engine import Engine, RoundResult, RunDiverged
from dfl.sim.node import Node
from dfl.sim.observer import Observer, run_rounds

N = 4


def mh(graph: nx.Graph, node_sizes: np.ndarray) -> np.ndarray:
    return mixing.metropolis_hastings(graph)


def gossip_rng(seed: int) -> Callable[[int], np.random.Generator]:
    return lambda r: seeding.numpy_rng(seed, Stream.GOSSIP, r)


def rms(states: list[StateDict]) -> float:
    """Oracle độc lập cho Ξ: khoảng cách RMS tới trung bình, float64."""
    x = np.stack(
        [
            np.concatenate([t.double().numpy().ravel() for t in s.values() if t.is_floating_point()])
            for s in states
        ]
    )
    return float(np.sqrt(((x - x.mean(axis=0)) ** 2).sum(axis=1).mean()))


def assert_states_equal(a: StateDict, b: StateDict) -> None:
    assert a.keys() == b.keys()
    for name in a:
        assert torch.equal(a[name], b[name]), name


def test_t_inv_11_fc_mh_nodes_share_one_model_so_acc_std_is_zero_and_acc_local_uses_one_recall_vector(
    make_nodes, sgd_trainer, make_evaluator, eval_set
) -> None:
    # Arrange: mỗi node có 3 lớp, nên P_i khác hẳn nhau; đánh giá mỗi round.
    nodes = make_nodes()
    evaluator = make_evaluator(nodes)
    engine = Engine(
        nodes=nodes,
        trainer=sgd_trainer(tau=2, lr=0.1),
        schedule=topo.fully_connected(N),
        mixing=mh,
        transport=InMemoryTransport(N),
    )
    observer = Observer(evaluator, tau=2, total_rounds=2, eval_every_steps=None)

    # Act
    records = list(run_rounds(engine, 2, observer))

    # Assert
    states = engine.states()
    for state in states[1:]:
        assert_states_equal(state, states[0])
    # Oracle: một vector recall duy nhất, tính bằng sklearn từ model của node 0 (đang giữ states[0]).
    with torch.no_grad():
        predictions = nodes[0].model(eval_set.x).argmax(dim=1).numpy()
    recall = recall_score(
        eval_set.y.numpy(), predictions, labels=list(range(10)), average=None, zero_division=0
    )
    last = records[-1]
    assert len(set(last["acc_per_node"])) == 1
    assert last["acc_std"] == 0.0
    for node, acc_local in zip(nodes, last["acc_local_per_node"], strict=True):
        p_i = np.bincount(node.data.y.numpy(), minlength=10) / len(node.data.y)
        assert acc_local == pytest.approx(float(p_i @ recall), abs=1e-12)
    # Không suông: acc^loc vẫn khác nhau giữa các node dù model trùng nhau (G15); drift > 0, consensus = 0.
    assert max(last["acc_local_per_node"]) - min(last["acc_local_per_node"]) > 0.05
    assert last["drift_pre_combine"] > 0
    assert last["consensus_distance"] <= 1e-6 * last["drift_pre_combine"]


class ExtraCallsSchedule:
    """Bọc một lịch: trước mỗi lần gọi thật, gọi thêm graph_at ở ba round khác (T-INV-12)."""

    def __init__(self, inner: topo.GossipSchedule) -> None:
        self.inner = inner
        self.num_nodes = inner.num_nodes
        self.is_static = inner.is_static
        self.extra_calls = 0

    def graph_at(self, round_idx: int) -> nx.Graph:
        for other in (round_idx + 1, 0, round_idx + 7):
            self.inner.graph_at(other)
            self.extra_calls += 1
        return self.inner.graph_at(round_idx)


class StatefulGossip:
    """Lịch SAI có chủ đích: lần gọi thứ k trả matching thứ k mod 3 của K_4, bất kể round (G18).

    Hai lần gọi liên tiếp luôn cho hai matching khác nhau, nên engine phải phát hiện được ngay.
    """

    num_nodes = N
    is_static = False
    MATCHINGS = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))

    def __init__(self) -> None:
        self.calls = 0

    def graph_at(self, round_idx: int) -> nx.Graph:
        graph = nx.empty_graph(N)
        graph.add_edges_from(self.MATCHINGS[self.calls % 3])
        self.calls += 1
        return graph


def gossip_trajectory(
    make_nodes: Callable[..., list[Node]],
    sgd_trainer: Callable[..., LocalTrainer],
    make_evaluator: Callable[..., Evaluator],
    schedule: TopologySchedule,
    *,
    observe: bool,
) -> tuple[tuple[StateDict, ...], list[RoundRecord]]:
    nodes = make_nodes()
    engine = Engine(
        nodes=nodes,
        trainer=sgd_trainer(tau=2, lr=0.1, mu=0.1),
        schedule=schedule,
        mixing=mh,
        transport=InMemoryTransport(N),
    )
    observer = (
        Observer(make_evaluator(nodes), tau=2, total_rounds=4, eval_every_steps=None) if observe else None
    )
    records = list(run_rounds(engine, 4, observer))
    return engine.states(), records


def test_t_inv_12_disabling_observer_or_extra_graph_at_calls_keeps_trajectory_bitwise(
    make_nodes, sgd_trainer, make_evaluator
) -> None:
    # Arrange
    wrapped = ExtraCallsSchedule(topo.gossip(N, gossip_rng(5)))

    # Act: A có Observer; B tắt Observer; C có Observer và lịch bị gọi thêm.
    states_a, records_a = gossip_trajectory(
        make_nodes, sgd_trainer, make_evaluator, topo.gossip(N, gossip_rng(5)), observe=True
    )
    states_b, records_b = gossip_trajectory(
        make_nodes, sgd_trainer, make_evaluator, topo.gossip(N, gossip_rng(5)), observe=False
    )
    states_c, records_c = gossip_trajectory(make_nodes, sgd_trainer, make_evaluator, wrapped, observe=True)

    # Assert
    for a, b, c in zip(states_a, states_b, states_c, strict=True):
        assert_states_equal(a, b)
        assert_states_equal(a, c)
    assert records_a == records_c
    assert len(records_a) == 4 and records_b == []
    assert wrapped.extra_calls >= 12


def test_stateful_schedule_is_refused_before_it_can_change_the_trajectory(
    make_nodes, sgd_trainer, make_evaluator
) -> None:
    # Đối chứng dương cho T-INV-12: lịch có trạng thái ẩn thì quỹ đạo phụ thuộc số lần gọi graph_at. Engine
    # kiểm tính thuần ở mỗi round (gọi graph_at hai lần), nên lượt dừng ngay ở round 0, trước khi Adapt chạy.
    trainer = sgd_trainer(tau=2, lr=0.1)
    engine = Engine(
        nodes=make_nodes(),
        trainer=trainer,
        schedule=StatefulGossip(),
        mixing=mh,
        transport=InMemoryTransport(N),
    )

    with pytest.raises(RuntimeError, match="thuần"):
        list(run_rounds(engine, 4, None))
    assert trainer.calls == []


def test_observer_evaluates_every_100_local_steps_and_the_final_round(make_nodes, make_evaluator) -> None:
    evaluator = make_evaluator(make_nodes())

    every_100 = Observer(evaluator, tau=25, total_rounds=10, eval_every_steps=100)
    every_round = Observer(evaluator, tau=25, total_rounds=10, eval_every_steps=None)

    assert [r for r in range(1, 11) if every_100.is_due(r)] == [4, 8, 10]
    assert [r for r in range(1, 11) if every_round.is_due(r)] == list(range(1, 11))
    with pytest.raises(ValueError, match="chia hết"):
        Observer(evaluator, tau=30, total_rounds=10, eval_every_steps=100)


def test_run_rounds_records_exactly_the_scheduled_rounds(make_nodes, sgd_trainer, make_evaluator) -> None:
    nodes = make_nodes()
    engine = Engine(
        nodes=nodes,
        trainer=sgd_trainer(tau=5, lr=0.1),
        schedule=topo.ring(N),
        mixing=mh,
        transport=InMemoryTransport(N),
    )
    observer = Observer(make_evaluator(nodes), tau=5, total_rounds=5, eval_every_steps=10)

    records = list(run_rounds(engine, 5, observer))

    assert [(r["round"], r["local_steps"]) for r in records] == [(2, 10), (4, 20), (5, 25)]


def test_round_record_has_every_contract_field_and_no_nan(make_nodes, sgd_trainer, make_evaluator) -> None:
    # Arrange
    nodes = make_nodes()
    transport = InMemoryTransport(N)
    engine = Engine(
        nodes=nodes, trainer=sgd_trainer(tau=2, lr=0.1), schedule=topo.ring(N), mixing=mh, transport=transport
    )
    observer = Observer(make_evaluator(nodes), tau=2, total_rounds=3, eval_every_steps=None)

    for t in range(3):
        # Act
        result = engine.run_round(t)
        record = observer.record(result, engine.states(), transport.counters())

        # Assert
        assert set(record) == RoundRecord.__required_keys__
        assert (record["round"], record["local_steps"]) == (t + 1, 2 * (t + 1))
        assert len(record["acc_per_node"]) == len(record["acc_local_per_node"]) == N
        counters = transport.counters()
        assert (record["messages_cum"], record["bytes_cum"], record["max_node_bytes_cum"]) == (
            counters.messages_cum,
            counters.bytes_cum,
            counters.max_node_bytes_cum,
        )
        assert record["train_loss_mean"] == pytest.approx(np.mean(result.train_losses), rel=1e-12)
        assert record["acc_mean"] == pytest.approx(np.mean(record["acc_per_node"]), rel=1e-12)
        assert record["acc_min"] == min(record["acc_per_node"])
        assert record["acc_max"] == max(record["acc_per_node"])
        numbers = [v for v in record.values() if isinstance(v, float)]
        numbers += record["acc_per_node"] + record["acc_local_per_node"]
        assert all(math.isfinite(v) for v in numbers)


def test_drift_and_consensus_are_rms_of_pre_and_post_combine_models(
    make_nodes, sgd_trainer, make_evaluator
) -> None:
    nodes = make_nodes()
    transport = InMemoryTransport(N)
    engine = Engine(
        nodes=nodes, trainer=sgd_trainer(tau=2, lr=0.1), schedule=topo.ring(N), mixing=mh, transport=transport
    )
    observer = Observer(make_evaluator(nodes), tau=2, total_rounds=2, eval_every_steps=None)

    for t in range(2):
        result = engine.run_round(t)
        states = engine.states()
        record = observer.record(result, states, transport.counters())

        drift = rms([s.state for s in result.snapshots])
        assert record["drift_pre_combine"] == pytest.approx(drift, rel=1e-9)
        assert record["consensus_distance"] == pytest.approx(rms(list(states)), rel=1e-9)
        assert 0 < record["consensus_distance"] <= record["drift_pre_combine"] * (1 + 1e-9)


class FailAt:
    """Bọc trainer: ở (node, round) chỉ định thì trả mất mát NaN, hoặc làm một tham số thành vô cực."""

    def __init__(self, trainer: LocalTrainer, *, node: int, round_idx: int, kind: str) -> None:
        self.trainer = trainer
        self.node, self.round_idx, self.kind = node, round_idx, kind
        self.calls: dict[int, int] = {}

    def __call__(
        self, model: torch.nn.Module, data: NodeDataset, *, loss_scale: float, generator: torch.Generator
    ) -> float:
        round_idx = self.calls.get(data.node_id, 0)
        self.calls[data.node_id] = round_idx + 1
        loss = self.trainer(model, data, loss_scale=loss_scale, generator=generator)
        if (data.node_id, round_idx) != (self.node, self.round_idx):
            return loss
        if self.kind == "nan-loss":
            return math.nan
        with torch.no_grad():
            model.fc2.bias[0] = math.inf
        return loss


@pytest.mark.parametrize(
    ("kind", "node", "round_idx", "eval_every_steps"),
    [("nan-loss", 1, 1, None), ("inf-parameter", 3, 2, 8)],
    ids=["nan-loss", "inf-parameter"],
)
def test_non_finite_loss_or_parameter_stops_the_run_at_that_round(
    kind, node, round_idx, eval_every_steps, make_nodes, sgd_trainer, make_evaluator
) -> None:
    # Arrange: ca thứ hai chỉ đánh giá ở round 4, 8, nên engine phải tự phát hiện, không đợi tới mốc đánh giá.
    nodes = make_nodes()
    engine = Engine(
        nodes=nodes,
        trainer=FailAt(sgd_trainer(tau=2, lr=0.1), node=node, round_idx=round_idx, kind=kind),
        schedule=topo.ring(N),
        mixing=mh,
        transport=InMemoryTransport(N),
    )
    observer = Observer(make_evaluator(nodes), tau=2, total_rounds=8, eval_every_steps=eval_every_steps)
    records = []

    # Act
    with pytest.raises(RunDiverged) as caught:
        for record in run_rounds(engine, 8, observer):
            records.append(record)

    # Assert: dừng ở đúng round bị lỗi, nêu node gây lỗi; mọi bản ghi đã ghi đều trước round đó.
    assert caught.value.round_idx == round_idx
    assert caught.value.diverged_at_round == round_idx + 1
    assert f"node {node}" in caught.value.reason
    assert all(r["round"] <= round_idx for r in records)
    assert [r["round"] for r in records] == ([1] if eval_every_steps is None else [])
    # Model đã ở trạng thái dở dang, nên engine không cho chạy tiếp.
    with pytest.raises(RuntimeError, match="phân kỳ"):
        engine.run_round(round_idx)


@pytest.mark.parametrize(
    "kwargs",
    [{"tau": 0}, {"total_rounds": 0}, {"eval_every_steps": 0}],
    ids=["tau", "total-rounds", "eval-every"],
)
def test_observer_rejects_invalid_schedule_parameters(kwargs, make_nodes, make_evaluator) -> None:
    options = {"tau": 2, "total_rounds": 4, "eval_every_steps": None, **kwargs}

    with pytest.raises(ValueError):
        Observer(make_evaluator(make_nodes()), **options)


def test_run_rounds_rejects_an_observer_built_for_another_number_of_rounds(
    make_nodes, sgd_trainer, make_evaluator
) -> None:
    nodes = make_nodes()
    engine = Engine(
        nodes=nodes,
        trainer=sgd_trainer(tau=1, lr=0.1),
        schedule=topo.ring(N),
        mixing=mh,
        transport=InMemoryTransport(N),
    )
    observer = Observer(make_evaluator(nodes), tau=1, total_rounds=3, eval_every_steps=None)

    with pytest.raises(ValueError, match="round"):
        list(run_rounds(engine, 5, observer))


class FixedEvaluator:
    """Evaluator giả trả cùng accuracy 0.1 cho 20 node: numpy.std ra 1.4e-17 ở đây, pstdev ra đúng 0."""

    def evaluate(self, states: list[StateDict]) -> EvalResult:
        same = (0.1,) * len(states)
        return EvalResult(acc=same, acc_local=same, acc_eff=same, f1_macro=same, acc_avg_model=0.1)


def test_std_across_nodes_is_exactly_zero_when_every_node_scores_the_same() -> None:
    # SAN-5 đòi acc_std = 0 đúng tuyệt đối trên FC + MH với N = 20 (SPEC §9.3).
    n = 20
    states = [{"w": torch.zeros(1)} for _ in range(n)]
    result = RoundResult(
        round_idx=0,
        weights=np.full((n, n), 1 / n),
        train_losses=(0.5,) * n,
        snapshots=tuple(Snapshot(i, 0, {"w": torch.zeros(1)}, 4) for i in range(n)),
    )
    observer = Observer(FixedEvaluator(), tau=1, total_rounds=1, eval_every_steps=None)

    record = observer.record(result, states, CommCounters(0, 0, 0))

    assert record["acc_std"] == record["acc_local_std"] == record["f1_macro_std"] == 0.0


class PerNodeEvaluator:
    """Evaluator giả: mỗi độ đo một bộ giá trị theo node, khác các bộ kia ở trung bình, độ lệch và min."""

    def evaluate(self, states: list[StateDict]) -> EvalResult:
        return EvalResult(
            acc=(0.1, 0.2, 0.3, 0.6),
            acc_local=(0.5, 0.4, 0.9, 0.2),
            acc_eff=(0.7, 0.1, 0.3, 0.25),
            f1_macro=(0.05, 0.15, 0.35, 0.95),
            acc_avg_model=0.42,
        )


def test_round_record_aggregates_are_population_statistics_of_the_matching_per_node_values() -> None:
    # REVIEW-C3-P3.7 P0-3: rounds.jsonl không ghi F1 hay acc^eff theo node, nên analysis không tính lại được
    # các trường này; trường nào lấy nhầm bộ giá trị hay nhầm phép thống kê đều lệch khỏi giá trị tính tay.
    states = [{"w": torch.zeros(1)} for _ in range(N)]
    result = RoundResult(
        round_idx=0,
        weights=np.full((N, N), 1 / N),
        train_losses=(0.5,) * N,
        snapshots=tuple(Snapshot(i, 0, {"w": torch.zeros(1)}, 4) for i in range(N)),
    )
    observer = Observer(PerNodeEvaluator(), tau=1, total_rounds=1, eval_every_steps=None)

    record = observer.record(result, states, CommCounters(0, 0, 0))

    # Tính tay; độ lệch chuẩn theo ddof = 0 (contracts.RoundRecord), ví dụ acc: (0.04 + 0.01 + 0 + 0.09) / 4.
    expected = {
        "acc_mean": 0.3,
        "acc_std": math.sqrt(0.035),
        "acc_min": 0.1,
        "acc_max": 0.6,
        "acc_local_mean": 0.5,
        "acc_local_std": math.sqrt(0.065),
        "acc_local_min": 0.2,
        "f1_macro_mean": 0.375,
        "f1_macro_std": math.sqrt(0.121875),
        "f1_macro_min": 0.05,
        "acc_eff_mean": 0.3375,
        "acc_avg_model": 0.42,
    }
    assert {key: record[key] for key in expected} == pytest.approx(expected, rel=1e-12)
    assert record["acc_per_node"] == [0.1, 0.2, 0.3, 0.6]
    assert record["acc_local_per_node"] == [0.5, 0.4, 0.9, 0.2]

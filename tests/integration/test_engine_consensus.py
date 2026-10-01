"""T-INV-1 qua engine (T10b; KH §3.1, §5.2; SPEC §9.2): với lr = 0, engine chỉ còn trộn, nên X_t = W^t X_0.

Ξ tính ngay trong test bằng công thức RMS, Ξ_t = sqrt(1/N Σ_i ‖w_i^t − w̄_t‖²), làm oracle độc lập
(không import metrics/consensus.py). Model float64: với float32, sai số làm tròn đã vượt cận
(1 + 1e-6)·ρ^t·Ξ_0 + 1e-12 từ t ≈ 7 trên ring(4) (đo ở H P3.3 bước 1), còn dạng "+1e-6" thì không còn kiểm
tốc độ co khi ρ^t·Ξ_0 < 1e-6. Gossip không có cận một phía cho Ξ_t (G18): chỉ kiểm từng W và E[W].
"""

from collections.abc import Callable

import networkx as nx
import numpy as np
import pytest
import torch

from dfl import seeding
from dfl.contracts import LocalTrainer, StateDict, TopologySchedule
from dfl.network import mixing
from dfl.network import topology as topo
from dfl.network.transport import InMemoryTransport
from dfl.seeding import Stream
from dfl.sim.engine import Engine
from dfl.sim.node import Node

N = 4
ROUNDS = 20


def mh(graph: nx.Graph, node_sizes: np.ndarray) -> np.ndarray:
    return mixing.metropolis_hastings(graph)


def gossip_rng(seed: int) -> Callable[[int], np.random.Generator]:
    return lambda r: seeding.numpy_rng(seed, Stream.GOSSIP, r)


def flat(states: tuple[StateDict, ...]) -> np.ndarray:
    """Ma trận N × P: mỗi hàng là mọi tensor float của một node, nối theo thứ tự khóa."""
    return np.stack(
        [
            np.concatenate([t.numpy().ravel() for t in state.values() if t.is_floating_point()])
            for state in states
        ]
    )


def xi(x: np.ndarray) -> float:
    return float(np.sqrt(((x - x.mean(axis=0)) ** 2).sum(axis=1).mean()))


def lr0_engine(
    make_nodes: Callable[..., list[Node]],
    sgd_trainer: Callable[..., LocalTrainer],
    schedule: TopologySchedule,
) -> Engine:
    # Khởi tạo khác nhau ở mỗi node, lr = 0: Adapt không đổi model, chỉ còn bước trộn.
    return Engine(
        nodes=make_nodes(dtype=torch.float64, shared_init=False),
        trainer=sgd_trainer(tau=1, lr=0.0),
        schedule=schedule,
        mixing=mh,
        transport=InMemoryTransport(N),
    )


STATIC = [
    topo.ring(N),
    topo.erdos_renyi(N, 0.5, seeding.numpy_rng(0, Stream.GRAPH)),  # cạnh 01, 02, 03, 13; ρ = 0.75
    topo.fully_connected(N),
]
STATIC_IDS = ["ring", "er-paw", "fc"]


@pytest.mark.parametrize("schedule", STATIC, ids=STATIC_IDS)
def test_t_inv_1_static_consensus_contracts_by_rho_and_mean_is_preserved(
    schedule, make_nodes, sgd_trainer
) -> None:
    # Arrange: ρ lấy từ ma trận MH do test tự dựng, không từ W của engine.
    engine = lr0_engine(make_nodes, sgd_trainer, schedule)
    w = mixing.metropolis_hastings(schedule.graph_at(0))
    rho = float(np.abs(np.linalg.eigvalsh(w - np.full((N, N), 1 / N))).max())
    x0 = flat(engine.states())
    xi0 = xi(x0)

    for t in range(1, ROUNDS + 1):
        # Act
        result = engine.run_round(t - 1)
        x = flat(engine.states())

        # Assert
        np.testing.assert_array_equal(result.weights, w)
        assert xi(x) <= rho**t * xi0 * (1 + 1e-6) + 1e-12, t
        np.testing.assert_allclose(x.mean(axis=0), x0.mean(axis=0), rtol=0, atol=1e-12)
    assert xi0 > 1  # không suông: các node xuất phát xa nhau


@pytest.mark.parametrize("schedule", STATIC, ids=STATIC_IDS)
def test_lr0_static_trajectory_is_w_to_the_t_times_initial_models(schedule, make_nodes, sgd_trainer) -> None:
    # Phép kiểm hai phía: bắt được W sai mà vẫn co (ví dụ uniform thay MH), điều cận ρ^t một phía bỏ sót.
    engine = lr0_engine(make_nodes, sgd_trainer, schedule)
    w = mixing.metropolis_hastings(schedule.graph_at(0))
    x0 = flat(engine.states())

    for t in range(1, ROUNDS + 1):
        engine.run_round(t - 1)

        np.testing.assert_allclose(
            flat(engine.states()), np.linalg.matrix_power(w, t) @ x0, rtol=0, atol=1e-12
        )


def test_t_inv_1_gossip_each_mixing_matrix_is_a_symmetric_idempotent_matching(
    make_nodes, sgd_trainer
) -> None:
    # Arrange: một lịch dựng độc lập với cùng seed cho đồ thị mà mỗi round phải dùng.
    engine = lr0_engine(make_nodes, sgd_trainer, topo.gossip(N, gossip_rng(1)))
    independent = topo.gossip(N, gossip_rng(1))
    engine.check_schedule(50)

    for r in range(50):
        # Act
        x_pre = flat(engine.states())
        result = engine.run_round(r)
        w = result.weights

        # Assert
        np.testing.assert_array_equal(w, w.T)
        np.testing.assert_allclose(w.sum(axis=0), 1.0, rtol=0, atol=1e-12)
        np.testing.assert_allclose(w.sum(axis=1), 1.0, rtol=0, atol=1e-12)
        np.testing.assert_allclose(w @ w, w, rtol=0, atol=1e-12)
        assert (w >= 0).all()
        # Giá ngoài đường chéo trùng đúng cạnh của graph_at(r); chỉ đòi "⊆" thì W = I cũng qua.
        support = {frozenset((i, j)) for i in range(N) for j in range(N) if i != j and w[i, j] != 0}
        assert support == {frozenset(edge) for edge in independent.graph_at(r).edges()}
        np.testing.assert_array_equal(engine.weights_at(r), w)
        np.testing.assert_allclose(flat(engine.states()), w @ x_pre, rtol=0, atol=1e-12)


def test_t_inv_1_gossip_expected_mixing_matrix_matches_closed_form(make_nodes, sgd_trainer) -> None:
    # E[W] = ½ I + (J − I)/(2(N − 1)) với N chẵn (KH §3.3). Lấy qua weights_at, chính hàm run_round dùng
    # (test trên buộc hai thứ trùng nhau), vì chạy 10⁴ round qua engine mất khoảng 11 s.
    engine = lr0_engine(make_nodes, sgd_trainer, topo.gossip(N, gossip_rng(2)))
    rounds = 10_000

    mean = sum(engine.weights_at(r) for r in range(rounds)) / rounds

    expected = 0.5 * np.eye(N) + (np.ones((N, N)) - np.eye(N)) / (2 * (N - 1))
    p = 1 / (N - 1)
    se = 0.5 * np.sqrt(p * (1 - p) / rounds)
    np.testing.assert_array_equal(np.diag(mean), 0.5)  # N chẵn: mọi node luôn được ghép
    off_diagonal = ~np.eye(N, dtype=bool)
    assert np.abs(mean - expected)[off_diagonal].max() <= 4 * se

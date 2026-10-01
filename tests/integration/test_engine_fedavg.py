"""T-INV-3 qua engine (T10b; KH §3.4 B3; SPEC §9.2): FC + size-weighted và đường server tương đương FedAvg.

FedAvg tham chiếu viết ngay trong test (SPEC §7): mọi client xuất phát từ cùng model toàn cục, chạy cùng
trainer với cùng luồng batch của mình, rồi model toàn cục mới là Σ_k (n_k/n) w_k.
"""

from collections.abc import Callable, Sequence

import networkx as nx
import numpy as np
import pytest
import torch

from dfl.contracts import LocalTrainer, StateDict
from dfl.network import mixing
from dfl.network import topology as topo
from dfl.network.transport import InMemoryTransport
from dfl.sim.engine import Engine
from dfl.sim.node import Node

N = 4
ROUNDS = 3
UNEQUAL_SIZES = (64, 96, 128, 200)
EQUAL_SIZES = (96, 96, 96, 96)


def mh(graph: nx.Graph, node_sizes: np.ndarray) -> np.ndarray:
    return mixing.metropolis_hastings(graph)


def size_weighted(graph: nx.Graph, node_sizes: np.ndarray) -> np.ndarray:
    return mixing.size_weighted(graph, node_sizes)


def fedavg_reference(nodes: Sequence[Node], trainer: LocalTrainer, rounds: int) -> list[StateDict]:
    """Model toàn cục sau mỗi round của FedAvg có server (McMahan et al. 2017), trọng số n_k/n."""
    sizes = np.array([len(node.data.y) for node in nodes], dtype=float)
    fractions = [float(f) for f in sizes / sizes.sum()]
    global_state = {name: t.clone() for name, t in nodes[0].model.state_dict().items()}
    history = []
    for _ in range(rounds):
        local = []
        for node in nodes:
            node.model.load_state_dict(global_state)
            trainer(node.model, node.data, loss_scale=1.0, generator=node.generator)
            local.append({name: t.detach().clone() for name, t in node.model.state_dict().items()})
        global_state = {
            name: sum(f * state[name] for f, state in zip(fractions, local, strict=True))
            for name in global_state
        }
        history.append(global_state)
    return history


def fc_engine(policy: Callable[[nx.Graph, np.ndarray], np.ndarray]) -> Callable[..., Engine]:
    def build(nodes: list[Node], trainer: LocalTrainer) -> Engine:
        return Engine(
            nodes=nodes,
            trainer=trainer,
            schedule=topo.fully_connected(N),
            mixing=policy,
            transport=InMemoryTransport(N),
        )

    return build


def server_engine(nodes: list[Node], trainer: LocalTrainer) -> Engine:
    return Engine(
        nodes=nodes, trainer=trainer, transport=InMemoryTransport(N, with_server=True), server_path=True
    )


def run(engine: Engine, rounds: int) -> list[tuple[StateDict, ...]]:
    trajectory = []
    for r in range(rounds):
        engine.run_round(r)
        trajectory.append(engine.states())
    return trajectory


@pytest.mark.parametrize(
    ("sizes", "build"),
    [(UNEQUAL_SIZES, fc_engine(size_weighted)), (UNEQUAL_SIZES, server_engine), (EQUAL_SIZES, fc_engine(mh))],
    ids=["fc-size-weighted", "server-path", "fc-mh-equal-sizes"],
)
def test_t_inv_3_fc_and_server_path_match_reference_fedavg(sizes, build, make_nodes, sgd_trainer) -> None:
    # Arrange: hai bộ node giống hệt nhau (cùng w⁰, dữ liệu, luồng batch), một cho FedAvg, một cho engine.
    reference = fedavg_reference(make_nodes(sizes), sgd_trainer(tau=3, lr=0.1), ROUNDS)
    engine = build(make_nodes(sizes), sgd_trainer(tau=3, lr=0.1))

    # Act
    trajectory = run(engine, ROUNDS)

    # Assert: mọi node khớp model toàn cục của FedAvg ở mọi round.
    for states, expected in zip(trajectory, reference, strict=True):
        for state in states:
            for name in expected:
                torch.testing.assert_close(state[name], expected[name], rtol=0, atol=1e-6)


def test_fc_mh_with_unequal_sizes_differs_from_reference_fedavg(make_nodes, sgd_trainer) -> None:
    # Đối chứng dương: MH tối ưu (1/N) Σ F_i, không phải Σ (n_k/n) F_k (G19), nên khi kích thước lệch thì
    # phải lệch FedAvg; nếu không, so sánh ở test trên không phân biệt được trọng số.
    reference = fedavg_reference(make_nodes(UNEQUAL_SIZES), sgd_trainer(tau=3, lr=0.1), ROUNDS)

    states = run(fc_engine(mh)(make_nodes(UNEQUAL_SIZES), sgd_trainer(tau=3, lr=0.1)), ROUNDS)[-1]

    gap = max((states[0][name] - reference[-1][name]).abs().max().item() for name in reference[-1])
    assert gap > 1e-3

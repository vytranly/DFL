"""Engine ATC đồng bộ, một tiến trình (module sim; KH §3.4, §4.3; ADR-002, ADR-005, ADR-006; INV-1…INV-4).

Một round t gồm bốn pha, pha sau chỉ bắt đầu khi pha trước xong ở mọi node:
1. Adapt: LocalTrainer chạy τ bước trên model của node, xuất phát từ w_i^t (nên anchor của prox là w_i^t).
2. Gửi: mỗi node tạo đúng một snapshot w_i^(t+½) và gửi chính nó tới mọi láng giềng của G_t qua Transport.
3. Combine: mỗi node tổng hợp từ snapshot của chính nó và inbox của nó vào bộ đệm mới (node.combine).
4. Ghi: chỉ khi mọi node đã tổng hợp xong mới nạp kết quả vào model (G2, G22).
Engine không tính đại lượng toàn cục nào để đưa vào huấn luyện (INV-1) và không tự đếm byte (INV-2).
Engine duyệt node theo đúng thứ tự của dãy `nodes` được giao; kết quả không được phụ thuộc thứ tự đó
(T-INV-4). G_t lấy từ `graph_at(t)` ở mọi round, nên topology tĩnh và gossip đi cùng một đường; node gossip
không được ghép vẫn chạy Adapt và giữ model của mình (W_ii = 1).

Đường server (B3/E1c, `server_path=True`): server ảo id N nhận N snapshot qua Transport, tổng hợp theo n_k/n
rồi gửi lại một snapshot cho cả N node, nên CommLedger đếm 2N message mỗi round (ADR-005, SPEC REQ-05).

torch.nn.Module.state_dict ("contains references to the module's parameters and buffers") và
load_state_dict ("Copy parameters and buffers from state_dict into this module"):
https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.state_dict
https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict
numpy.ndarray.setflags (W của round chỉ đọc): https://numpy.org/doc/2.5/reference/generated/numpy.ndarray.setflags.html
networkx.Graph.has_edge:
https://networkx.org/documentation/networkx-3.7/reference/classes/generated/networkx.Graph.has_edge.html
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import networkx as nx
import numpy as np

from dfl.contracts import LocalTrainer, MixingPolicy, Snapshot, StateDict, TopologySchedule, Transport
from dfl.sim.node import Node, combine, make_snapshot, weighted_sum

# Sai số cho phép của tổng hàng: MH, uniform và size-weighted cho tổng lệch 1 cỡ vài ulp.
ROW_SUM_ATOL = 1e-12


@dataclass(frozen=True, slots=True)
class RoundResult:
    """Những gì Observer cần từ một round; mọi dãy xếp theo id node.

    `weights` là W^(t) thực dùng (chỉ đọc; ở đường server là ma trận hiệu dụng với mọi hàng bằng n_k/n);
    `snapshots` là đúng các snapshot w^(t+½) đã gửi, dùng để đo drift trước tổng hợp.
    """

    round_idx: int
    weights: np.ndarray
    train_losses: tuple[float, ...]
    snapshots: tuple[Snapshot, ...]


def _check_inv_4(weights: np.ndarray, graph: nx.Graph, num_nodes: int) -> None:
    if weights.shape != (num_nodes, num_nodes):
        raise ValueError(f"W phải có shape ({num_nodes}, {num_nodes}), nhận {weights.shape}")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("W phải hữu hạn và không âm (INV-4)")
    if not np.allclose(weights.sum(axis=1), 1.0, rtol=0.0, atol=ROW_SUM_ATOL):
        raise ValueError("tổng mỗi hàng của W phải bằng 1 (INV-4)")
    rows, cols = np.nonzero(weights)
    off_graph = [
        (int(i), int(j)) for i, j in zip(rows, cols, strict=True) if i != j and not graph.has_edge(i, j)
    ]
    if off_graph:
        raise ValueError(f"W khác 0 ngoài cạnh của đồ thị round này, ví dụ tại {off_graph[:3]} (INV-4)")


class Engine:
    def __init__(
        self,
        *,
        nodes: Sequence[Node],
        trainer: LocalTrainer,
        transport: Transport,
        schedule: TopologySchedule | None = None,
        mixing: MixingPolicy | None = None,
        server_path: bool = False,
    ) -> None:
        num_nodes = len(nodes)
        if sorted(node.node_id for node in nodes) != list(range(num_nodes)):
            raise ValueError("node_id của các node phải là một hoán vị của 0..N-1")
        if server_path and (schedule is not None or mixing is not None):
            raise ValueError(
                "đường server không nhận schedule hay mixing: server ảo luôn tổng hợp theo n_k/n"
            )
        if not server_path and (schedule is None or mixing is None):
            raise ValueError("cần schedule và mixing, trừ khi dùng đường server (server_path=True)")
        if schedule is not None and schedule.num_nodes != num_nodes:
            raise ValueError(
                f"lịch topology có {schedule.num_nodes} node, nhưng engine nhận {num_nodes} node"
            )
        self._nodes = tuple(nodes)
        self._by_id = tuple(sorted(nodes, key=lambda node: node.node_id))
        self._sizes = np.array([len(node.data.y) for node in self._by_id])
        self._sizes.setflags(write=False)
        self._trainer = trainer
        self._transport = transport
        self._schedule = schedule
        self._mixing = mixing
        self._server_path = server_path
        sizes = self._sizes.astype(float)
        self._server_weights = sizes / sizes.sum()
        # Ở đường server, node chỉ lấy snapshot của server ảo (id N): hàng W dài N + 1 với W[i, N] = 1.
        self._from_server = np.zeros(num_nodes + 1)
        self._from_server[num_nodes] = 1.0
        self._next_round = 0

    def weights_at(self, round_idx: int) -> np.ndarray:
        """W^(t) của round t (chỉ đọc), đã kiểm INV-4; hàm thuần của (seed, t), cũng là hàm run_round gọi."""
        return self._plan(round_idx)[1]

    def _plan(self, round_idx: int) -> tuple[nx.Graph | None, np.ndarray]:
        num_nodes = len(self._nodes)
        if self._server_path:
            weights = np.tile(self._server_weights, (num_nodes, 1))
            graph = None
        else:
            graph = self._schedule.graph_at(round_idx)
            weights = np.array(self._mixing(graph, self._sizes), dtype=float)
            _check_inv_4(weights, graph, num_nodes)
        weights.setflags(write=False)
        return graph, weights

    def run_round(self, round_idx: int) -> RoundResult:
        if round_idx != self._next_round:
            raise RuntimeError(
                f"round phải chạy theo thứ tự 0, 1, 2, …: đang chờ {self._next_round}, nhận {round_idx}"
            )
        graph, weights = self._plan(round_idx)

        losses: dict[int, float] = {}
        for node in self._nodes:
            losses[node.node_id] = float(
                self._trainer(node.model, node.data, loss_scale=node.loss_scale, generator=node.generator)
            )
        snapshots: dict[int, Snapshot] = {}
        for node in self._nodes:
            snapshots[node.node_id] = make_snapshot(node.node_id, round_idx, node.model.state_dict())
        if graph is None:
            self._exchange_via_server(round_idx, snapshots)
        else:
            for node in self._nodes:
                for dst in sorted(graph.neighbors(node.node_id)):
                    self._transport.send(dst, snapshots[node.node_id])
        combined = {
            node.node_id: combine(
                snapshots[node.node_id],
                self._transport.inbox(node.node_id, round_idx),
                self._from_server if graph is None else weights[node.node_id],
            )
            for node in self._nodes
        }
        for node in self._nodes:
            node.model.load_state_dict(combined[node.node_id])
        self._transport.end_round(round_idx)
        self._next_round += 1

        ids = range(len(self._nodes))
        return RoundResult(
            round_idx=round_idx,
            weights=weights,
            train_losses=tuple(losses[i] for i in ids),
            snapshots=tuple(snapshots[i] for i in ids),
        )

    def _exchange_via_server(self, round_idx: int, snapshots: dict[int, Snapshot]) -> None:
        """N snapshot lên server ảo, tổng hợp Σ_k (n_k/n) w_k theo id tăng dần, một snapshot về mọi node."""
        server = len(self._nodes)
        for node in self._nodes:
            self._transport.send(server, snapshots[node.node_id])
        inbox = self._transport.inbox(server, round_idx)
        missing = sorted(set(range(server)) - set(inbox))
        if missing:
            raise RuntimeError(f"server ảo thiếu snapshot của node {missing}")
        aggregate = weighted_sum([(float(self._server_weights[k]), inbox[k]) for k in range(server)])
        broadcast = make_snapshot(server, round_idx, aggregate)
        for node in self._nodes:
            self._transport.send(node.node_id, broadcast)

    def states(self) -> tuple[StateDict, ...]:
        """Bản sao sở hữu riêng của model mọi node, xếp theo id; sửa chúng không ảnh hưởng engine."""
        return tuple(
            {name: tensor.detach().clone() for name, tensor in node.model.state_dict().items()}
            for node in self._by_id
        )

"""Engine ATC đồng bộ, một tiến trình (module sim; KH §3.4, §4.3; ADR-002, ADR-006; INV-1…INV-4).

Một round t gồm bốn pha, pha sau chỉ bắt đầu khi pha trước xong ở mọi node:
1. Adapt: LocalTrainer chạy τ bước trên model của node, xuất phát từ w_i^t (nên anchor của prox là w_i^t).
2. Gửi: mỗi node tạo đúng một snapshot w_i^(t+½) và gửi chính nó tới mọi láng giềng của G_t qua Transport.
3. Combine: mỗi node tổng hợp từ snapshot của chính nó và inbox của nó vào bộ đệm mới (node.combine).
4. Ghi: chỉ khi mọi node đã tổng hợp xong mới nạp kết quả vào model (G2, G22).
Engine không tính đại lượng toàn cục nào để đưa vào huấn luyện (INV-1) và không tự đếm byte (INV-2).
Engine duyệt node theo đúng thứ tự của dãy `nodes` được giao; kết quả không được phụ thuộc thứ tự đó
(T-INV-4).

torch.nn.Module.state_dict ("contains references to the module's parameters and buffers") và
load_state_dict ("Copy parameters and buffers from state_dict into this module"):
https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.state_dict
https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict
numpy.ndarray.setflags (W của round chỉ đọc): https://numpy.org/doc/2.5/reference/generated/numpy.ndarray.setflags.html
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from dfl.contracts import LocalTrainer, MixingPolicy, Snapshot, StateDict, TopologySchedule, Transport
from dfl.sim.node import Node, combine, make_snapshot


@dataclass(frozen=True, slots=True)
class RoundResult:
    """Những gì Observer cần từ một round; mọi dãy xếp theo id node.

    `weights` là W^(t) thực dùng (chỉ đọc); `snapshots` là đúng các snapshot w^(t+½) đã gửi, dùng để đo
    drift trước tổng hợp.
    """

    round_idx: int
    weights: np.ndarray
    train_losses: tuple[float, ...]
    snapshots: tuple[Snapshot, ...]


class Engine:
    def __init__(
        self,
        *,
        nodes: Sequence[Node],
        trainer: LocalTrainer,
        schedule: TopologySchedule,
        mixing: MixingPolicy,
        transport: Transport,
    ) -> None:
        num_nodes = len(nodes)
        if sorted(node.node_id for node in nodes) != list(range(num_nodes)):
            raise ValueError("node_id của các node phải là một hoán vị của 0..N-1")
        if schedule.num_nodes != num_nodes:
            raise ValueError(
                f"lịch topology có {schedule.num_nodes} node, nhưng engine nhận {num_nodes} node"
            )
        self._nodes = tuple(nodes)
        self._by_id = tuple(sorted(nodes, key=lambda node: node.node_id))
        self._sizes = np.array([len(node.data.y) for node in self._by_id])
        self._sizes.setflags(write=False)
        self._trainer = trainer
        self._schedule = schedule
        self._mixing = mixing
        self._transport = transport
        self._next_round = 0

    def run_round(self, round_idx: int) -> RoundResult:
        if round_idx != self._next_round:
            raise RuntimeError(
                f"round phải chạy theo thứ tự 0, 1, 2, …: đang chờ {self._next_round}, nhận {round_idx}"
            )
        graph = self._schedule.graph_at(round_idx)
        weights = np.array(self._mixing(graph, self._sizes), dtype=float)
        weights.setflags(write=False)

        losses: dict[int, float] = {}
        for node in self._nodes:
            losses[node.node_id] = float(
                self._trainer(node.model, node.data, loss_scale=node.loss_scale, generator=node.generator)
            )
        snapshots: dict[int, Snapshot] = {}
        for node in self._nodes:
            snapshot = make_snapshot(node.node_id, round_idx, node.model.state_dict())
            snapshots[node.node_id] = snapshot
            for dst in sorted(graph.neighbors(node.node_id)):
                self._transport.send(dst, snapshot)
        combined = {
            node.node_id: combine(
                snapshots[node.node_id], self._transport.inbox(node.node_id, round_idx), weights[node.node_id]
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

    def states(self) -> tuple[StateDict, ...]:
        """Bản sao sở hữu riêng của model mọi node, xếp theo id; sửa chúng không ảnh hưởng engine."""
        return tuple(
            {name: tensor.detach().clone() for name, tensor in node.model.state_dict().items()}
            for node in self._by_id
        )

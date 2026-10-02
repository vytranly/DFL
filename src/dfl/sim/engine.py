"""Engine ATC đồng bộ, một tiến trình (module sim; KH §3.4, §4.3; ADR-002, ADR-005, ADR-006, ADR-010).

Một round t gồm bốn pha, pha sau chỉ bắt đầu khi pha trước xong ở mọi node:
1. Adapt: LocalTrainer chạy τ bước trên model của node, xuất phát từ w_i^t (nên anchor của prox là w_i^t).
2. Gửi: mỗi node tạo đúng một snapshot w_i^(t+½) và gửi chính nó tới mọi láng giềng của G_t qua Transport.
3. Combine: mỗi node tổng hợp từ snapshot của chính nó và inbox của nó vào bộ đệm mới (node.combine).
4. Ghi: chỉ khi mọi node đã tổng hợp xong mới nạp kết quả vào model (G2, G22).

Bảo đảm và giới hạn:
- Ngoài đường server (B3/E1c: server ảo id N tổng hợp Σ_k (n_k/n) w_k, 2N message mỗi round), engine không
  tính đại lượng toàn cục nào cho huấn luyện (INV-1) và không tự đếm byte (INV-2).
- Kết quả không phụ thuộc thứ tự của dãy `nodes` (T-INV-4), với điều kiện trainer không có trạng thái giữa các
  lần gọi (contracts.LocalTrainer); engine không kiểm được điều này.
- Lúc dựng, engine từ chối:
  - Generator dùng chung, Generator trùng 32 bit thấp của seed (MT19937 chỉ dùng chừng ấy), hay
    `torch.default_generator`;
  - bộ nhớ tham số dùng chung giữa các node; dữ liệu mang nhãn node khác; c_i không dương;
  - BatchNorm, Dropout hay lớp giữ thống kê chạy (G3, ADR-004). `batch_norm` gọi dạng hàm thì không phát
    hiện được;
  - cấu trúc state khác nhau giữa các node; w⁰ không hữu hạn.
- Mỗi round, trước Adapt:
  - G_t và W^(t) qua INV-4;
  - với gossip, round phải nằm trong phạm vi đã `check_schedule` (hợp các round liên thông), và `graph_at`
    phải thuần: hai lần gọi cho cùng đồ thị (G18).
- Trong round:
  - RNG toàn cục của torch không đổi qua Adapt; numpy và `random` toàn cục đã bị lint chặn (NPY002, S311);
  - inbox chỉ chứa snapshot của đúng round và đúng người gửi;
  - mất mát và tham số hữu hạn, nếu không thì ném RunDiverged trước khi có bản ghi nào.
- Sau RunDiverged hay lỗi giữa round, engine không chạy tiếp và `states()` báo lỗi; `counters()` vẫn đọc
  được, để ghi SummaryDiverged. Lỗi lắp ráp đường server (Transport không có id N) chỉ lộ ra ở lần gửi đầu,
  sau Adapt của round 0; engine khi đó dừng hẳn.
- **Số học:** với model float32, Combine nhân bằng float32(W_ij), nên tổng hàng thực dùng lệch 1 một lượng cố
  định theo đồ thị (ring +3.0e-8, K_10 +1.5e-8; trên ER có hàng lệch âm). Khi các node còn khác nhau, sai lệch
  cộng dồn (ring 10 node, 4,000 round: khoảng 1.2e-4 tương đối của trung bình mạng), nên phép kiểm "trung bình
  bảo toàn" phải dùng float64 (T-INV-1). Với float64 trên FC + MH, W_ii = 1 − Σ_j W_ij có thể lệch 1/N ở bit
  cuối, nên các node trùng tới khoảng 1e-14 (float32 thì trùng từng bit).
- **Chi phí:** `nbytes` tính mọi tensor của state (contracts.Snapshot), kể cả tensor nguyên và tham số buộc
  chung. Snapshot server gửi xuống không có tensor nguyên. SmallCNN không có hai loại tensor này.

torch: state_dict ("contains references to the module's parameters and buffers"), load_state_dict ("Copy
parameters and buffers from state_dict into this module"), named_buffers, untyped_storage, Generator
(initial_seed), get_rng_state, equal, isfinite, BatchNorm1d, LazyBatchNorm1d (không phải lớp con của
BatchNorm1d), SyncBatchNorm, Dropout:
https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.state_dict
https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict
https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.named_buffers
https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.untyped_storage.html
https://docs.pytorch.org/docs/2.14/generated/torch.Generator.html
https://docs.pytorch.org/docs/2.14/generated/torch.get_rng_state.html
https://docs.pytorch.org/docs/2.14/generated/torch.equal.html
https://docs.pytorch.org/docs/2.14/generated/torch.isfinite.html
https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm1d.html
https://docs.pytorch.org/docs/2.14/generated/torch.nn.LazyBatchNorm1d.html
https://docs.pytorch.org/docs/2.14/generated/torch.nn.SyncBatchNorm.html
https://docs.pytorch.org/docs/2.14/generated/torch.nn.Dropout.html
numpy: frombuffer (mảng trên một bytes bất biến, không ai bật lại quyền ghi được), tile, isfinite, allclose:
https://numpy.org/doc/2.5/reference/generated/numpy.frombuffer.html
https://numpy.org/doc/2.5/reference/generated/numpy.tile.html
https://numpy.org/doc/2.5/reference/generated/numpy.isfinite.html
https://numpy.org/doc/2.5/reference/generated/numpy.allclose.html
networkx: Graph.edges, Graph.neighbors, Graph.copy, freeze, number_of_selfloops, is_connected, UnionFind:
https://networkx.org/documentation/networkx-3.7/reference/classes/generated/networkx.Graph.edges.html
https://networkx.org/documentation/networkx-3.7/reference/classes/generated/networkx.Graph.neighbors.html
https://networkx.org/documentation/networkx-3.7/reference/classes/generated/networkx.Graph.copy.html
https://networkx.org/documentation/networkx-3.7/reference/generated/networkx.classes.function.freeze.html
https://networkx.org/documentation/networkx-3.7/reference/generated/networkx.classes.function.number_of_selfloops.html
https://networkx.org/documentation/networkx-3.7/reference/algorithms/generated/networkx.algorithms.components.is_connected.html
https://networkx.org/documentation/networkx-3.7/reference/generated/networkx.utils.union_find.UnionFind.union.html
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch
from networkx.utils import UnionFind

from dfl.contracts import (
    CommCounters,
    LocalTrainer,
    MixingPolicy,
    Snapshot,
    StateDict,
    TopologySchedule,
    Transport,
)
from dfl.sim.node import Node, combine, make_snapshot, weighted_sum

# Sai số cho phép của tổng hàng: MH, uniform và size-weighted cho tổng lệch 1 cỡ vài ulp.
ROW_SUM_ATOL = 1e-12
# MT19937, Generator CPU của torch, chỉ dùng 32 bit thấp của seed.
SEED_MASK = 0xFFFF_FFFF
BATCH_NORMS = (
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.LazyBatchNorm1d,
    torch.nn.LazyBatchNorm2d,
    torch.nn.LazyBatchNorm3d,
    torch.nn.SyncBatchNorm,
)
DROPOUTS = (
    torch.nn.Dropout,
    torch.nn.Dropout1d,
    torch.nn.Dropout2d,
    torch.nn.Dropout3d,
    torch.nn.AlphaDropout,
    torch.nn.FeatureAlphaDropout,
)


@dataclass(frozen=True, slots=True, eq=False)
class RoundResult:
    """Những gì Observer cần từ round `round_idx` (đếm từ 0; RoundRecord.round = round_idx + 1).

    Mọi dãy xếp theo id node. `weights` là W^(t) đã kiểm INV-4, bất biến (ở đường server là ma trận hiệu dụng
    với mọi hàng bằng n_k/n). `snapshots` là snapshot w^(t+½) của mọi node, cũng là đúng các đối tượng đã gửi
    (node không có láng giềng trong round thì không gửi), dùng để đo drift trước tổng hợp.
    So sánh bằng danh tính.
    """

    round_idx: int
    weights: np.ndarray
    train_losses: tuple[float, ...]
    snapshots: tuple[Snapshot, ...]


class RunDiverged(Exception):
    """Mất mát hay tham số không còn hữu hạn; lượt dừng tại đó (SPEC §5.3).

    Phân kỳ là kết quả tất định của lượt, không phải lỗi cấu hình (SummaryDiverged). `round_idx` là round
    (đếm từ 0) đang chạy khi phát hiện, ở một trong ba chỗ: mất mát của Adapt, w^(t+½), hay w^(t+1).
    `diverged_at_round` = round_idx + 1 theo cách đánh số của RoundRecord (số round đã tổng hợp xong):
    rounds.jsonl chỉ có các dòng có `round` nhỏ hơn giá trị này.
    """

    def __init__(self, round_idx: int, reason: str) -> None:
        # args khớp chữ ký, để pickle dựng lại được ngoại lệ khi lượt chạy trong tiến trình con.
        super().__init__(round_idx, reason)
        self.round_idx = round_idx
        self.reason = reason

    def __str__(self) -> str:
        return f"round {self.round_idx}: {self.reason}"

    @property
    def diverged_at_round(self) -> int:
        return self.round_idx + 1


def _immutable(array: np.ndarray) -> np.ndarray:
    """Bản sao mà không ai bật lại quyền ghi được, kể cả qua `.base`: bộ nhớ là một đối tượng bytes."""
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _first_non_finite(state: StateDict) -> str | None:
    for name, tensor in state.items():
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            return name
    return None


def _check_finite(states: dict[int, StateDict], round_idx: int, phase: str) -> None:
    for node_id in sorted(states):
        name = _first_non_finite(states[node_id])
        if name is not None:
            raise RunDiverged(round_idx, f"tham số {name} của node {node_id} không hữu hạn {phase}")


def _check_nodes(nodes: Sequence[Node]) -> None:
    num_nodes = len(nodes)
    if num_nodes == 0:
        raise ValueError("engine cần ít nhất một node")
    if sorted(node.node_id for node in nodes) != list(range(num_nodes)):
        raise ValueError("node_id của các node phải là một hoán vị của 0..N-1")
    for node in nodes:
        i = node.node_id
        if node.data.node_id != i:
            raise ValueError(f"node {i} giữ dữ liệu của node {node.data.node_id}")
        if len(node.data.y) == 0:
            raise ValueError(f"node {i} không có mẫu nào")
        if len(node.data.x) != len(node.data.y):
            raise ValueError(
                f"node {i}: x và y của dữ liệu khác số mẫu ({len(node.data.x)} và {len(node.data.y)})"
            )
        if not (math.isfinite(node.loss_scale) and node.loss_scale > 0):
            raise ValueError(f"loss_scale (c_i) của node {i} phải hữu hạn và dương, nhận {node.loss_scale}")
        if node.generator is torch.default_generator:
            raise ValueError(
                f"node {i} dùng torch.default_generator (RNG toàn cục) làm luồng batch; "
                "cần Generator riêng (INV-5)"
            )
        for module_name, module in node.model.named_modules():
            where = module_name or "gốc"
            buffers = {name for name, _ in module.named_buffers(recurse=False)}
            if isinstance(module, BATCH_NORMS) or {"running_mean", "running_var"} & buffers:
                raise ValueError(
                    f"model của node {i} có BatchNorm hay lớp giữ thống kê chạy ({where}): "
                    "thống kê đó sẽ bị trộn giữa các node (G3, ADR-004)"
                )
            if isinstance(module, DROPOUTS):
                raise ValueError(
                    f"model của node {i} có Dropout ({where}): Dropout rút số từ RNG toàn cục, và ADR-004 "
                    "không dùng dropout"
                )
    if len({id(node.generator) for node in nodes}) != num_nodes:
        raise ValueError("mỗi node cần torch.Generator riêng cho luồng batch (INV-5)")
    if len({node.generator.initial_seed() & SEED_MASK for node in nodes}) != num_nodes:
        raise ValueError(
            "các Generator của node trùng seed (xét 32 bit thấp, phần MT19937 dùng), nên các node rút "
            "cùng một dãy minibatch"
        )
    owner: dict[int, int] = {}
    for node in nodes:
        for tensor in node.model.state_dict().values():
            if tensor.numel() == 0:
                continue
            pointer = tensor.untyped_storage().data_ptr()
            if owner.setdefault(pointer, node.node_id) != node.node_id:
                raise ValueError(f"node {owner[pointer]} và node {node.node_id} dùng chung bộ nhớ tham số")
    schema = [{name: (t.shape, t.dtype) for name, t in node.model.state_dict().items()} for node in nodes]
    if any(entry != schema[0] for entry in schema[1:]):
        raise ValueError("mọi node phải có cùng cấu trúc state (tên, shape, dtype của tensor)")
    for node in nodes:
        name = _first_non_finite(node.model.state_dict())
        if name is not None:
            raise ValueError(f"w⁰ của node {node.node_id} không hữu hạn ở {name}")


def _check_round(graph: nx.Graph, weights: np.ndarray, num_nodes: int, *, static: bool) -> None:
    """INV-4 cho G_t và W^(t), trước khi round dùng chúng."""
    if set(graph.nodes) != set(range(num_nodes)):
        raise ValueError(f"đồ thị của round phải có đúng các node 0..{num_nodes - 1}")
    if nx.number_of_selfloops(graph) > 0:
        raise ValueError("đồ thị của round có self-loop: node không tự gửi message cho mình (G7)")
    if weights.shape != (num_nodes, num_nodes):
        raise ValueError(f"W phải có shape ({num_nodes}, {num_nodes}), nhận {weights.shape}")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("W phải hữu hạn và không âm (INV-4)")
    if not np.allclose(weights.sum(axis=1), 1.0, rtol=0.0, atol=ROW_SUM_ATOL):
        raise ValueError("tổng mỗi hàng của W phải bằng 1 (INV-4)")
    # Ma trận kề dựng thẳng từ danh sách cạnh: bỏ qua mọi thuộc tính cạnh (kể cả "weight"), và nhanh hơn
    # to_numpy_array, vì gossip gọi phép kiểm này mỗi round.
    edges = np.array(list(graph.edges()), dtype=int).reshape(-1, 2)
    adjacency = np.zeros((num_nodes, num_nodes), dtype=bool)
    adjacency[edges[:, 0], edges[:, 1]] = True
    adjacency[edges[:, 1], edges[:, 0]] = True
    if ((weights != 0) & ~adjacency & ~np.eye(num_nodes, dtype=bool)).any():
        raise ValueError("W khác 0 ngoài cạnh của đồ thị round này (INV-4)")
    if (adjacency & (weights == 0)).any():
        raise ValueError(
            "W bằng 0 trên một cạnh: message trên cạnh đó bị tính phí mà không được dùng (INV-4)"
        )
    if static and graph.number_of_edges() > 0 and not nx.is_connected(graph):
        raise ValueError("đồ thị tĩnh phải liên thông; chỉ W = I (B2, không cạnh) được phép rời rạc (INV-4)")


def _edge_set(graph: nx.Graph) -> set[frozenset[int]]:
    return {frozenset(edge) for edge in graph.edges()}


def _check_inbox(inbox: Mapping[int, Snapshot], dst: int, round_idx: int) -> None:
    for sender, message in inbox.items():
        if message.sender != sender or message.round_idx != round_idx:
            raise RuntimeError(
                f"inbox của node {dst} ở round {round_idx} có snapshot sai: khóa {sender}, người gửi "
                f"{message.sender}, round {message.round_idx}"
            )


def check_union_connected(schedule: TopologySchedule, num_nodes: int, total_rounds: int) -> None:
    """INV-4 cho lịch thay đổi theo round: hợp các G_t của round 0 … total_rounds − 1 phải liên thông.

    Dừng ngay khi hợp đã liên thông, nên với gossip đúng chỉ tốn vài lần gọi `graph_at`. Engine và process
    runtime (ADR-010) đều gọi qua `check_schedule_union`.
    """
    components = UnionFind(range(num_nodes))
    groups = num_nodes
    for round_idx in range(total_rounds):
        for u, v in schedule.graph_at(round_idx).edges():
            if components[u] != components[v]:
                components.union(u, v)
                groups -= 1
        if groups == 1:
            return
    raise ValueError(
        f"hợp các round 0…{total_rounds - 1} của lịch gossip không liên thông: {groups} nhóm node "
        "không bao giờ trao đổi với nhau (INV-4)"
    )


def check_schedule_union(schedule: TopologySchedule | None, num_nodes: int, total_rounds: int) -> None:
    """Kiểm INV-4 theo hợp các round khi lịch thay đổi theo round; lịch tĩnh và đường server không cần."""
    if total_rounds < 1:
        raise ValueError(f"total_rounds phải ≥ 1, nhận {total_rounds}")
    if schedule is not None and not schedule.is_static and num_nodes > 1:
        check_union_connected(schedule, num_nodes, total_rounds)


def preflight(nodes: Sequence[Node], schedule: TopologySchedule | None, total_rounds: int) -> None:
    """Các phép kiểm trước round đầu mà mọi runtime phải qua: node (G3, INV-5, G22; như lúc dựng Engine) và
    INV-4 theo hợp các round.

    Process runtime (ADR-010) chạy hàm này trên các node mà tiến trình chạy lượt tự lắp, trước khi khởi chạy
    tiến trình node, để hai runtime từ chối đúng cùng các lượt (T-INV-13).
    """
    _check_nodes(nodes)
    check_schedule_union(schedule, len(nodes), total_rounds)


class Engine:
    """Chạy các round ATC trên N node; mọi giao tiếp đi qua Transport được giao."""

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
        _check_nodes(nodes)
        num_nodes = len(nodes)
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
        self._sizes = _immutable(np.array([len(node.data.y) for node in self._by_id]))
        self._trainer = trainer
        self._transport = transport
        self._schedule = schedule
        self._mixing = mixing
        sizes = self._sizes.astype(float)
        self._server_weights = _immutable(sizes / sizes.sum())
        self._server_plan = _immutable(np.tile(self._server_weights, (num_nodes, 1)))
        # Ở đường server, node chỉ lấy snapshot của server ảo (id N): hàng W dài N + 1 với W[i, N] = 1.
        from_server = np.zeros(num_nodes + 1)
        from_server[num_nodes] = 1.0
        self._from_server = _immutable(from_server)
        self._static_plan: tuple[nx.Graph, np.ndarray] | None = None
        self._dynamic = schedule is not None and not schedule.is_static
        self._checked_rounds = 0
        self._next_round = 0
        self._halted: str | None = None

    def weights_at(self, round_idx: int) -> np.ndarray:
        """W^(t) của round t, bất biến, đã kiểm INV-4; ở đường server là ma trận hiệu dụng n_k/n.

        Cùng hàm tính mà run_round dùng, nên trùng với W thực dùng khi lịch và mixing thuần (run_round kiểm
        tính thuần của lịch).
        """
        return self._plan(round_idx)[1]

    def check_schedule(self, total_rounds: int) -> None:
        """Kiểm INV-4 cho cả lượt trước khi chạy; với gossip, chỉ các round 0 … total_rounds − 1 được chạy.

        `run_rounds` gọi hàm này. Đồ thị tĩnh được kiểm ở `_check_round`; đường server không có đồ thị.
        """
        check_schedule_union(self._schedule, len(self._nodes), total_rounds)
        self._checked_rounds = max(self._checked_rounds, total_rounds)

    def _plan(self, round_idx: int) -> tuple[nx.Graph | None, np.ndarray]:
        schedule, mixing = self._schedule, self._mixing
        if schedule is None or mixing is None:
            return None, self._server_plan
        if self._static_plan is not None:
            return self._static_plan
        graph = schedule.graph_at(round_idx)
        if schedule.is_static:
            graph = nx.freeze(
                graph.copy()
            )  # bản riêng: người khác sửa đồ thị gốc cũng không lệch khỏi W đã kiểm
        weights = np.array(mixing(graph, self._sizes), dtype=float)
        _check_round(graph, weights, len(self._nodes), static=schedule.is_static)
        plan = (graph, _immutable(weights))
        if schedule.is_static:
            self._static_plan = plan
        return plan

    def run_round(self, round_idx: int) -> RoundResult:
        """Chạy round `round_idx`; các round phải chạy lần lượt 0, 1, 2, …"""
        if self._halted is not None:
            raise RuntimeError(f"engine không chạy tiếp: {self._halted}")
        if round_idx != self._next_round:
            raise RuntimeError(
                f"round phải chạy theo thứ tự 0, 1, 2, …: đang chờ {self._next_round}, nhận {round_idx}"
            )
        schedule = self._schedule
        if self._dynamic and round_idx >= self._checked_rounds:
            raise RuntimeError(
                f"round {round_idx} nằm ngoài phạm vi đã kiểm INV-4 ({self._checked_rounds} round); gọi "
                "check_schedule(total_rounds) trước (run_rounds làm việc này)"
            )
        graph, weights = self._plan(round_idx)  # lỗi ở đây chưa chạm model nào
        if self._dynamic and schedule is not None and graph is not None:
            if _edge_set(schedule.graph_at(round_idx)) != _edge_set(graph):
                raise RuntimeError(
                    f"graph_at({round_idx}) không thuần: hai lần gọi cho hai đồ thị khác nhau (G18, T-INV-12)"
                )
        try:
            return self._run(round_idx, graph, weights)
        except RunDiverged:
            self._halted = f"lượt đã phân kỳ ở round {round_idx}"
            raise
        except BaseException:
            self._halted = f"round {round_idx} dừng giữa chừng vì lỗi; model có thể đã đổi một phần"
            raise

    def _run(self, round_idx: int, graph: nx.Graph | None, weights: np.ndarray) -> RoundResult:
        rng_before = torch.get_rng_state()
        losses = {
            node.node_id: float(
                self._trainer(node.model, node.data, loss_scale=node.loss_scale, generator=node.generator)
            )
            for node in self._nodes
        }
        if not torch.equal(torch.get_rng_state(), rng_before):
            raise ValueError(
                "Adapt đã dùng RNG toàn cục của torch (ví dụ rút số ngẫu nhiên không qua Generator riêng của "
                "node); quỹ đạo khi đó phụ thuộc thứ tự duyệt (INV-5)"
            )
        for node_id in sorted(losses):  # theo id, để lý do không phụ thuộc thứ tự duyệt
            if not math.isfinite(losses[node_id]):
                raise RunDiverged(
                    round_idx, f"mất mát huấn luyện của node {node_id} không hữu hạn ({losses[node_id]})"
                )
        snapshots: dict[int, Snapshot] = {}
        for node in self._nodes:
            snapshots[node.node_id] = make_snapshot(node.node_id, round_idx, node.model.state_dict())
        _check_finite({i: s.state for i, s in snapshots.items()}, round_idx, "sau huấn luyện cục bộ")
        if graph is None:
            self._exchange_via_server(round_idx, snapshots)
        else:
            for node in self._nodes:
                for dst in sorted(graph.neighbors(node.node_id)):
                    self._transport.send(dst, snapshots[node.node_id])
        combined: dict[int, StateDict] = {}
        for node in self._nodes:
            inbox = self._transport.inbox(node.node_id, round_idx)
            _check_inbox(inbox, node.node_id, round_idx)
            row = self._from_server if graph is None else weights[node.node_id]
            combined[node.node_id] = combine(snapshots[node.node_id], inbox, row)
        _check_finite(combined, round_idx, "sau tổng hợp")
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
        _check_inbox(inbox, server, round_idx)
        missing = sorted(set(range(server)) - set(inbox))
        if missing:
            raise RuntimeError(f"server ảo thiếu snapshot của node {missing}")
        aggregate = weighted_sum([(float(self._server_weights[k]), inbox[k]) for k in range(server)])
        broadcast = make_snapshot(server, round_idx, aggregate)
        for node in self._nodes:
            self._transport.send(node.node_id, broadcast)

    def counters(self) -> CommCounters:
        """Số đếm tích lũy của Transport (chỉ đọc; INV-2: engine không tự đếm)."""
        return self._transport.counters()

    def states(self) -> tuple[StateDict, ...]:
        """Bản sao sở hữu riêng của model mọi node, xếp theo id; sửa chúng không ảnh hưởng engine."""
        if self._halted is not None:
            raise RuntimeError(f"engine đã dừng ({self._halted}); state có thể là hỗn hợp w^t và w^(t+½)")
        return tuple(
            {name: tensor.detach().clone() for name, tensor in node.model.state_dict().items()}
            for node in self._by_id
        )

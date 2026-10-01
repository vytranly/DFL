"""Transport in-memory và CommLedger (module network; ADR-002; INV-1, INV-2; SPEC REQ-05).

Transport coi payload là đối tượng mờ: chỉ đọc `sender`, `round_idx`, `nbytes` của Snapshot, không đọc
`state`, nên module không import torch (contract pure-no-torch). Không copy payload: cùng một Snapshot
được giao cho mọi láng giềng, và byte được tính theo `nbytes` đã đo lúc tạo snapshot.

types.MappingProxyType (view chỉ đọc của dict):
https://docs.python.org/3.12/library/types.html#types.MappingProxyType
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from dfl.contracts import CommCounters, Snapshot


class CommLedger:
    """Số message, số byte và tải (byte gửi + byte nhận) theo từng node, cộng dồn từ đầu lượt.

    Chỉ InMemoryTransport.send gọi `record` (INV-2: chỉ Transport đếm byte).
    """

    def __init__(self, num_ids: int) -> None:
        self._messages = 0
        self._bytes = 0
        self._node_bytes = [0] * num_ids

    def record(self, sender: int, dst: int, nbytes: int) -> None:
        self._messages += 1
        self._bytes += nbytes
        self._node_bytes[sender] += nbytes
        self._node_bytes[dst] += nbytes

    def counters(self) -> CommCounters:
        return CommCounters(
            messages_cum=self._messages, bytes_cum=self._bytes, max_node_bytes_cum=max(self._node_bytes)
        )

    def node_bytes(self) -> tuple[int, ...]:
        return tuple(self._node_bytes)


class InMemoryTransport:
    """Transport mặc định: một tiến trình, đồng bộ theo round (ADR-002).

    Với `with_server=True`, id N là server ảo của đường server B3/E1c và được đếm tải như một node.
    """

    def __init__(self, num_nodes: int, *, with_server: bool = False) -> None:
        if num_nodes < 1:
            raise ValueError(f"num_nodes phải ≥ 1, nhận {num_nodes}")
        self._with_server = with_server
        self._num_ids = num_nodes + 1 if with_server else num_nodes
        self._ledger = CommLedger(self._num_ids)
        # round → dst → sender → snapshot
        self._inboxes: dict[int, dict[int, dict[int, Snapshot]]] = {}
        self._ended: set[int] = set()

    def _check_id(self, node_id: int, role: str) -> None:
        if not 0 <= node_id < self._num_ids:
            allowed = "0..N (N là server ảo)" if self._with_server else "0..N-1"
            raise ValueError(f"{role} phải nằm trong {allowed}, nhận {node_id}")

    def _check_open(self, round_idx: int) -> None:
        if round_idx in self._ended:
            raise RuntimeError(f"round {round_idx} đã kết thúc (end_round); không gửi hay đọc inbox được nữa")

    def send(self, dst: int, message: Snapshot) -> None:
        self._check_id(dst, "dst")
        self._check_id(message.sender, "sender")
        if dst == message.sender:
            raise ValueError(f"node {dst} tự gửi cho mình; không có message như vậy (G7)")
        if message.nbytes < 0:
            raise ValueError(f"nbytes phải ≥ 0, nhận {message.nbytes}")
        self._check_open(message.round_idx)
        inbox = self._inboxes.setdefault(message.round_idx, {}).setdefault(dst, {})
        if message.sender in inbox:
            raise ValueError(f"node {message.sender} đã gửi cho {dst} trong round {message.round_idx}")
        inbox[message.sender] = message
        self._ledger.record(message.sender, dst, message.nbytes)

    def inbox(self, dst: int, round_idx: int) -> Mapping[int, Snapshot]:
        self._check_id(dst, "dst")
        self._check_open(round_idx)
        return MappingProxyType(dict(self._inboxes.get(round_idx, {}).get(dst, {})))

    def end_round(self, round_idx: int) -> None:
        self._inboxes.pop(round_idx, None)
        self._ended.add(round_idx)

    def counters(self) -> CommCounters:
        return self._ledger.counters()

    def node_bytes(self) -> tuple[int, ...]:
        """Tải cộng dồn theo id (0..N-1, thêm N nếu có server ảo)."""
        return self._ledger.node_bytes()

"""Transport đa tiến trình qua torch.distributed (gloo): mỗi tiến trình hệ điều hành là một node (module sim;
ADR-010; INV-1, INV-2, G7). Đặt ở `sim`, không ở `network`, vì phải import torch (contract pure-no-torch).

Một message là state của snapshot, mã hóa thành một tensor uint8 liền theo thứ tự khóa cố định của
`StateLayout`. Gloo chép đúng các byte đó, và người nhận giải mã bằng `Tensor.view(dtype)`, nên state đi qua
mạng trùng từng bit (điều kiện cho T-INV-13). Mọi node có cùng cấu trúc state (engine kiểm), nên mỗi tiến
trình tự dựng layout từ model của nó.

Mỗi round, tiến trình node gọi `expect` (đăng ký `irecv` cho từng láng giềng), rồi `send` (`isend` một
snapshot tới từng láng giềng), rồi `inbox` (đợi và giải mã), rồi `end_round`. Gửi và nhận đều không chặn,
nên không có deadlock dù các node gửi theo thứ tự nào. `tag` là chỉ số round, để message của hai round không
lẫn nhau.
- `isend`/`irecv` có trong tài liệu hàm của torch 2.14, nhưng bảng backend chỉ ghi `send`/`recv` cho gloo.
  Dự án kiểm bằng thực nghiệm (tests/integration/test_torchdist.py; ghi là UNVERIFIED theo tài liệu).
- Tài liệu ghi "Modifying tensor before the request completes causes undefined behavior", nên buffer đã gửi
  được giữ tới `end_round`, lúc mọi lần gửi đã xong.

CommLedger ở đây là của riêng node: message và byte nó gửi, cùng tải (gửi + nhận). Observer của process
runtime cộng số đếm của các node để ra số của cả lượt, cùng định nghĩa với InMemoryTransport (INV-2).

torch.distributed (isend, irecv, Work, init_process_group, bảng backend):
https://docs.pytorch.org/docs/2.14/distributed.html#torch.distributed.isend
https://docs.pytorch.org/docs/2.14/distributed.html#torch.distributed.irecv
https://docs.pytorch.org/docs/2.14/distributed.html
torch.Tensor.view(dtype): https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.view.html
torch.cat: https://docs.pytorch.org/docs/2.14/generated/torch.cat.html
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch
import torch.distributed as dist

from dfl.contracts import CommCounters, Snapshot, StateDict


@dataclass(frozen=True, slots=True)
class LayoutEntry:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    offset: int
    nbytes: int


@dataclass(frozen=True, slots=True)
class StateLayout:
    """Tên, shape, dtype và vị trí byte của từng tensor của state trong một message."""

    entries: tuple[LayoutEntry, ...]
    nbytes: int

    @classmethod
    def of(cls, state: StateDict) -> StateLayout:
        entries = []
        offset = 0
        for name, tensor in state.items():
            size = tensor.numel() * tensor.element_size()
            entries.append(LayoutEntry(name, tuple(tensor.shape), tensor.dtype, offset, size))
            offset += size
        return cls(tuple(entries), offset)


def encode_state(state: StateDict, layout: StateLayout) -> torch.Tensor:
    """Byte của mọi tensor theo thứ tự của layout, trong một tensor uint8 mới, tách khỏi bộ nhớ của state."""
    if list(state) != [entry.name for entry in layout.entries]:
        raise ValueError("state không khớp layout: tên hay thứ tự tensor khác nhau")
    parts = []
    for entry in layout.entries:
        tensor = state[entry.name]
        if tuple(tensor.shape) != entry.shape or tensor.dtype != entry.dtype:
            raise ValueError(
                f"state không khớp layout ở {entry.name}: {tuple(tensor.shape)} {tensor.dtype}, "
                f"layout ghi {entry.shape} {entry.dtype}"
            )
        parts.append(tensor.detach().reshape(-1).view(torch.uint8))
    return torch.cat(parts) if parts else torch.zeros(0, dtype=torch.uint8)


def decode_state(buffer: torch.Tensor, layout: StateLayout) -> StateDict:
    """Ngược với encode_state; mỗi tensor có bộ nhớ riêng, nên sửa buffer sau đó không ảnh hưởng."""
    if buffer.dtype != torch.uint8 or buffer.dim() != 1 or buffer.numel() != layout.nbytes:
        raise ValueError(
            f"message phải là tensor uint8 một chiều {layout.nbytes} byte, "
            f"nhận {buffer.numel()} byte {buffer.dtype}"
        )
    state: StateDict = {}
    for entry in layout.entries:
        # clone: bộ nhớ riêng bắt đầu từ offset 0, nên view(dtype) không vướng điều kiện căn lề của offset.
        chunk = buffer[entry.offset : entry.offset + entry.nbytes].clone()
        state[entry.name] = chunk.view(entry.dtype).reshape(entry.shape)
    return state


class TorchDistTransport:
    """Transport của một tiến trình node; process group gloo phải được khởi tạo trước lần gửi hay nhận đầu."""

    def __init__(self, node_id: int, num_nodes: int, layout: StateLayout) -> None:
        if not 0 <= node_id < num_nodes:
            raise ValueError(f"node_id phải nằm trong 0..{num_nodes - 1}, nhận {node_id}")
        self._me = node_id
        self._num_nodes = num_nodes
        self._layout = layout
        self._receives: dict[int, dict[int, tuple[dist.Work, torch.Tensor]]] = {}
        self._sends: dict[int, list[tuple[dist.Work, torch.Tensor]]] = {}
        self._sent_to: dict[int, set[int]] = {}
        self._inboxes: dict[int, Mapping[int, Snapshot]] = {}
        self._ended: set[int] = set()
        self._sent_messages = 0
        self._sent_bytes = 0
        self._received_bytes = 0

    def _check_peer(self, peer: int, role: str) -> None:
        if peer == self._me:
            raise ValueError(f"node {peer} tự gửi cho mình; không có message như vậy (G7)")
        if not 0 <= peer < self._num_nodes:
            raise ValueError(f"{role} phải nằm trong 0..N-1, nhận {peer}")

    def _check_open(self, round_idx: int) -> None:
        if round_idx in self._ended:
            raise RuntimeError(f"round {round_idx} đã kết thúc (end_round); không gửi hay nhận được nữa")

    def expect(self, round_idx: int, senders: Iterable[int]) -> None:
        """Đăng ký nhận đúng một snapshot từ mỗi người gửi của round (irecv không chặn)."""
        senders = list(senders)
        for sender in senders:
            self._check_peer(sender, "người gửi")
        self._check_open(round_idx)
        posted = self._receives.setdefault(round_idx, {})
        if round_idx in self._inboxes or any(sender in posted for sender in senders):
            raise RuntimeError(f"round {round_idx}: người gửi đã được đăng ký, hoặc inbox đã được đọc")
        for sender in senders:
            buffer = torch.empty(self._layout.nbytes, dtype=torch.uint8)
            posted[sender] = (dist.irecv(buffer, src=sender, tag=round_idx), buffer)

    def send(self, dst: int, message: Snapshot) -> None:
        if message.sender != self._me:
            raise ValueError(
                f"tiến trình node {self._me} chỉ gửi snapshot của chính nó, nhận người gửi {message.sender}"
            )
        self._check_peer(dst, "dst")
        self._check_open(message.round_idx)
        sent = self._sent_to.setdefault(message.round_idx, set())
        if dst in sent:
            raise ValueError(f"node {self._me} đã gửi cho {dst} trong round {message.round_idx}")
        buffer = encode_state(message.state, self._layout)
        if buffer.numel() != message.nbytes:
            raise ValueError(f"nbytes của snapshot là {message.nbytes}, nhưng state có {buffer.numel()} byte")
        work = dist.isend(buffer, dst=dst, tag=message.round_idx)
        self._sends.setdefault(message.round_idx, []).append((work, buffer))
        sent.add(dst)
        self._sent_messages += 1
        self._sent_bytes += message.nbytes

    def inbox(self, dst: int, round_idx: int) -> Mapping[int, Snapshot]:
        """Đợi mọi snapshot đã `expect` của round, giải mã, và trả về theo id người gửi tăng dần."""
        if dst != self._me:
            raise ValueError(
                f"tiến trình node {self._me} chỉ đọc được inbox của chính nó, nhận {dst} (INV-1)"
            )
        self._check_open(round_idx)
        if round_idx not in self._inboxes:
            received = {}
            for sender, (work, buffer) in sorted(self._receives.pop(round_idx, {}).items()):
                work.wait()
                state = decode_state(buffer, self._layout)
                received[sender] = Snapshot(
                    sender=sender, round_idx=round_idx, state=state, nbytes=self._layout.nbytes
                )
                self._received_bytes += self._layout.nbytes
            self._inboxes[round_idx] = MappingProxyType(received)
        return self._inboxes[round_idx]

    def end_round(self, round_idx: int) -> None:
        """Đợi mọi lần gửi của round xong rồi mới thả buffer; inbox của round phải đã được đọc."""
        if self._receives.get(round_idx):
            raise RuntimeError(f"round {round_idx} còn snapshot đã expect nhưng chưa đọc inbox")
        for work, _ in self._sends.pop(round_idx, []):
            work.wait()
        self._receives.pop(round_idx, None)
        self._inboxes.pop(round_idx, None)
        self._sent_to.pop(round_idx, None)
        self._ended.add(round_idx)

    def counters(self) -> CommCounters:
        """Số đếm của riêng node: message và byte nó gửi; `max_node_bytes_cum` là tải của nó (gửi + nhận)."""
        return CommCounters(
            messages_cum=self._sent_messages,
            bytes_cum=self._sent_bytes,
            max_node_bytes_cum=self._sent_bytes + self._received_bytes,
        )

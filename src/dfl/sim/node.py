"""Phần của một node trong engine: trạng thái riêng, snapshot gửi đi và bước Combine (KH §3.4; INV-1, INV-3).

Tensor.detach trả về tensor "shares the same storage with the original one", còn torch.clone "is
differentiable", nên snapshot dùng detach().clone(): bản sao riêng, không dính autograd, để aggregate
đọc snapshot trong khi model của node có thể đã đổi (INV-3).
https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.detach.html
https://docs.pytorch.org/docs/2.14/generated/torch.clone.html
Byte tính từ tensor thật, numel × element_size:
https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.numel.html
https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.element_size.html
Chỉ trộn tensor có kiểu dấu phẩy động (float64, float32, float16, bfloat16):
https://docs.pytorch.org/docs/2.14/generated/torch.is_floating_point.html
numpy.flatnonzero: https://numpy.org/doc/2.5/reference/generated/numpy.flatnonzero.html
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch

from dfl.contracts import NodeDataset, Snapshot, StateDict


@dataclass(frozen=True, eq=False)
class Node:
    """Những gì chỉ node i được dùng: model w_i, dữ liệu D_i, luồng batch riêng, hệ số mất mát c_i (KH §3.4).

    Gom vào một chỗ để engine không phải giữ các dãy song song theo chỉ số, vốn dễ lệch nhau.
    """

    node_id: int
    model: torch.nn.Module
    data: NodeDataset
    generator: torch.Generator
    loss_scale: float = 1.0


def make_snapshot(sender: int, round_idx: int, state: StateDict) -> Snapshot:
    """Bản sao sở hữu riêng của `state` (thường là model.state_dict()), gửi được tới mọi láng giềng."""
    owned = {name: tensor.detach().clone() for name, tensor in state.items()}
    nbytes = sum(t.numel() * t.element_size() for t in owned.values())
    return Snapshot(sender=sender, round_idx=round_idx, state=owned, nbytes=nbytes)


def combine(own: Snapshot, inbox: Mapping[int, Snapshot], weights_row: np.ndarray) -> StateDict:
    """w_i^(t+1) = Σ_j W_ij · w_j^(t+½), chỉ từ snapshot của chính node và inbox của nó (INV-1).

    Cộng theo id người gửi tăng dần, kể cả số hạng của chính node ở đúng vị trí id của nó, nên kết quả trùng
    từng bit bất kể thứ tự duyệt node hay thứ tự trong inbox (INV-3, T-INV-4). Mọi phép tính ra tensor mới,
    không phép tại chỗ nào chạm vào snapshot. Tensor không phải dấu phẩy động (ví dụ bộ đếm int64) không được
    trộn mà giữ giá trị của chính node (G3).
    """
    contributions = {**inbox, own.sender: own}
    needed = sorted(int(j) for j in np.flatnonzero(weights_row))
    missing = [j for j in needed if j not in contributions]
    if missing:
        raise RuntimeError(f"node {own.sender} thiếu snapshot của node {missing} dù W khác 0 (INV-4)")
    combined: StateDict = {}
    for name, own_tensor in own.state.items():
        if not own_tensor.is_floating_point():
            combined[name] = own_tensor.clone()
            continue
        total = None
        for j in needed:
            term = float(weights_row[j]) * contributions[j].state[name]
            total = term if total is None else total + term
        combined[name] = total
    return combined

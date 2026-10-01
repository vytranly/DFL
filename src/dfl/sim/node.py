"""Phần của một node trong engine: tạo snapshot gửi đi (KH §3.4 bước 2; INV-3; SPEC REQ-05).

Tensor.detach trả về tensor "shares the same storage with the original one", còn torch.clone "is
differentiable", nên snapshot dùng detach().clone(): bản sao riêng, không dính autograd, để aggregate
đọc snapshot trong khi model của node có thể đã đổi (INV-3).
https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.detach.html
https://docs.pytorch.org/docs/2.14/generated/torch.clone.html
Byte tính từ tensor thật, numel × element_size:
https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.numel.html
https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.element_size.html
"""

from __future__ import annotations

from dfl.contracts import Snapshot, StateDict


def make_snapshot(sender: int, round_idx: int, state: StateDict) -> Snapshot:
    """Bản sao sở hữu riêng của `state` (thường là model.state_dict()), gửi được tới mọi láng giềng."""
    owned = {name: tensor.detach().clone() for name, tensor in state.items()}
    nbytes = sum(t.numel() * t.element_size() for t in owned.values())
    return Snapshot(sender=sender, round_idx=round_idx, state=owned, nbytes=nbytes)

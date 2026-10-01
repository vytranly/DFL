"""Khoảng cách RMS tới model trung bình: drift trước tổng hợp và consensus distance sau tổng hợp (KH §3.6).

Ξ = sqrt(1/N Σ_i ‖w_i − w̄‖²) trên mọi tensor dấu phẩy động (Kong et al. 2021). Cùng một công thức cho cả hai
đại lượng: drift Ξ_(t+½) tính trên các snapshot w^(t+½), consensus distance Ξ_(t+1) trên các model sau tổng
hợp. Tính bằng float64, vì trung bình của các tensor float32 gần nhau không biểu diễn được bằng float32, và
SAN-5 so hai đại lượng với sai số tương đối 1e-9 (SPEC §9.3).
torch.Tensor.to: https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.to.html
torch.is_floating_point: https://docs.pytorch.org/docs/2.14/generated/torch.is_floating_point.html
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from dfl.contracts import StateDict


def consensus_distance(states: Sequence[StateDict]) -> float:
    """Ξ của các model `states[i]`; tensor không phải dấu phẩy động (bộ đếm) không được tính."""
    if not states:
        raise ValueError("cần ít nhất một model để tính khoảng cách tới trung bình")
    names = [name for name, tensor in states[0].items() if tensor.is_floating_point()]
    rows = torch.stack(
        [
            torch.cat([state[name].detach().reshape(-1).to(torch.float64) for name in names])
            for state in states
        ]
    )
    deviations = rows - rows.mean(dim=0)
    return math.sqrt(float((deviations**2).sum(dim=1).mean()))

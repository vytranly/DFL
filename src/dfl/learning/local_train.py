"""Bước Adapt của vòng ATC: τ bước SGD trên dữ liệu cục bộ của một node (module learning; KH §3.4; ADR-004,
ADR-006; T-INV-5, T-INV-7; G6).

Mỗi bước rút B chỉ số có hoàn lại bằng Generator batch của node, tính
    mất mát = c_i · cross-entropy + μ/2 · Σ‖w − anchor‖²,
rồi cập nhật w ← w − η·∇. Đây là SGD với momentum 0 và weight decay 0, đúng phép cập nhật của vòng for trong
torch.optim.SGD (`param.add_(grad, alpha=-lr)`), nên với μ = 0 kết quả trùng từng bit với torch.optim.SGD
(T-INV-7). Gradient lấy bằng torch.autograd.grad, nên model không giữ `.grad` nào giữa các lần gọi.

Anchor là bản clone + detach của tham số lúc bắt đầu lời gọi, tức w_i^t, và cố định suốt τ bước (G6). Với
μ = 0 thì không tính số hạng prox và không cần anchor.

torch.randint: https://docs.pytorch.org/docs/2.14/generated/torch.randint.html
torch.nn.functional.cross_entropy: https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.cross_entropy.html
torch.autograd.grad: https://docs.pytorch.org/docs/2.14/generated/torch.autograd.grad.html
torch.optim.SGD (thuật toán, momentum = 0, weight_decay = 0): https://docs.pytorch.org/docs/2.14/generated/torch.optim.SGD.html
torch.Tensor.add_(other, *, alpha): https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.add_.html
torch.Tensor.item: https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.item.html
torch.no_grad: https://docs.pytorch.org/docs/2.14/generated/torch.no_grad.html
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from dfl.contracts import NodeDataset


def make_anchor(model: torch.nn.Module) -> list[torch.Tensor]:
    """Anchor của số hạng prox: bản clone + detach của mọi tham số, theo thứ tự `model.parameters()`."""
    return [param.detach().clone() for param in model.parameters()]


def prox_term(params: Sequence[torch.Tensor], anchor: Sequence[torch.Tensor], mu: float) -> torch.Tensor:
    """μ/2 · Σ‖w − anchor‖²; gradient theo w là μ(w − anchor), anchor không nhận gradient (T-INV-7)."""
    return mu / 2 * sum(((p - a) ** 2).sum() for p, a in zip(params, anchor, strict=True))


def _check_int(value: object, what: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{what} phải là số nguyên ≥ 1, nhận {value!r}")


@dataclass(frozen=True, slots=True)
class SgdTrainer:
    """LocalTrainer của dự án: τ bước, bước η, prox μ (0 = DFedAvg/D-PSGD), minibatch B (KH §3.5)."""

    tau: int
    lr: float
    mu: float = 0.0
    batch_size: int = 32

    def __post_init__(self) -> None:
        _check_int(self.tau, "tau")
        _check_int(self.batch_size, "batch_size")
        if not (math.isfinite(self.lr) and self.lr > 0):
            raise ValueError(f"lr phải hữu hạn và dương, nhận {self.lr!r}")
        if not (math.isfinite(self.mu) and self.mu >= 0):
            raise ValueError(f"mu phải hữu hạn và không âm, nhận {self.mu!r}")

    def __call__(
        self, model: torch.nn.Module, data: NodeDataset, *, loss_scale: float, generator: torch.Generator
    ) -> float:
        """τ bước tại chỗ trên `model`; trả về mất mát trung bình, hoặc mất mát không hữu hạn đầu tiên.

        Bước có mất mát không hữu hạn không cập nhật tham số, và các bước sau không chạy: engine dừng lượt và
        ghi SummaryDiverged (contracts.LocalTrainer).
        """
        params = list(model.parameters())
        anchor = make_anchor(model) if self.mu else []
        total = 0.0
        for _ in range(self.tau):
            idx = torch.randint(len(data.y), (self.batch_size,), generator=generator)
            loss = loss_scale * torch.nn.functional.cross_entropy(model(data.x[idx]), data.y[idx])
            if self.mu:
                loss = loss + prox_term(params, anchor, self.mu)
            value = loss.item()
            if not math.isfinite(value):
                return value
            grads = torch.autograd.grad(loss, params)
            with torch.no_grad():
                for param, grad in zip(params, grads, strict=True):
                    param.add_(grad, alpha=-self.lr)
            total += value
        return total / self.tau

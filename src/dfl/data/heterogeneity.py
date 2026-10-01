"""Độ đo mức Non-IID của một cách chia dữ liệu (module data; KH §3.2; ADR-005).

Chỉ dùng numpy để chạy trong mili-giây và không nạp torch (contract pure-no-torch, SPEC §3.2).
Mọi độ đo đặt các cách chia khác nhau (Dirichlet, #C = k, quantity) lên cùng một trục, và phải được
báo cáo kèm kích thước node và lệch mục tiêu (G11, G19).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class HeterogeneityStats:
    """Tóm tắt mức Non-IID của một cách chia (các cột của bảng KH §3.2)."""

    # Trung bình qua node của JS(P_i ‖ P), đơn vị bit, trong [0, 1].
    js_mean: float
    # Trung bình qua node của TV(P_i, P).
    tv_mean: float
    # Trung bình qua node của số lớp chiếm ≥ 5% dữ liệu của node.
    classes_ge_5pct_mean: float
    n_min: int
    n_max: int
    # std / mean của kích thước node, std với ddof = 0 (như spike).
    cv_n: float
    # TV(Σ_j π_j P_j, P): khoảng cách giữa phân bố nhãn mà lượt chạy thực sự tối ưu và phân bố toàn cục.
    objective_shift_tv: float
    # N × số lớp: số mẫu mỗi lớp của từng node (ghi vào manifest để vẽ P7).
    node_class_counts: tuple[tuple[int, ...], ...]


def js_divergence_bits(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen–Shannon divergence với log cơ số 2, nên nằm trong [0, 1] và hai phân bố rời nhau cho 1."""
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0  # quy ước 0 · log 0 = 0; trên mask này b ≥ a/2 > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    """Khoảng cách total variation, ½‖p − q‖₁."""
    return float(0.5 * np.abs(p - q).sum())


def class_counts(labels: np.ndarray, parts: Sequence[np.ndarray], num_classes: int = 10) -> np.ndarray:
    """Ma trận node × lớp: số mẫu mỗi lớp trong từng phần (`parts` là các mảng chỉ số trỏ vào `labels`)."""
    return np.array([np.bincount(labels[ix], minlength=num_classes) for ix in parts], dtype=np.int64)


def summarize(
    labels: np.ndarray,
    parts: Sequence[np.ndarray],
    num_classes: int = 10,
    objective_weights: np.ndarray | None = None,
) -> HeterogeneityStats:
    """Tính các độ đo của KH §3.2 cho một cách chia.

    Phân bố toàn cục P là phân bố nhãn của mọi mẫu được chia. `objective_weights` là π của ADR-005
    (MH: 1/N, mặc định; size-weighted và size-corrected: n_j/n); phải không âm, đủ N phần tử, tổng bằng 1.
    """
    counts = class_counts(labels, parts, num_classes)
    sizes = counts.sum(axis=1)
    if (sizes == 0).any():
        raise ValueError("có node rỗng: mọi phần phải có ít nhất một mẫu")
    n_nodes = len(parts)
    if objective_weights is None:
        weights = np.full(n_nodes, 1.0 / n_nodes)
    else:
        weights = np.asarray(objective_weights, dtype=float)
        if weights.shape != (n_nodes,) or (weights < 0).any() or not np.isclose(weights.sum(), 1.0):
            raise ValueError("objective_weights phải không âm, có đúng N phần tử và tổng bằng 1")
    dists = counts / sizes[:, None]
    pooled = counts.sum(axis=0) / sizes.sum()
    return HeterogeneityStats(
        js_mean=float(np.mean([js_divergence_bits(d, pooled) for d in dists])),
        tv_mean=float(np.mean([total_variation(d, pooled) for d in dists])),
        classes_ge_5pct_mean=float(np.mean((dists >= 0.05).sum(axis=1))),
        n_min=int(sizes.min()),
        n_max=int(sizes.max()),
        cv_n=float(sizes.std() / sizes.mean()),
        objective_shift_tv=total_variation(weights @ dists, pooled),
        node_class_counts=tuple(tuple(int(c) for c in row) for row in counts),
    )

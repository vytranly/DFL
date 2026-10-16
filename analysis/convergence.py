"""Độ đo suy ra từ rounds.jsonl: R@x, bytes@x, AUC, độ dao động, mức sụt, độ phân tán giữa node (module
analysis; T15b; SPEC REQ-04, REQ-05, REQ-06, REQ-09; ADR-007; KH §3.6; G24).

- **Lưới:** mọi lượt của lưới đánh giá ở 100, 200, …, τ·R bước (ADR-007). Lưới khác thì báo lỗi, trừ hai ca:
  B1/B1-4k dùng lưới bội của 100 tới hết ngân sách của chính nó (cùng quy tắc), và lượt phân kỳ có lưới là
  phần đầu của lưới chuẩn. Ở --draft, `every` là lịch riêng của profile (INT-1).
- **R@x** (REQ-04): một *lần vượt* là mốc p có acc(p) ≥ x trong khi mốc trước < x (mốc đầu ≥ x cũng là lần
  vượt). Lần vượt *bền vững* khi mọi mốc trong (p, p + 200] ≥ x − 0.5 điểm % và sau p còn ít nhất một mốc.
  R@x nội suy tuyến tính giữa mốc trước p và p, ở lần vượt bền vững đầu tiên. Mốc đầu bền vững ghi "≤ 100"
  (tính là 100 khi lấy trung vị); không có lần vượt bền vững ghi "> ngân sách" (tính là +∞); lượt phân kỳ là
  "không đạt (phân kỳ)".
- **AUC:** trung bình acc theo bước trên lưới, quy tắc hình thang; không xác định với lượt phân kỳ.
- **Độ dao động:** std (ddof = 0) của hiệu số liên tiếp trên 20 mốc cuối, tính trên chuỗi accuracy của từng
  node rồi lấy trung bình qua node (G24). **Mức sụt lớn nhất:** với từng node, max(acc_{k−1} − acc_k)⁺ trên
  các mốc sau 10% đầu của ngân sách; báo trung bình và max qua node.
- So sánh với ngưỡng có sai số 1e-12, để "sụt đúng bằng x − 0.5 điểm %" không trượt vì làm tròn số thực.

numpy.interp, numpy.trapezoid, numpy.median, numpy.std:
https://numpy.org/doc/2.5/reference/generated/numpy.interp.html
https://numpy.org/doc/2.5/reference/generated/numpy.trapezoid.html
https://numpy.org/doc/2.5/reference/generated/numpy.median.html
https://numpy.org/doc/2.5/reference/generated/numpy.std.html
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from analysis.aggregate import AnalysisError, Run

GRID_STEP = 100
WINDOW = 200
TOLERANCE = 0.005  # 0.5 điểm %
EPSILON = 1e-12
LAST_MARKS = 20
SKIP_FRACTION = 0.1
WORST_FRACTION = 0.1


@dataclass(frozen=True)
class Reach:
    """R@x của một lượt: `step` hữu hạn khi đạt (100 với "≤ 100"), +∞ khi không đạt."""

    step: float
    label: str

    @property
    def reached(self) -> bool:
        return math.isfinite(self.step)


def _marks(run: Run) -> list[int]:
    return [int(record["local_steps"]) for record in run.rounds]


def check_grid(run: Run, *, every: int = GRID_STEP) -> None:
    """Lưới mốc của `run` phải là every, 2·every, …, τ·R (phần đầu của nó nếu lượt phân kỳ)."""
    budget = int(run.config["local_steps"])
    expected = list(range(every, budget + 1, every))
    marks = _marks(run)
    ok = marks == expected or (run.status == "diverged" and marks == expected[: len(marks)])
    if not ok:
        raise AnalysisError(
            f"lượt {run.run_id} có lưới mốc {marks[:5]}… khác lưới chuẩn {expected[:5]}… tới {budget} "
            "bước (ADR-007)"
        )


def reach(steps: Sequence[float], acc: Sequence[float], x: float) -> Reach:
    """R@x trên chuỗi (bước, accuracy) theo REQ-04."""
    for i, (step, value) in enumerate(zip(steps, acc, strict=True)):
        if value < x - EPSILON or (i > 0 and acc[i - 1] >= x - EPSILON):
            continue
        after = [a for s, a in zip(steps[i + 1 :], acc[i + 1 :], strict=True) if s <= step + WINDOW]
        if not after or any(a < x - TOLERANCE - EPSILON for a in after):
            continue
        if i == 0:
            return Reach(float(step), "≤ 100")
        before_step, before = steps[i - 1], acc[i - 1]
        at = before_step + (x - before) / (value - before) * (step - before_step)
        return Reach(float(at), f"{at:.0f}")
    return Reach(math.inf, "> ngân sách")


def run_reach(run: Run, x: float, *, every: int = GRID_STEP) -> Reach:
    """R@x của một lượt trên chuỗi `acc_mean`; lượt phân kỳ là không đạt."""
    if run.status == "diverged":
        return Reach(math.inf, "không đạt (phân kỳ)")
    check_grid(run, every=every)
    return reach(_marks(run), [float(r["acc_mean"]) for r in run.rounds], x)


def median_reach(reaches: Sequence[Reach]) -> tuple[float, str]:
    """Trung vị qua seed (không đạt là +∞) và nhãn "k/n seed đạt"."""
    reached = sum(r.reached for r in reaches)
    return float(np.median([r.step for r in reaches])), f"{reached}/{len(reaches)} seed đạt"


def diverged_label(runs: Sequence[Run]) -> str:
    return f"{sum(run.status == 'diverged' for run in runs)}/{len(runs)} seed phân kỳ"


def rounds_at(reached: Reach, *, tau: int) -> float:
    return reached.step / tau


def bytes_at(steps: Sequence[float], bytes_cum: Sequence[float], reached: Reach) -> float:
    """bytes@x: byte tích lũy nội suy tuyến tính tại R@x (REQ-05); +∞ khi không đạt."""
    if not reached.reached:
        return math.inf
    return float(np.interp(reached.step, steps, bytes_cum))


def auc(steps: Sequence[float], acc: Sequence[float]) -> float:
    """Trung bình acc theo bước, quy tắc hình thang giữa mốc đầu và mốc cuối."""
    return float(np.trapezoid(acc, steps) / (steps[-1] - steps[0]))


def run_auc(run: Run, *, every: int = GRID_STEP) -> float | None:
    """AUC trên `acc_mean`; None ("không xác định") với lượt phân kỳ."""
    if run.status == "diverged":
        return None
    check_grid(run, every=every)
    return auc(_marks(run), [float(r["acc_mean"]) for r in run.rounds])


def oscillation(per_node: np.ndarray) -> float:
    """Trung bình qua node của std hiệu số liên tiếp trên LAST_MARKS mốc cuối; `per_node` là (mốc, node)."""
    tail = np.asarray(per_node, dtype=float)[-LAST_MARKS:]
    return float(np.mean(np.std(np.diff(tail, axis=0), axis=0)))


def max_drop(steps: Sequence[float], per_node: np.ndarray, *, budget: float) -> tuple[float, float]:
    """(trung bình, max) qua node của mức sụt lớn nhất giữa hai mốc liên tiếp, sau SKIP_FRACTION ngân
    sách."""
    keep = np.asarray(steps, dtype=float) > SKIP_FRACTION * budget
    series = np.asarray(per_node, dtype=float)[keep]
    drops = np.clip(series[:-1] - series[1:], 0.0, None).max(axis=0) if len(series) > 1 else np.zeros(1)
    return float(drops.mean()), float(drops.max())


def spread(values: Sequence[float]) -> dict[str, float]:
    """Độ phân tán giữa node ở round cuối (REQ-09): std (ddof = 0), min, max − min, trung bình 10% node kém
    nhất."""
    array = np.sort(np.asarray(values, dtype=float))
    worst = max(1, math.ceil(WORST_FRACTION * len(array)))
    return {
        "std": float(array.std()),
        "min": float(array[0]),
        "range": float(array[-1] - array[0]),
        "worst10_mean": float(array[:worst].mean()),
    }


def max_node_load(run: Run) -> int:
    """Tải lớn nhất trên một node (byte gửi + nhận, cộng dồn) ở mốc cuối (REQ-05)."""
    return int(run.rounds[-1]["max_node_bytes_cum"]) if run.rounds else 0

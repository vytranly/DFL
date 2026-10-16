"""Thống kê và quy tắc quyết định đăng ký trước (module analysis; T15c; ADR-007; SPEC REQ-04, REQ-13; plan
K11).

- **CI 95%** của trung bình theo phân phối t (std ddof = 1). Giá trị tới hạn hai phía lấy từ bảng của NIST/
  SEMATECH e-Handbook of Statistical Methods §1.3.6.7.2 (cột 0.975, ν = 1…30; đọc ngày 2026-10-16):
  https://www.itl.nist.gov/div898/handbook/eda/section3/eda3672.htm . Không thêm phụ thuộc chạy (K11): scipy
  chỉ có trong môi trường dev và chỉ test dùng nó để đối chiếu.
- **Quy tắc quyết định** trên hiệu số ghép cặp theo seed (chiều kỳ vọng `expected` = +1 hay −1):
  - 5 seed: "ủng hộ" khi đúng chiều ở cả 5 seed (kiểm định dấu một phía, p = 1/32) và CI không chứa 0;
    "không ủng hộ" khi CI nằm hẳn về phía ngược lại; còn lại "chưa kết luận".
  - 3 seed: tối đa "gợi ý" (đúng chiều ở cả 3 seed và CI không chứa 0), không bao giờ "ủng hộ".
  - Hiệu bằng 0 không tính là đúng chiều. Có seed phân kỳ thì "chưa kết luận" (trừ khi thẻ thí nghiệm quy định
    khác, việc của người gọi).
- **R@x bị kiểm duyệt hai phía** không dùng CI t mà dùng kiểm định dấu trên thứ tự ghép cặp: "> ngân sách"
  lớn hơn mọi giá trị hữu hạn, "≤ 100" nhỏ hơn mọi giá trị lớn hơn 100, hai giá trị cùng bị kiểm duyệt ở một
  cặp là hòa, và hòa tính là không đúng chiều.
- **H2:** với mỗi seed, độ dốc OLS của khoảng cách Acc_FC − Acc_ring theo JS đo được; quyết định bằng quy
  tắc 5 seed trên các độ dốc (CI t với df = 4).
- **Khuyến nghị (REQ-13):** "ủng hộ" → khuyến nghị chính; "gợi ý" → khuyến nghị tạm; còn lại không khuyến
  nghị.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from analysis.aggregate import AnalysisError
from analysis.convergence import Reach

SUPPORTED, SUGGESTED, REJECTED, OPEN = "ủng hộ", "gợi ý", "không ủng hộ", "chưa kết luận"
DIVERGED_LABEL = "không đạt (phân kỳ)"
# NIST/SEMATECH e-Handbook §1.3.6.7.2, giá trị tới hạn t_{0.975, ν}.
T975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
    20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048,
    29: 2.045, 30: 2.042,
}  # fmt: skip


def t_critical(df: int) -> float:
    if df not in T975:
        raise AnalysisError(f"bảng t chỉ có df 1…30, nhận df = {df}")
    return T975[df]


@dataclass(frozen=True)
class Estimate:
    mean: float
    low: float
    high: float
    n: int


def mean_ci(values: Sequence[float]) -> Estimate:
    """Trung bình kèm CI 95% theo phân phối t."""
    array = np.asarray(values, dtype=float)
    if len(array) < 2:
        raise AnalysisError("CI cần ít nhất 2 giá trị")
    half = t_critical(len(array) - 1) * array.std(ddof=1) / math.sqrt(len(array))
    mean = float(array.mean())
    return Estimate(mean, mean - half, mean + half, len(array))


def paired_differences(
    a: Mapping[int, float], b: Mapping[int, float], *, seeds: Sequence[int]
) -> list[float]:
    """a − b theo từng seed của tập seed đã khai báo; thiếu seed là lỗi (INT-2, INT-3)."""
    missing = [s for s in seeds if s not in a or s not in b]
    if missing:
        raise AnalysisError(
            f"thiếu seed {missing[0]} trong so sánh ghép cặp (tập seed khai báo {list(seeds)})"
        )
    return [a[s] - b[s] for s in seeds]


def decide(differences: Sequence[float], *, expected: int, diverged: bool = False) -> str:
    """Kết luận theo ADR-007 cho hiệu số ghép cặp của 5 hay 3 seed."""
    n = len(differences)
    if n not in (5, 3):
        raise AnalysisError(f"quy tắc quyết định chỉ có cho 5 hay 3 seed, nhận {n}")
    if diverged:
        return OPEN
    estimate = mean_ci(differences)
    right = all(d * expected > 0 for d in differences)
    beyond_zero = estimate.low > 0 if expected > 0 else estimate.high < 0
    opposite = estimate.high < 0 if expected > 0 else estimate.low > 0
    if right and beyond_zero:
        return SUPPORTED if n == 5 else SUGGESTED
    return REJECTED if opposite else OPEN


def _rank(reach: Reach) -> tuple[float, str]:
    if reach.label == "≤ 100":
        return reach.step, "low"
    if not reach.reached:
        return math.inf, "high"
    return reach.step, ""


def compare_reach(a: Reach, b: Reach) -> int:
    """−1 khi a đạt nhanh hơn b, +1 khi chậm hơn, 0 khi hòa (kể cả hai giá trị cùng bị kiểm duyệt)."""
    (va, ca), (vb, cb) = _rank(a), _rank(b)
    if ca and ca == cb:
        return 0
    return -1 if va < vb else 1 if va > vb else 0


def decide_speed(faster: Sequence[Reach], slower: Sequence[Reach]) -> tuple[str, str]:
    """Kiểm định dấu cho giả thuyết "`faster` đạt R@x sớm hơn `slower`" trên các cặp cùng seed."""
    if len(faster) != len(slower) or len(faster) not in (5, 3):
        raise AnalysisError("kiểm định dấu cần 5 hay 3 cặp ghép theo seed")
    orders = [compare_reach(a, b) for a, b in zip(faster, slower, strict=True)]
    count = f"{sum(o < 0 for o in orders)}/{len(orders)} cặp đúng chiều"
    if any(r.label == DIVERGED_LABEL for r in (*faster, *slower)):
        return OPEN, count
    if all(o < 0 for o in orders):
        return (SUPPORTED if len(orders) == 5 else SUGGESTED), count
    if all(o > 0 for o in orders):
        return REJECTED, count
    return OPEN, count


def h2_slopes(points: Mapping[int, Sequence[tuple[float, float]]]) -> list[float]:
    """Độ dốc OLS của (JS, Acc_FC − Acc_ring) cho từng seed, theo thứ tự seed tăng dần."""
    slopes = []
    for seed in sorted(points):
        x, y = (np.asarray(v, dtype=float) for v in zip(*points[seed], strict=True))
        slopes.append(float(np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)))
    return slopes


def recommendation(verdict: str) -> str:
    if verdict == SUPPORTED:
        return "khuyến nghị chính"
    if verdict == SUGGESTED:
        return "khuyến nghị tạm (cần thêm seed để khẳng định)"
    return "không khuyến nghị"

"""Test cho analysis/stats.py: CI t, hiệu số ghép cặp, quy tắc quyết định, kiểm định dấu cho R@x bị kiểm
duyệt, độ dốc H2, mức khuyến nghị (T15c; ADR-007; SPEC REQ-04, REQ-13; plan K11).

scipy chỉ có trong môi trường dev và chỉ được dùng ở test làm oracle; analysis/ không import nó (no-dev-deps).
"""

import math

import numpy as np
import pytest
from scipy import stats as scipy_stats

from analysis import stats
from analysis.aggregate import AnalysisError
from analysis.convergence import Reach


@pytest.mark.parametrize("df", range(1, 31))
def test_t_critical_values_match_scipy_to_three_decimals(df: int) -> None:
    assert stats.t_critical(df) == pytest.approx(scipy_stats.t.ppf(0.975, df), abs=5e-4)


def test_t_critical_refuses_degrees_of_freedom_outside_the_table() -> None:
    with pytest.raises(AnalysisError, match="df"):
        stats.t_critical(31)


def test_mean_ci_is_the_t_interval_of_the_sample() -> None:
    values = [0.12, 0.08, 0.15, 0.11, 0.09]

    estimate = stats.mean_ci(values)

    low, high = scipy_stats.t.interval(0.95, 4, loc=np.mean(values), scale=scipy_stats.sem(values))
    assert estimate.mean == pytest.approx(np.mean(values))
    assert (estimate.low, estimate.high) == (pytest.approx(low, abs=1e-4), pytest.approx(high, abs=1e-4))


def test_paired_differences_follow_the_seeds_and_refuse_a_missing_one() -> None:
    a, b = {0: 0.9, 1: 0.8, 2: 0.85}, {0: 0.85, 1: 0.82, 2: 0.80}

    assert stats.paired_differences(a, b, seeds=[0, 1, 2]) == pytest.approx([0.05, -0.02, 0.05])
    with pytest.raises(AnalysisError, match="seed 3"):
        stats.paired_differences(a, b, seeds=[0, 1, 2, 3])


@pytest.mark.parametrize(
    ("differences", "expected", "verdict"),
    [
        ([0.05, 0.04, 0.06, 0.05, 0.03], 1, "ủng hộ"),  # 5/5 đúng chiều, CI không chứa 0
        ([0.05, 0.04, -0.01, 0.05, 0.03], 1, "chưa kết luận"),  # 4/5 đúng chiều
        ([0.05, 0.04, 0.0, 0.05, 0.03], 1, "chưa kết luận"),  # hiệu bằng 0 không tính là đúng chiều
        ([-0.05, -0.04, -0.06, -0.05, -0.03], 1, "không ủng hộ"),  # CI nằm hẳn về phía ngược lại
        ([-0.05, -0.04, -0.06, -0.05, -0.03], -1, "ủng hộ"),  # chiều kỳ vọng âm
        ([0.05, 0.04, 0.06], 1, "gợi ý"),  # 3 seed: tối đa là gợi ý
        ([-0.05, -0.04, -0.06], 1, "không ủng hộ"),
        ([0.05, 0.001, 0.06], 1, "chưa kết luận"),  # 3/3 đúng chiều nhưng CI chứa 0
    ],
)
def test_decision_rule_of_adr_007(differences: list[float], expected: int, verdict: str) -> None:
    assert stats.decide(differences, expected=expected) == verdict


def test_a_diverged_seed_makes_the_conclusion_open() -> None:
    assert stats.decide([0.05, 0.04, 0.06, 0.05, 0.03], expected=1, diverged=True) == "chưa kết luận"


def test_only_five_or_three_seeds_have_a_decision_rule() -> None:
    with pytest.raises(AnalysisError, match="5 hay 3"):
        stats.decide([0.05, 0.04, 0.06, 0.05], expected=1)


def reach(label: str) -> Reach:
    if label == "> ngân sách":
        return Reach(math.inf, label)
    if label == "≤ 100":
        return Reach(100.0, label)
    return Reach(float(label), label)


@pytest.mark.parametrize(
    ("a", "b", "order"),
    [
        ("350", "450", -1),
        ("450", "> ngân sách", -1),  # "> ngân sách" lớn hơn mọi giá trị hữu hạn
        ("≤ 100", "150", -1),  # "≤ 100" nhỏ hơn mọi giá trị lớn hơn 100
        ("> ngân sách", "> ngân sách", 0),  # cùng bị kiểm duyệt: hòa
        ("≤ 100", "≤ 100", 0),
        ("300", "300", 0),
    ],
)
def test_censored_reaches_are_ordered_like_adr_007(a: str, b: str, order: int) -> None:
    assert stats.compare_reach(reach(a), reach(b)) == order


def test_sign_test_on_censored_reaches_needs_every_pair_in_the_expected_direction() -> None:
    faster = [reach(v) for v in ("300", "350", "≤ 100", "400", "450")]
    slower = [reach(v) for v in ("400", "> ngân sách", "150", "500", "> ngân sách")]

    assert stats.decide_speed(faster, slower) == ("ủng hộ", "5/5 cặp đúng chiều")
    tie = [*faster[:4], reach("> ngân sách")]
    assert stats.decide_speed(tie, slower) == ("chưa kết luận", "4/5 cặp đúng chiều")  # hòa không đúng chiều
    assert stats.decide_speed(faster[:3], slower[:3]) == ("gợi ý", "3/3 cặp đúng chiều")
    assert stats.decide_speed(slower, faster)[0] == "không ủng hộ"


def test_a_diverged_reach_makes_the_speed_conclusion_open() -> None:
    diverged = Reach(math.inf, "không đạt (phân kỳ)")
    faster = [reach(v) for v in ("300", "350", "400", "400", "450")]

    assert stats.decide_speed([*faster[:4], diverged], [reach("500")] * 5)[0] == "chưa kết luận"


def test_h2_slopes_are_ols_per_seed_and_decided_with_df_4() -> None:
    js = [0.001, 0.002, 0.18, 0.29, 0.55, 0.64]
    gaps = {s: [0.001 + (0.05 + 0.01 * s) * x for x in js] for s in range(5)}

    slopes = stats.h2_slopes({s: list(zip(js, g, strict=True)) for s, g in gaps.items()})

    assert slopes == pytest.approx([np.polyfit(js, gaps[s], 1)[0] for s in range(5)])
    assert stats.decide(slopes, expected=1) == "ủng hộ"


def test_recommendation_level_follows_the_verdict() -> None:
    assert stats.recommendation("ủng hộ") == "khuyến nghị chính"
    assert stats.recommendation("gợi ý") == "khuyến nghị tạm (cần thêm seed để khẳng định)"
    assert (
        stats.recommendation("chưa kết luận") == stats.recommendation("không ủng hộ") == "không khuyến nghị"
    )

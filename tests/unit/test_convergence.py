"""Test cho analysis/convergence.py: R@x, AUC, bytes@x, độ dao động, mức sụt, độ phân tán cuối (T15b; SPEC
REQ-04, REQ-05, REQ-06, REQ-09; ADR-007; KH §3.6; G24).

Ví dụ R@x lấy từ docs/decisions/REVIEW-C3-P3.7.md §5 (lưới 100, 200, … bước; x = 0.95): đáp án được tính tay
ở đó trước khi có code.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from analysis import convergence as cv
from analysis.aggregate import AnalysisError, read_run
from tests.analysis_fakes import config, curve, round_record, write_run


def steps(n: int) -> list[int]:
    return [100 * (i + 1) for i in range(n)]


@pytest.mark.parametrize(
    ("acc", "expected"),
    [
        ([0.80, 0.90, 0.94, 0.96, 0.955, 0.96], 350.0),  # 1: nội suy giữa 300 và 400
        ([0.90, 0.96, 0.93, 0.94, 0.96, 0.96, 0.97], 450.0),  # 4: lần vượt không bền bị bỏ qua
        ([0.90, 0.96, 0.945, 0.95], 100 + 100 * 0.05 / 0.06),  # 5: sụt đúng bằng x − 0.5 điểm % vẫn bền
        ([0.90, 0.96, 0.95, 0.944, 0.96, 0.96], 437.5),  # 6a: sụt ở p + 200 làm lần vượt không bền
        ([0.90, 0.96, 0.95, 0.95, 0.944], 100 + 100 * 0.05 / 0.06),  # 6b: sụt ở p + 300 không ảnh hưởng
        ([0.90, 0.96, 0.95], 100 + 100 * 0.05 / 0.06),  # 8: vượt ở mốc kế cuối chỉ cần mốc còn lại
        ([0.90, 0.95, 0.95, 0.95], 200.0),  # 9: bằng đúng x là vượt
    ],
    ids=[
        "interpolate",
        "skip-unsustained",
        "drop-at-tolerance",
        "drop-in-window",
        "drop-after-window",
        "second-to-last",
        "equal-to-x",
    ],
)
def test_reach_is_the_first_sustained_crossing_interpolated_linearly(
    acc: list[float], expected: float
) -> None:
    reach = cv.reach(steps(len(acc)), acc, 0.95)

    assert reach.reached and reach.step == pytest.approx(expected, rel=1e-12)


def test_reach_is_censored_at_both_ends() -> None:
    first = cv.reach(steps(4), [0.96, 0.97, 0.96, 0.98], 0.95)  # 3: mốc đầu đã ≥ x và bền
    never = cv.reach(steps(4), [0.80, 0.90, 0.94, 0.93], 0.95)  # 2: không bao giờ đạt
    last_only = cv.reach(steps(3), [0.90, 0.92, 0.96], 0.95)  # 7: chỉ vượt ở mốc cuối

    assert (first.label, first.step) == ("≤ 100", 100.0)
    assert (never.label, never.step) == ("> ngân sách", math.inf) and not never.reached
    assert last_only.label == "> ngân sách"


def test_median_over_seeds_treats_over_budget_as_infinity_and_reports_k_of_n() -> None:
    # 13: [350, "> ngân sách", "≤ 100", 450, 500] → trung vị 450, "4/5 seed đạt".
    reaches = [
        cv.reach(steps(6), [0.80, 0.90, 0.94, 0.96, 0.955, 0.96], 0.95),
        cv.reach(steps(3), [0.80, 0.90, 0.91], 0.95),
        cv.reach(steps(3), [0.96, 0.96, 0.96], 0.95),
        cv.reach(steps(7), [0.90, 0.96, 0.93, 0.94, 0.96, 0.96, 0.97], 0.95),
        cv.reach(steps(7), [0.90, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97], 0.95),
    ]

    median, label = cv.median_reach(reaches)

    assert median == pytest.approx(450.0) and label == "4/5 seed đạt"


def test_auc_is_the_trapezoid_mean_of_acc_over_steps() -> None:
    # 14: [0.5, 0.7, 0.9, 1.0] trên 100 … 400 → 0.78333…
    assert cv.auc(steps(4), [0.5, 0.7, 0.9, 1.0]) == pytest.approx(0.783333333333, rel=1e-9)


def test_reach_converts_to_rounds_and_bytes() -> None:
    # 12: R@x quy ra round (chia τ); bytes@x nội suy tuyến tính trên bytes_cum.
    reach = cv.reach(steps(5), [0.80, 0.90, 0.94, 0.96, 0.96], 0.95)  # 350

    assert reach.step == pytest.approx(350.0)
    assert cv.rounds_at(reach, tau=20) == pytest.approx(17.5)
    assert cv.bytes_at(steps(5), [1000, 2000, 3000, 4000, 5000], reach) == pytest.approx(3500.0)
    assert cv.bytes_at(steps(4), [1000, 2000, 3000, 4000], cv.reach(steps(2), [0.1, 0.2], 0.95)) == math.inf


def test_runs_use_the_standard_grid_except_baselines_and_diverged_prefixes(tmp_path: Path) -> None:
    # 11: lưới khác lưới 100, 200, …, τ·R thì báo lỗi; B1 dùng lưới của chính ngân sách nó; lượt phân kỳ có
    # lưới là phần đầu của lưới chuẩn.
    write_run(tmp_path, "ok", rounds=curve([0.5, 0.7, 0.8, 0.85]))
    write_run(tmp_path, "gap", rounds=[round_record(s, 20, [0.5] * 4) for s in (100, 200, 400)])
    write_run(
        tmp_path,
        "b1",
        cfg=config(num_nodes=1, local_steps=600, topology__kind="central"),
        rounds=curve([0.5, 0.6, 0.7, 0.8, 0.9, 0.95], nodes=1),
    )
    write_run(tmp_path, "div", status="diverged", rounds=curve([0.5, 0.6]))

    for run_id in ("ok", "b1", "div"):
        cv.check_grid(read_run(tmp_path / "runs" / run_id).finished)
    with pytest.raises(AnalysisError, match="lưới"):
        cv.check_grid(read_run(tmp_path / "runs" / "gap").finished)


def test_a_diverged_run_does_not_reach_and_has_no_auc(tmp_path: Path) -> None:
    # 10: R@x "không đạt", AUC "không xác định", và lượt được đếm vào "k/n seed phân kỳ".
    write_run(tmp_path, "div", status="diverged", rounds=curve([0.96, 0.97]))
    write_run(tmp_path, "ok", rounds=curve([0.8, 0.9, 0.96, 0.97]))
    div = read_run(tmp_path / "runs" / "div").finished
    ok = read_run(tmp_path / "runs" / "ok").finished

    reach = cv.run_reach(div, 0.95)

    assert not reach.reached and reach.label == "không đạt (phân kỳ)"
    assert cv.run_auc(div) is None and cv.run_auc(ok) is not None
    assert cv.diverged_label([div, ok]) == "1/2 seed phân kỳ"


def test_oscillation_is_computed_per_node_before_averaging() -> None:
    # 15 (G24): hai node dao động ngược pha thì acc_mean phẳng, nhưng độ dao động theo node > 0.
    a = np.array([0.80, 0.90] * 10)
    per_node = np.stack([a, 1.7 - a], axis=1)  # cột là node

    assert cv.oscillation(per_node) > 0.09
    assert cv.oscillation(per_node.mean(axis=1, keepdims=True)) == pytest.approx(0.0, abs=1e-12)


def test_oscillation_uses_only_the_last_20_marks() -> None:
    calm = np.full(20, 0.9)
    per_node = np.concatenate([np.array([0.1, 0.9, 0.1, 0.9]), calm])[:, None]

    assert cv.oscillation(per_node) == pytest.approx(0.0, abs=1e-12)


def test_max_drop_skips_the_first_tenth_of_the_budget_and_reports_mean_and_max_over_nodes() -> None:
    marks = steps(10)  # ngân sách 1,000: bỏ các mốc ≤ 100
    node0 = [0.9, 0.2, 0.5, 0.6, 0.6, 0.55, 0.6, 0.7, 0.7, 0.7]  # sụt lớn ở 100 → 200 bị bỏ; sau đó 0.05
    node1 = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.6, 0.9]  # sụt 0.3

    mean, worst = cv.max_drop(marks, np.array([node0, node1]).T, budget=1000)

    assert worst == pytest.approx(0.3) and mean == pytest.approx((0.05 + 0.3) / 2)


def test_final_spread_has_std_min_range_and_mean_of_the_worst_tenth() -> None:
    values = [0.95] * 18 + [0.70, 0.80]

    spread = cv.spread(values)

    assert spread["min"] == 0.70 and spread["range"] == pytest.approx(0.25)
    assert spread["worst10_mean"] == pytest.approx(0.75)  # 10% của 20 node là 2 node kém nhất
    assert spread["std"] == pytest.approx(float(np.std(values)))


def test_max_node_load_is_the_last_cumulative_value(tmp_path: Path) -> None:
    write_run(tmp_path, "r", rounds=curve([0.5, 0.7]))

    assert cv.max_node_load(read_run(tmp_path / "runs" / "r").finished) == 4 * 10 * 1000

"""Test cho analysis/checks.py: mỗi luật của INT-2, INT-3, INT-6 có một ca báo lỗi rõ ràng và một ca hợp lệ
(T15a; SPEC §9.3).
"""

from pathlib import Path

import pandas as pd
import pytest

from analysis import aggregate, checks
from tests.analysis_fakes import CODE_VERSION, config, write_index, write_run


def runs(tmp_path: Path, specs: dict[str, dict[str, object]]) -> dict[str, aggregate.Run]:
    """Ghi các lượt theo `specs` (run_id → tham số của write_run), đọc lại qua chỉ mục; run_id → lượt."""
    for run_id, kwargs in specs.items():
        write_run(tmp_path, run_id, **kwargs)
    write_index(tmp_path, "e", {CODE_VERSION: list(specs)})
    return {a.run_id: a.finished for a in aggregate.read_experiment(tmp_path, "e") if a.finished is not None}


# ---------------------------------------------------------------------------------------------- INT-2


def test_mixed_code_versions_are_refused_and_a_single_one_is_returned(tmp_path: Path) -> None:
    write_run(tmp_path, "a", code_version="a" * 64)
    write_run(tmp_path, "b")
    mixed = [s.finished for s in aggregate.scan(tmp_path)]

    with pytest.raises(checks.AnalysisError, match="code_version"):
        checks.single_code_version(mixed)
    assert checks.single_code_version(mixed[1:]) == CODE_VERSION


@pytest.mark.parametrize("flag", [{"dirty": True}, {"env_match": False}], ids=["dirty", "env-mismatch"])
def test_dirty_or_mismatched_runs_are_refused_unless_draft(tmp_path: Path, flag: dict[str, bool]) -> None:
    selected = runs(tmp_path, {"a": {}, "b": flag})

    with pytest.raises(checks.AnalysisError, match="b"):
        checks.check_clean(selected.values(), draft=False)
    checks.check_clean(selected.values(), draft=True)
    checks.check_clean([selected["a"]], draft=False)


@pytest.mark.parametrize(
    ("override", "name"),
    [
        ({"algorithm__lr": 0.1}, "lr"),
        ({"local_steps": 800}, "local_steps"),
        ({"eval__every_steps": 200}, "every_steps"),
        ({"dataset": "cifar10"}, "dataset"),
    ],
    ids=["lr", "budget", "schedule", "dataset"],
)
def test_a_differing_fixed_variable_is_refused_unless_it_is_the_axis(
    tmp_path: Path, override: dict[str, object], name: str
) -> None:
    selected = list(runs(tmp_path, {"a": {}, "b": {"cfg": config(**override)}}).values())

    with pytest.raises(checks.AnalysisError, match=name):
        checks.check_fixed(selected)
    checks.check_fixed(selected, axes={name})


def test_runs_from_another_machine_are_a_differing_fixed_variable(tmp_path: Path) -> None:
    selected = list(runs(tmp_path, {"a": {}, "b": {"machine": "Other CPU (flags ffff)"}}).values())

    with pytest.raises(checks.AnalysisError, match="machine"):
        checks.check_fixed(selected)


def test_baselines_drawn_as_references_are_exempt_but_the_server_path_is_not(tmp_path: Path) -> None:
    selected = runs(
        tmp_path,
        {
            "dfl": {},
            "b1": {"cfg": config(num_nodes=1, local_steps=17_200, topology__kind="central")},
            "b3": {"cfg": config(algorithm__lr=0.1, topology__kind="server", mixing="size_weighted")},
        },
    )
    dfl, b1, b3 = selected["dfl"], selected["b1"], selected["b3"]

    checks.check_fixed([dfl, b1], references={"b1"})
    with pytest.raises(checks.AnalysisError, match="server"):
        checks.check_fixed([dfl, b3], references={"b3"})
    with pytest.raises(checks.AnalysisError, match="tham chiếu"):
        checks.check_fixed([b1, dfl], references={"dfl"})


def test_different_seed_sets_need_a_declared_common_set(tmp_path: Path) -> None:
    selected = runs(
        tmp_path,
        {
            "ring0": {"cfg": config(seed=0)},
            "ring1": {"cfg": config(seed=1)},
            "fc0": {"cfg": config(seed=0, topology__kind="fully_connected")},
        },
    )
    groups = {"ring": [selected["ring0"], selected["ring1"]], "fc": [selected["fc0"]]}

    with pytest.raises(checks.AnalysisError, match="seed"):
        checks.select_seeds(groups, declared=None)
    chosen, missing = checks.select_seeds(groups, declared={0, 1})
    assert {name: [r.run_id for r in group] for name, group in chosen.items()} == {
        "ring": ["ring0", "ring1"],
        "fc": ["fc0"],
    }
    assert missing == [("fc", 1)]  # báo cáo theo INT-3, không lặng lẽ bỏ
    same, none_missing = checks.select_seeds(
        {"ring": [selected["ring0"]], "fc": [selected["fc0"]]}, declared=None
    )
    assert none_missing == [] and len(same["fc"]) == 1


# ---------------------------------------------------------------------------------------------- INT-3


def test_nan_in_a_finished_run_is_refused_but_metrics_of_diverged_or_missing_runs_may_be_empty() -> None:
    frame = pd.DataFrame(
        {
            "run_id": ["ok", "div", "missing"],
            "status": ["ok", "diverged", "missing"],
            "final_acc_avg_model": [0.9, None, None],
        }
    )

    checks.check_no_nan(frame)
    frame.loc[0, "final_acc_avg_model"] = float("nan")
    with pytest.raises(checks.AnalysisError, match="NaN"):
        checks.check_no_nan(frame)


def test_attempt_report_lists_failed_and_diverged_attempts_with_reasons(tmp_path: Path) -> None:
    write_run(tmp_path, "r1", status="failed", attempt=1)
    write_run(tmp_path, "r1", status="ok", attempt=2)
    write_run(tmp_path, "r2", status="diverged")
    write_index(tmp_path, "e", {CODE_VERSION: ["r1", "r2"]})

    report = checks.attempt_report(aggregate.read_experiment(tmp_path, "e"))

    assert report == [
        {"run_id": "r1", "attempt": 1, "status": "failed", "reason": "MemoryError: giả"},
        {
            "run_id": "r2",
            "attempt": 1,
            "status": "diverged",
            "reason": "mất mát huấn luyện của node 0 không hữu hạn (nan)",
        },
    ]


# ---------------------------------------------------------------------------------------------- INT-6


def test_every_code_version_needs_a_line_in_code_versions_md(tmp_path: Path) -> None:
    ledger = tmp_path / "CODE_VERSIONS.md"

    with pytest.raises(checks.AnalysisError, match="CODE_VERSIONS"):
        checks.check_documented({CODE_VERSION}, ledger)
    ledger.write_text(f"| 2026-10-16 | abc1234 | {CODE_VERSION} | E0a |\n", encoding="utf-8")
    checks.check_documented({CODE_VERSION}, ledger)
    with pytest.raises(checks.AnalysisError, match="a" * 64):
        checks.check_documented({CODE_VERSION, "a" * 64}, ledger)

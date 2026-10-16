"""Test cho analysis/aggregate.py: đọc results/ qua chỉ mục nhóm thí nghiệm, mỗi lượt một dòng, đếm attempt
failed và bị ngắt, không bỏ lượt phân kỳ hay thiếu (T15a; SPEC §5.3, INT-2, INT-3; ADR-008).
"""

from pathlib import Path

import pytest

from analysis import aggregate
from tests.analysis_fakes import CODE_VERSION, config, curve, write_index, write_run


def test_frame_has_one_row_per_indexed_run_with_its_config_and_final_metrics(tmp_path: Path) -> None:
    write_run(tmp_path, "r1", cfg=config(seed=0))
    write_run(tmp_path, "r2", cfg=config(seed=1, algorithm__lr=0.1, topology__kind="fully_connected"))
    write_run(tmp_path, "unlisted")
    write_index(tmp_path, "e1", {CODE_VERSION: ["r1", "r2"]})

    frame = aggregate.frame(aggregate.read_experiment(tmp_path, "e1"))

    assert list(frame["run_id"]) == ["r1", "r2"]
    assert list(frame["seed"]) == [0, 1] and list(frame["lr"]) == [0.05, 0.1]
    assert list(frame["topology"]) == ["ring", "fully_connected"]
    assert list(frame["status"]) == ["ok", "ok"] and list(frame["final_acc_avg_model"]) == [0.9, 0.9]


def test_failed_and_interrupted_attempts_are_counted_and_the_finished_one_is_read(tmp_path: Path) -> None:
    write_run(tmp_path, "r1", status="failed", attempt=1)
    write_run(tmp_path, "r1", status=None, attempt=2)
    write_run(tmp_path, "r1", status="ok", attempt=3)
    write_index(tmp_path, "e1", {CODE_VERSION: ["r1"]})

    (attempts,) = aggregate.read_experiment(tmp_path, "e1")

    assert attempts.finished is not None and attempts.finished.path.name == "attempt-3"
    assert attempts.failed == [(1, "MemoryError: giả")] and attempts.interrupted == [2]


def test_runs_without_a_finished_attempt_and_diverged_runs_stay_in_the_frame(tmp_path: Path) -> None:
    # INT-3: không lượt nào bị lọc khỏi bảng; lượt thiếu hay phân kỳ được đánh dấu, không bị bỏ.
    write_run(tmp_path, "ok")
    write_run(tmp_path, "div", status="diverged")
    write_index(tmp_path, "e1", {CODE_VERSION: ["div", "missing", "ok"]})

    frame = aggregate.frame(aggregate.read_experiment(tmp_path, "e1")).set_index("run_id")

    assert frame.loc["div", "status"] == "diverged" and frame.loc["div", "diverged_at_round"] == 5
    assert frame.loc["missing", "status"] == "missing"
    assert frame.loc["ok", "status"] == "ok"


def test_an_index_with_several_code_versions_needs_one_chosen(tmp_path: Path) -> None:
    write_run(tmp_path, "old", code_version="a" * 64)
    write_run(tmp_path, "new")
    write_index(tmp_path, "e1", {"a" * 64: ["old"], CODE_VERSION: ["new"]})

    with pytest.raises(aggregate.AnalysisError, match="code_version"):
        aggregate.read_experiment(tmp_path, "e1")
    (chosen,) = aggregate.read_experiment(tmp_path, "e1", code_version=CODE_VERSION)
    assert chosen.run_id == "new"


def test_two_finished_attempts_of_one_run_id_are_an_error(tmp_path: Path) -> None:
    write_run(tmp_path, "r1", attempt=1)
    write_run(tmp_path, "r1", attempt=2, status="diverged")
    write_index(tmp_path, "e1", {CODE_VERSION: ["r1"]})

    with pytest.raises(aggregate.AnalysisError, match="r1"):
        aggregate.read_experiment(tmp_path, "e1")


def test_a_nan_in_a_result_file_is_refused_when_reading(tmp_path: Path) -> None:
    # INT-3: kết quả không bao giờ chứa NaN (ghi với allow_nan=False); gặp NaN là file hỏng, phải báo lỗi.
    attempt = write_run(tmp_path, "r1")
    path = attempt / "rounds.jsonl"
    path.write_text(path.read_text().replace('"acc_mean": 0.5', '"acc_mean": NaN'), encoding="utf-8")
    write_index(tmp_path, "e1", {CODE_VERSION: ["r1"]})

    with pytest.raises(aggregate.AnalysisError, match="NaN"):
        aggregate.read_experiment(tmp_path, "e1")


def test_a_missing_index_names_the_experiment(tmp_path: Path) -> None:
    with pytest.raises(aggregate.AnalysisError, match="e9"):
        aggregate.read_experiment(tmp_path, "e9")


def test_scan_reads_every_run_of_a_directory_without_index(tmp_path: Path) -> None:
    # --draft (smoke, demo): `run --out` không ghi chỉ mục, nên đọc mọi runs/* của thư mục.
    write_run(tmp_path, "b", rounds=curve([0.4, 0.6]))
    write_run(tmp_path, "a")

    assert [attempts.run_id for attempts in aggregate.scan(tmp_path)] == ["a", "b"]


def test_command_line_writes_one_csv_per_experiment(tmp_path: Path) -> None:
    write_run(tmp_path, "r1")
    write_index(tmp_path, "e1", {CODE_VERSION: ["r1"]})

    code = aggregate.main(["--results", str(tmp_path), "--out", str(tmp_path / "out")])

    assert code == 0
    text = (tmp_path / "out" / "e1.csv").read_text(encoding="utf-8")
    assert text.splitlines()[0].startswith("run_id,") and "r1" in text

"""Test cho src/dfl/io/run_logger.py: attempt tạo nguyên tử, thứ tự start → log → finish, JSON không NaN,
schema bản ghi đúng contracts.py (T11a; SPEC §5.3; ADR-008; INV-6).
"""

import json
import math
from pathlib import Path

import pytest

from dfl.contracts import (
    ManifestRecord,
    RoundRecord,
    RunLogger,
    SummaryDiverged,
    SummaryFailed,
    SummaryOk,
    TimingRecord,
)
from dfl.io.run_logger import JsonRunLogger


def manifest(attempt: int = 1) -> ManifestRecord:
    record = dict.fromkeys(ManifestRecord.__required_keys__, 0)
    record.update(run_id="0123456789abcdef", attempt=attempt, config={"num_nodes": 4}, spectral_gap=None)
    return record


def round_record(round_number: int, **overrides: object) -> RoundRecord:
    record: dict[str, object] = dict.fromkeys(RoundRecord.__required_keys__, 0.5)
    record.update(round=round_number, local_steps=2 * round_number, acc_per_node=[0.5, 0.25], **overrides)
    return record


def timing(round_number: int) -> TimingRecord:
    return {
        "round": round_number,
        "wall_time_s": 1.5,
        "train_s": 1.0,
        "combine_s": 0.25,
        "eval_s": 0.25,
        "comm_s": 0.0,
    }


OK: SummaryOk = {
    "status": "ok",
    "messages_total": 48,
    "bytes_total": 4800,
    "final_eval_split": "test",
    "final_eval_size": 500,
    "final_acc_per_node": [0.9, 0.8],
    "final_acc_local_per_node": [0.9, 0.8],
    "final_f1_macro_per_node": [0.85, 0.75],
    "final_acc_avg_model": 0.88,
}
DIVERGED: SummaryDiverged = {
    "status": "diverged",
    "diverged_at_round": 3,
    "reason": "mất mát không hữu hạn ở node 2",
    "messages_total": 16,
    "bytes_total": 1600,
}


def read_lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_an_attempt_writes_manifest_rounds_timing_and_summary(tmp_path: Path) -> None:
    logger = JsonRunLogger(tmp_path / "runs" / "0123456789abcdef")

    logger.start(manifest())
    logger.log_round(round_record(1))
    logger.log_timing(timing(1))
    logger.log_round(round_record(2))
    logger.finish(DIVERGED)

    attempt = tmp_path / "runs" / "0123456789abcdef" / "attempt-1"
    assert sorted(p.name for p in attempt.iterdir()) == [
        "manifest.json",
        "rounds.jsonl",
        "summary.json",
        "timing.jsonl",
    ]
    assert json.loads((attempt / "manifest.json").read_text(encoding="utf-8")) == manifest()
    assert read_lines(attempt / "rounds.jsonl") == [round_record(1), round_record(2)]
    assert read_lines(attempt / "timing.jsonl") == [timing(1)]
    assert json.loads((attempt / "summary.json").read_text(encoding="utf-8")) == DIVERGED
    assert logger.attempt_dir == attempt


def test_the_same_records_give_byte_identical_rounds_files(tmp_path: Path) -> None:
    # INV-6: rounds.jsonl chỉ phụ thuộc nội dung bản ghi, không phụ thuộc thứ tự khóa trong dict.
    first, second = JsonRunLogger(tmp_path / "a"), JsonRunLogger(tmp_path / "b")
    record = round_record(1)
    reordered = dict(reversed(list(record.items())))

    for logger, rec in ((first, record), (second, reordered)):
        logger.start(manifest())
        logger.log_round(rec)
        logger.finish(OK)

    assert (tmp_path / "a" / "attempt-1" / "rounds.jsonl").read_bytes() == (
        tmp_path / "b" / "attempt-1" / "rounds.jsonl"
    ).read_bytes()


def test_starting_an_existing_attempt_raises_file_exists_and_leaves_it_untouched(tmp_path: Path) -> None:
    existing = tmp_path / "attempt-1"
    existing.mkdir()
    (existing / "manifest.json").write_text("cũ")

    with pytest.raises(FileExistsError):
        JsonRunLogger(tmp_path).start(manifest())

    assert (existing / "manifest.json").read_text() == "cũ"


def test_a_new_attempt_number_goes_next_to_the_old_one(tmp_path: Path) -> None:
    (tmp_path / "attempt-1").mkdir()

    JsonRunLogger(tmp_path).start(manifest(attempt=2))

    assert (tmp_path / "attempt-2" / "manifest.json").exists()


@pytest.mark.parametrize(
    "calls",
    [
        ["log_round"],
        ["log_timing"],
        ["finish"],
        ["start", "start"],
        ["start", "finish", "finish"],
        ["start", "finish", "log_round"],
    ],
    ids=[
        "round-before-start",
        "timing-before-start",
        "finish-before-start",
        "start-twice",
        "finish-twice",
        "round-after-finish",
    ],
)
def test_calls_out_of_order_raise_runtime_error(tmp_path: Path, calls: list[str]) -> None:
    logger = JsonRunLogger(tmp_path)
    arguments = {"start": manifest(), "log_round": round_record(1), "log_timing": timing(1), "finish": OK}
    *setup, last = calls
    for name in setup:
        getattr(logger, name)(arguments[name])

    with pytest.raises(RuntimeError):
        getattr(logger, last)(arguments[last])


def test_a_non_finite_value_is_refused_and_nothing_is_written(tmp_path: Path) -> None:
    logger = JsonRunLogger(tmp_path)
    logger.start(manifest())
    logger.log_round(round_record(1))

    with pytest.raises(ValueError):
        logger.log_round(round_record(2, acc_mean=math.nan))

    assert read_lines(tmp_path / "attempt-1" / "rounds.jsonl") == [round_record(1)]


def test_a_refused_summary_leaves_the_attempt_open_for_a_failed_summary(tmp_path: Path) -> None:
    # Lượt bắt lỗi khi ghi summary rồi ghi SummaryFailed; summary hỏng không được nằm lại trên đĩa.
    logger = JsonRunLogger(tmp_path)
    logger.start(manifest())
    failed: SummaryFailed = {
        "status": "failed",
        "error": "ValueError: NaN",
        "messages_total": 0,
        "bytes_total": 0,
    }

    with pytest.raises(ValueError):
        logger.finish({**OK, "final_acc_avg_model": math.inf})
    assert not (tmp_path / "attempt-1" / "summary.json").exists()
    logger.finish(failed)

    assert json.loads((tmp_path / "attempt-1" / "summary.json").read_text(encoding="utf-8")) == failed


@pytest.mark.parametrize(
    ("method", "record"),
    [
        ("log_round", {k: v for k, v in round_record(1).items() if k != "train_loss_mean"}),
        ("log_round", {**round_record(1), "extra": 1}),
        ("log_timing", {"round": 1}),
        ("finish", {"status": "ok", "messages_total": 0, "bytes_total": 0}),
        ("finish", {**DIVERGED, "status": "exploded"}),
    ],
    ids=["round-missing-key", "round-extra-key", "timing-missing", "ok-missing-finals", "unknown-status"],
)
def test_records_must_match_the_schema_in_contracts(
    tmp_path: Path, method: str, record: dict[str, object]
) -> None:
    logger = JsonRunLogger(tmp_path)
    logger.start(manifest())

    with pytest.raises(ValueError, match="khóa|status"):
        getattr(logger, method)(record)


def test_manifest_with_missing_fields_is_refused_before_creating_the_attempt(tmp_path: Path) -> None:
    incomplete = {k: v for k, v in manifest().items() if k != "machine"}

    with pytest.raises(ValueError, match="machine"):
        JsonRunLogger(tmp_path).start(incomplete)

    assert not (tmp_path / "attempt-1").exists()


@pytest.mark.parametrize("attempt", [0, True, "1"], ids=["zero", "bool", "text"])
def test_attempt_number_must_be_a_positive_integer(tmp_path: Path, attempt: object) -> None:
    with pytest.raises(ValueError, match="attempt"):
        JsonRunLogger(tmp_path).start({**manifest(), "attempt": attempt})

    assert list(tmp_path.iterdir()) == []


def test_manifest_with_a_non_finite_value_is_refused_before_creating_the_attempt(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        JsonRunLogger(tmp_path).start({**manifest(), "spectral_gap": math.nan})

    assert not (tmp_path / "attempt-1").exists()


def test_json_run_logger_satisfies_the_run_logger_protocol(tmp_path: Path) -> None:
    assert isinstance(JsonRunLogger(tmp_path), RunLogger)

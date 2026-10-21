"""Test cho analysis/comm_cost.py: chi phí giao tiếp theo loại (message, byte, thời gian) và theo node, đọc từ
rounds.jsonl và timing.jsonl (T29c; ADR-013; SPEC REQ-05)."""

import json
from pathlib import Path

import numpy as np
import pytest

from analysis import aggregate, comm_cost
from analysis.aggregate import AnalysisError
from tests.analysis_fakes import config, curve, round_record, write_run


def one(root: Path, run_id: str = "r", **kwargs: object) -> aggregate.Run:
    write_run(root, run_id, **kwargs)
    run = aggregate.read_run(root / "runs" / run_id).finished
    assert run is not None
    return run


def test_read_run_reads_timing_and_an_old_run_without_timing_still_reads(tmp_path: Path) -> None:
    run = one(tmp_path, "new")
    path = write_run(tmp_path, "old")
    (path / "timing.jsonl").unlink()

    old = aggregate.read_run(tmp_path / "runs" / "old").finished

    assert [t["round"] for t in run.timing] == [5, 10, 15, 20]
    assert old is not None and old.timing == []


def test_cost_series_follow_the_cumulative_messages_and_bytes(tmp_path: Path) -> None:
    run = one(tmp_path)

    series = comm_cost.cost_series(run)

    assert list(series.steps) == [100, 200, 300, 400]
    assert list(series.messages) == [40, 80, 120, 160]  # 8 message mỗi round, τ = 20
    assert list(series.bytes) == [40_000, 80_000, 120_000, 160_000]


def test_phase_seconds_are_per_round_and_other_is_wall_time_outside_every_phase(tmp_path: Path) -> None:
    # 4 lần ghi, 20 round: train 4 × 0.4 s, combine 4 × 0.1 s, eval 4 × 0.2 s, wall 3.2 s.
    run = one(tmp_path)

    phases = comm_cost.phase_seconds_per_round(run)

    assert phases == pytest.approx({"train": 0.08, "comm": 0.0, "combine": 0.02, "eval": 0.04, "other": 0.02})


def test_other_time_is_never_negative_when_phases_overlap(tmp_path: Path) -> None:
    # Ở process runtime, eval của observer chạy song song với train của node, nên tổng các pha vượt được wall.
    rounds = curve([0.5])
    timing = [
        {
            "round": 5,
            "wall_time_s": 1.0,
            "train_s": 0.9,
            "combine_s": 0.1,
            "eval_s": 0.5,
            "comm_s": 0.2,
            "train_s_per_node": [0.9] * 4,
            "comm_s_per_node": [0.2] * 4,
        }
    ]
    run = one(tmp_path, rounds=rounds, timing=timing)

    assert comm_cost.phase_seconds_per_round(run)["other"] == 0.0


def test_a_run_without_timing_has_no_phase_seconds(tmp_path: Path) -> None:
    path = write_run(tmp_path, "old")
    (path / "timing.jsonl").unlink()
    run = aggregate.read_run(tmp_path / "runs" / "old").finished

    with pytest.raises(AnalysisError, match="timing.jsonl"):
        comm_cost.phase_seconds_per_round(run)


def test_node_load_is_per_round_at_the_last_record(tmp_path: Path) -> None:
    run = one(tmp_path)

    load = comm_cost.node_load(run)

    # Mốc cuối: round 20, mỗi node 80 message và 80,000 byte.
    assert list(load.messages) == [4.0] * 4 and list(load.bytes) == [4000.0] * 4
    assert load.server_messages is None and load.server_bytes is None


def test_node_load_puts_the_virtual_server_aside(tmp_path: Path) -> None:
    # Đường server: phần tử thứ N là server ảo, nhận N và gửi N message mỗi round.
    rounds = [
        round_record(
            100, 20, [0.5] * 4, node_messages_cum=[10] * 4 + [40], node_bytes_cum=[10_000] * 4 + [40_000]
        )
    ]
    run = one(tmp_path, cfg=config(topology__kind="server"), rounds=rounds)

    load = comm_cost.node_load(run)

    assert list(load.messages) == [2.0] * 4 and load.server_messages == 8.0
    assert list(load.bytes) == [2000.0] * 4 and load.server_bytes == 8000.0


def test_node_load_refuses_a_run_written_before_adr_013(tmp_path: Path) -> None:
    rounds = curve([0.5, 0.8])
    for r in rounds:
        del r["node_messages_cum"], r["node_bytes_cum"]
    run = one(tmp_path, rounds=rounds)

    with pytest.raises(AnalysisError, match="ADR-013"):
        comm_cost.node_load(run)


def test_node_load_refuses_a_vector_of_the_wrong_length(tmp_path: Path) -> None:
    rounds = [round_record(100, 20, [0.5] * 4, node_messages_cum=[1, 2], node_bytes_cum=[1, 2])]
    run = one(tmp_path, rounds=rounds)

    with pytest.raises(AnalysisError, match="N = 4"):
        comm_cost.node_load(run)


def test_node_seconds_sum_every_record_per_node_and_divide_by_rounds(tmp_path: Path) -> None:
    run = one(tmp_path)

    train, comm = comm_cost.node_seconds_per_round(run)

    assert train == pytest.approx(np.full(4, 4 * 0.1 / 20))
    assert comm == pytest.approx(np.zeros(4))


def test_node_seconds_refuse_timing_written_before_adr_013(tmp_path: Path) -> None:
    path = write_run(tmp_path, "old")
    lines = [json.loads(line) for line in (path / "timing.jsonl").read_text().splitlines()]
    for t in lines:
        del t["train_s_per_node"], t["comm_s_per_node"]
    (path / "timing.jsonl").write_text("".join(json.dumps(t) + "\n" for t in lines))
    run = aggregate.read_run(tmp_path / "runs" / "old").finished

    with pytest.raises(AnalysisError, match="ADR-013"):
        comm_cost.node_seconds_per_round(run)


def test_one_figure_never_mixes_the_two_runtimes(tmp_path: Path) -> None:
    runs = [one(tmp_path, "a"), one(tmp_path, "b", cfg=config(runtime="processes"))]

    with pytest.raises(AnalysisError, match="runtime"):
        comm_cost.single_runtime(runs)
    assert comm_cost.single_runtime(runs[:1]) == "inprocess"

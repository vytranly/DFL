"""Process runtime: mỗi node một tiến trình qua gloo, tiến trình chạy lượt kiêm observer (T22b; ADR-010;
SPEC §9.2 T-INV-13; T-INV-6 ở N = 4).

Mỗi lượt khởi chạy 4 tiến trình con `python -m dfl node` thật; cấu hình là profile smoke rút gọn (2 round,
tập con đánh giá 100 ảnh) để test nhanh. Provenance được truyền vào để test không phụ thuộc trạng thái git.
"""

import dataclasses
import json
from pathlib import Path

import pytest

from dfl.config import AlgorithmConfig, EvalConfig, RunConfig, TopologyConfig, load_config
from dfl.io import manifest
from dfl.sim import process_runtime, run

ROOT = Path(__file__).resolve().parents[2]
SHORT = dataclasses.replace(
    load_config(ROOT / "configs" / "profiles" / "smoke.yaml"),
    local_steps=4,
    eval=EvalConfig(subset_size=100, every_steps=None),
)


@pytest.fixture(scope="module")
def here() -> manifest.Provenance:
    return dataclasses.replace(manifest.current_provenance(), dirty=False)


def both(config: RunConfig, out: Path, provenance: manifest.Provenance) -> tuple[Path, Path]:
    inprocess = run.run(config, out=out / "inprocess", provenance=provenance)
    processes = run.run(
        dataclasses.replace(config, runtime="processes"), out=out / "processes", provenance=provenance
    )
    return inprocess, processes


def read(attempt: Path, name: str) -> object:
    return json.loads((attempt / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("kind", ["ring", "gossip", "fully_connected"])
def test_t_inv_13_processes_reproduce_the_simulator_byte_for_byte(
    tmp_path: Path, here: manifest.Provenance, kind: str
) -> None:
    config = dataclasses.replace(SHORT, topology=TopologyConfig(kind=kind))

    inprocess, processes = both(config, tmp_path, here)

    assert (processes / "rounds.jsonl").read_bytes() == (inprocess / "rounds.jsonl").read_bytes()
    assert read(processes, "summary.json") == read(inprocess, "summary.json")
    manifest_in, manifest_proc = read(inprocess, "manifest.json"), read(processes, "manifest.json")
    for key in ("node_class_counts", "objective_weights", "spectral_gap", "partition_retries"):
        assert manifest_proc[key] == manifest_in[key]


@pytest.mark.parametrize("kind", ["ring", "gossip"])
def test_t_inv_13_processes_match_the_simulator_on_a_sparse_evaluation_grid(
    tmp_path: Path, here: manifest.Provenance, kind: str
) -> None:
    # Review C3 P3.7, P1-7: τ = 2, 10 bước, đánh giá mỗi 4 bước, nên các mốc là round 2, 4 và round cuối 5,
    # thưa như E8. Node và observer phải chọn cùng các round; lệch thì observer chờ mãi, nên đặt timeout ngắn.
    config = dataclasses.replace(
        SHORT,
        local_steps=10,
        topology=TopologyConfig(kind=kind),
        eval=EvalConfig(subset_size=100, every_steps=4),
    )

    inprocess = run.run(config, out=tmp_path / "inprocess", provenance=here)
    processes = run.run(
        dataclasses.replace(config, runtime="processes"),
        out=tmp_path / "processes",
        provenance=here,
        process_timeout=120.0,
    )

    rounds = (inprocess / "rounds.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["round"] for line in rounds] == [2, 4, 5]
    assert (processes / "rounds.jsonl").read_bytes() == (inprocess / "rounds.jsonl").read_bytes()


def test_t_inv_6_messages_and_bytes_count_only_node_traffic(
    tmp_path: Path, here: manifest.Provenance
) -> None:
    # Ring N = 4: 8 message mỗi round, 4 × 46,730 byte mỗi message; kênh tới observer không được đếm.
    config = dataclasses.replace(SHORT, runtime="processes")

    attempt = run.run(config, out=tmp_path, provenance=here)

    rounds = [json.loads(line) for line in (attempt / "rounds.jsonl").read_text().splitlines()]
    assert [r["messages_cum"] for r in rounds] == [8, 16]
    assert all(r["bytes_cum"] == r["messages_cum"] * 4 * 46_730 for r in rounds)
    timing = [json.loads(line) for line in (attempt / "timing.jsonl").read_text().splitlines()]
    assert all(t["comm_s"] >= 0 for t in timing) and len(timing) == 2


def test_a_diverging_node_stops_every_process_at_the_simulator_round(
    tmp_path: Path, here: manifest.Provenance
) -> None:
    exploding = dataclasses.replace(SHORT, algorithm=AlgorithmConfig(tau=2, lr=1.0e4), local_steps=12)

    inprocess, processes = both(exploding, tmp_path, here)

    expected, actual = read(inprocess, "summary.json"), read(processes, "summary.json")
    assert actual["status"] == expected["status"] == "diverged"
    assert actual["diverged_at_round"] == expected["diverged_at_round"]
    assert (actual["messages_total"], actual["bytes_total"]) == (
        expected["messages_total"],
        expected["bytes_total"],
    )
    assert (processes / "rounds.jsonl").read_bytes() == (inprocess / "rounds.jsonl").read_bytes()


def test_a_node_that_fails_to_start_fails_the_run_quickly(tmp_path: Path, here: manifest.Provenance) -> None:
    # Tiến trình con thấy code_version khác và thoát trước khi vào process group; lượt phải dừng ngay (không
    # đợi hết thời gian chờ của gloo) và ghi SummaryFailed.
    config = dataclasses.replace(SHORT, runtime="processes")
    stale = dataclasses.replace(here, code_version="0" * 64)

    with pytest.raises(process_runtime.NodeFailed, match="node"):
        run.run(config, out=tmp_path, provenance=stale, process_timeout=60)

    (attempt,) = sorted(tmp_path.glob("runs/*/attempt-*"))
    summary = read(attempt, "summary.json")
    assert summary["status"] == "failed" and "NodeFailed" in summary["error"]


def test_t_inv_13_processes_refuse_a_gossip_schedule_whose_union_never_connects(
    tmp_path: Path, here: manifest.Provenance
) -> None:
    # Seed 1: round 0 và round 1 dùng cùng một matching trên K₄, nên hợp hai round không liên thông (INV-4).
    # Cả hai runtime phải từ chối trước round đầu với cùng lý do (REVIEW-C3-P3.7, P0-1).
    config = dataclasses.replace(SHORT, topology=TopologyConfig(kind="gossip"), seed=1)
    summaries = []

    for runtime in ("inprocess", "processes"):
        out = tmp_path / runtime
        with pytest.raises(ValueError, match="hợp các round"):
            run.run(
                dataclasses.replace(config, runtime=runtime), out=out, provenance=here, process_timeout=60
            )
        (attempt,) = sorted(out.glob("runs/*/attempt-*"))
        summaries.append(read(attempt, "summary.json"))

    assert summaries[0] == summaries[1]
    assert summaries[0]["status"] == "failed" and summaries[0]["messages_total"] == 0


def linger(self: process_runtime.ProcessRun, children: object, scratch: object) -> None:
    raise process_runtime.NodeFailed("node không thoát sau lần gom cuối (giả lập)")


def test_a_node_lingering_after_the_last_round_does_not_fail_a_finished_run(
    tmp_path: Path,
    here: manifest.Provenance,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Mọi round đã được quan sát và đánh giá cuối đã xong, nên summary vẫn là ok, chỉ thêm cảnh báo
    # (REVIEW-C3-P4.1 S1).
    monkeypatch.setattr(process_runtime.ProcessRun, "_wait_exit", linger)

    attempt = run.run(dataclasses.replace(SHORT, runtime="processes"), out=tmp_path, provenance=here)

    assert read(attempt, "summary.json")["status"] == "ok"
    assert "không thoát" in capsys.readouterr().err


def test_a_node_lingering_after_divergence_keeps_the_run_diverged(
    tmp_path: Path, here: manifest.Provenance, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(process_runtime.ProcessRun, "_wait_exit", linger)
    exploding = dataclasses.replace(
        SHORT, algorithm=AlgorithmConfig(tau=2, lr=1.0e4), local_steps=12, runtime="processes"
    )

    attempt = run.run(exploding, out=tmp_path, provenance=here)

    assert read(attempt, "summary.json")["status"] == "diverged"

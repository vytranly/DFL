"""Smoke e2e: `python -m dfl run --config configs/profiles/smoke.yaml --out …` trong một tiến trình mới
(T11b; SPEC §5.2, §5.3; ENG-4; T-INV-6 ở N = 4).
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from dfl.config import canonical_json, load_config
from dfl.contracts import ManifestRecord, RoundRecord, SummaryOk, TimingRecord

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "configs" / "profiles" / "smoke.yaml"
BUDGET_S = 60
# SmallCNN float32: 4 × 46,730 byte mỗi message; ring N = 4 có 4 cạnh, tức 8 message mỗi round.
MESSAGE_BYTES = 4 * 46_730
RING_MESSAGES_PER_ROUND = 8


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, float]:
    out = tmp_path_factory.mktemp("smoke")
    start = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-m", "dfl", "run", "--config", str(SMOKE), "--out", str(out)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=3 * BUDGET_S,
        check=False,
    )
    elapsed = time.perf_counter() - start
    assert result.returncode == 0, result.stderr
    (attempt,) = (out / "runs").glob("*/attempt-1")
    return attempt, elapsed


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_smoke_run_finishes_within_the_eng_4_budget(smoke_run: tuple[Path, float]) -> None:
    assert smoke_run[1] < BUDGET_S


def test_smoke_run_writes_four_files_that_follow_the_contract_schema(smoke_run: tuple[Path, float]) -> None:
    attempt = smoke_run[0]

    rounds = read_lines(attempt / "rounds.jsonl")
    timing = read_lines(attempt / "timing.jsonl")
    summary = read_json(attempt / "summary.json")

    assert sorted(p.name for p in attempt.iterdir()) == [
        "manifest.json",
        "rounds.jsonl",
        "summary.json",
        "timing.jsonl",
    ]
    assert set(read_json(attempt / "manifest.json")) == set(ManifestRecord.__required_keys__)
    assert [r["round"] for r in rounds] == [1, 2, 3, 4, 5, 6]
    assert [r["local_steps"] for r in rounds] == [2, 4, 6, 8, 10, 12]
    assert all(set(r) == set(RoundRecord.__required_keys__) for r in rounds)
    assert [t["round"] for t in timing] == [1, 2, 3, 4, 5, 6]
    assert all(set(t) == set(TimingRecord.__required_keys__) for t in timing)
    assert set(summary) == set(SummaryOk.__required_keys__) and summary["status"] == "ok"
    assert (summary["final_eval_split"], summary["final_eval_size"]) == ("test", 500)
    assert all(len(summary[k]) == 4 for k in summary if str(k).endswith("_per_node"))


def test_manifest_records_the_threads_the_config_and_its_own_run_id(smoke_run: tuple[Path, float]) -> None:
    attempt = smoke_run[0]
    config = load_config(SMOKE)

    manifest = read_json(attempt / "manifest.json")

    assert manifest["cpu_threads"] == config.threads == 1
    assert manifest["config"] == json.loads(canonical_json(config))
    assert manifest["run_id"] == attempt.parent.name
    assert manifest["attempt"] == 1
    assert manifest["spectral_gap"] == pytest.approx(2 / 3)  # ring N = 4 với MH: 1 − |λ₂| = 1 − 1/3
    assert manifest["objective_weights"] == [0.25] * 4
    assert [sum(row) for row in manifest["node_class_counts"]] == [450] * 4  # 1,800 ảnh chia đều


def test_messages_and_bytes_follow_the_ring_with_one_message_per_neighbour(
    smoke_run: tuple[Path, float],
) -> None:
    attempt = smoke_run[0]

    rounds = read_lines(attempt / "rounds.jsonl")
    summary = read_json(attempt / "summary.json")

    assert [r["messages_cum"] for r in rounds] == [RING_MESSAGES_PER_ROUND * k for k in range(1, 7)]
    assert all(r["bytes_cum"] == r["messages_cum"] * MESSAGE_BYTES for r in rounds)
    assert (summary["messages_total"], summary["bytes_total"]) == (48, 48 * MESSAGE_BYTES)


def test_timing_phases_add_up_to_the_wall_time_between_records(smoke_run: tuple[Path, float]) -> None:
    timing = read_lines(smoke_run[0] / "timing.jsonl")

    previous = 0.0
    for record in timing:
        phases = record["train_s"] + record["combine_s"] + record["eval_s"] + record["comm_s"]
        assert min(record["train_s"], record["combine_s"], record["eval_s"], record["comm_s"]) >= 0
        assert record["comm_s"] == 0  # Transport trong bộ nhớ không chờ (ADR-010)
        assert phases == pytest.approx(record["wall_time_s"] - previous, abs=1e-6)
        previous = record["wall_time_s"]

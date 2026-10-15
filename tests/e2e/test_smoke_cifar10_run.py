"""Smoke e2e của đường CIFAR-10: `python -m dfl run --config configs/profiles/smoke_cifar10.yaml --out …`
trong một tiến trình mới (T28d; ADR-012; ENG-4).
"""

import json
import subprocess
import sys
from pathlib import Path

from dfl.config import canonical_json, load_config

ROOT = Path(__file__).resolve().parents[2]
SMOKE_CIFAR10 = ROOT / "configs" / "profiles" / "smoke_cifar10.yaml"
# CNN của NIID-Bench float32: 4 × 62,006 byte mỗi message; ring N = 4, 6 round, 8 message mỗi round.
MESSAGE_BYTES = 4 * 62_006


def test_smoke_cifar10_run_finishes_on_the_command_line_and_counts_niidbench_bytes(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "dfl", "run", "--config", str(SMOKE_CIFAR10), "--out", str(tmp_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    (attempt,) = (tmp_path / "runs").glob("*/attempt-1")
    summary = json.loads((attempt / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((attempt / "manifest.json").read_text(encoding="utf-8"))
    assert summary["status"] == "ok"
    assert (summary["messages_total"], summary["bytes_total"]) == (48, 48 * MESSAGE_BYTES)
    assert manifest["config"] == json.loads(canonical_json(load_config(SMOKE_CIFAR10)))

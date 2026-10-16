"""e2e: `python -m analysis.plots --draft` vẽ được từ kết quả smoke thật; không có --draft thì từ chối vì
--results nằm ngoài results/ (T16a; SPEC INT-1, REQ-10). Kết quả không phụ thuộc trạng thái git của cây.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_plots_draw_from_smoke_results_in_draft_mode_only(tmp_path: Path) -> None:
    def call(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", *args], cwd=ROOT, capture_output=True, text=True, timeout=300
        )

    smoke = call("dfl", "run", "--config", "configs/profiles/smoke.yaml", "--out", str(tmp_path / "smoke"))
    assert smoke.returncode == 0, smoke.stderr

    draft = call(
        "analysis.plots", "--results", str(tmp_path / "smoke"), "--out", str(tmp_path / "fig"), "--draft"
    )
    refused = call("analysis.plots", "--results", str(tmp_path / "smoke"), "--out", str(tmp_path / "fig2"))

    assert draft.returncode == 0, draft.stderr
    assert (tmp_path / "fig" / "P1.png").stat().st_size > 0 and (
        tmp_path / "fig" / "P7.png"
    ).stat().st_size > 0
    record = json.loads((tmp_path / "fig" / "PROVENANCE.json").read_text(encoding="utf-8"))
    assert record["draft"] is True and len(record["analysis_version"]) == 64
    assert refused.returncode != 0 and "results/" in refused.stderr

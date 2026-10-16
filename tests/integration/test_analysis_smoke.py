"""analysis/ trên kết quả thật của một lượt smoke: schema mà analysis tự đọc khớp schema mà dfl ghi (T15a)."""

import dataclasses
import json
from pathlib import Path

from analysis import aggregate, checks
from dfl.config import EvalConfig, load_config
from dfl.sim import run

ROOT = Path(__file__).resolve().parents[2]


def test_analysis_reads_a_real_smoke_run_like_the_files_say(tmp_path: Path) -> None:
    config = dataclasses.replace(
        load_config(ROOT / "configs" / "profiles" / "smoke.yaml"),
        local_steps=4,
        eval=EvalConfig(subset_size=100, every_steps=None),
    )
    attempt = run.run(config, out=tmp_path)

    (entry,) = aggregate.scan(tmp_path)
    table = aggregate.frame([entry])

    summary = json.loads((attempt / "summary.json").read_text(encoding="utf-8"))
    assert entry.finished is not None and entry.finished.path == attempt
    assert [r["round"] for r in entry.finished.rounds] == [1, 2]
    assert table.loc[0, "status"] == "ok" and table.loc[0, "dataset"] == "synthetic"
    assert table.loc[0, "final_acc_avg_model"] == summary["final_acc_avg_model"]
    assert table.loc[0, "bytes_total"] == summary["bytes_total"]
    checks.check_clean([entry.finished], draft=True)
    checks.check_no_nan(table)

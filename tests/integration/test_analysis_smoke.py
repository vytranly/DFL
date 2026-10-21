"""analysis/ trên kết quả thật của một lượt smoke: schema mà analysis tự đọc khớp schema mà dfl ghi (T15a)."""

import dataclasses
import json
from pathlib import Path

from analysis import aggregate, centrality, checks
from dfl.config import EvalConfig, PartitionConfig, TopologyConfig, load_config
from dfl.data import partition
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


def test_centrality_reads_a_real_e7_run_and_agrees_with_the_partitioner(tmp_path: Path) -> None:
    # analysis giữ bản sao RARE_CLASSES vì không được import dfl.data (ADR-014).
    config = dataclasses.replace(
        load_config(ROOT / "configs" / "profiles" / "smoke.yaml"),
        num_nodes=8,
        local_steps=4,
        topology=TopologyConfig(kind="barabasi_albert", m=2),
        partition=PartitionConfig(scheme="rare_classes", holders=2, placement="hub"),
        eval=EvalConfig(subset_size=100, every_steps=None),
    )
    run.run(config, out=tmp_path)

    (entry,) = aggregate.scan(tmp_path)
    assert entry.finished is not None
    degree, held = centrality.degrees(entry.finished), centrality.holders(entry.finished)

    assert centrality.RARE_CLASSES == partition.RARE_CLASSES
    assert degree.sum() == 2 * 2 * (8 - 2) and held.sum() == 2
    assert degree[held].min() >= degree[~held].max()
    assert 0 <= centrality.non_holder_accuracy(entry.finished) <= 1
    # ADR-015: lượt ghi trước ADR-014 không có graph_edges; bậc suy từ tải message phải trùng bậc thật.
    old = dataclasses.replace(entry.finished, manifest={**entry.finished.manifest, "graph_edges": None})
    assert centrality.degrees(old).tolist() == degree.tolist()

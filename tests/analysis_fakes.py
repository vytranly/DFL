"""Thư mục kết quả giả theo đúng schema của src/dfl/contracts.py, cho test của analysis/ (T15a–T16a).

Không chạy mô phỏng: mỗi lượt là manifest.json, rounds.jsonl, summary.json (và timing.jsonl) viết tay, để test
kiểm từng luật của analysis mà không phụ thuộc kết quả thật. Một test integration riêng chạy lại analysis trên
lượt smoke thật.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

CODE_VERSION = "c" * 64
MACHINE = "Test CPU (flags 0123456789ab)"
BASE_CONFIG: dict[str, object] = {
    "algorithm": {"batch_size": 32, "lr": 0.05, "mu": 0.0, "tau": 20},
    "augmentation": "none",
    "dataset": "mnist",
    "deploy": "local",
    "eval": {"every_steps": 100, "split": "test", "subset_size": 2000},
    "local_steps": 400,
    "mixing": "metropolis_hastings",
    "model": "smallcnn",
    "num_nodes": 4,
    "partition": {"alpha": None, "beta": None, "k": None, "min_size": 64, "scheme": "iid"},
    "runtime": "inprocess",
    "seed": 0,
    "threads": 1,
    "topology": {"kind": "ring", "p": None},
}


def config(**overrides: object) -> dict[str, object]:
    """Cấu hình chuẩn hóa của BASE_CONFIG, ghi đè theo khóa phẳng `section__field` hay khóa gốc."""
    result = copy.deepcopy(BASE_CONFIG)
    for key, value in overrides.items():
        if "__" in key:
            section, field = key.split("__")
            result[section][field] = value
        else:
            result[key] = value
    return result


def round_record(step: int, tau: int, acc: list[float], **fields: object) -> dict[str, object]:
    """Một dòng rounds.jsonl ở mốc `step` bước cục bộ; các trường tổng hợp tính từ `acc` theo node."""
    n = len(acc)
    mean = sum(acc) / n
    std = (sum((a - mean) ** 2 for a in acc) / n) ** 0.5
    record: dict[str, object] = {
        "round": step // tau,
        "local_steps": step,
        "acc_mean": mean,
        "acc_std": std,
        "acc_min": min(acc),
        "acc_max": max(acc),
        "acc_local_mean": mean,
        "acc_local_std": std,
        "acc_local_min": min(acc),
        "f1_macro_mean": mean,
        "f1_macro_std": std,
        "f1_macro_min": min(acc),
        "acc_avg_model": mean,
        "acc_eff_mean": mean,
        "drift_pre_combine": 0.5,
        "consensus_distance": 0.1,
        "messages_cum": 8 * (step // tau),
        "bytes_cum": 8 * (step // tau) * 1000,
        "max_node_bytes_cum": 4 * (step // tau) * 1000,
        "train_loss_mean": 1.0,
        "acc_per_node": list(acc),
        "acc_local_per_node": list(acc),
    }
    record.update(fields)
    return record


def curve(values: list[float], *, tau: int = 20, nodes: int = 4, every: int = 100) -> list[dict[str, object]]:
    """rounds.jsonl với acc_mean = values[i] ở mốc (i + 1) × every; mọi node cùng accuracy."""
    return [round_record((i + 1) * every, tau, [v] * nodes) for i, v in enumerate(values)]


def write_run(
    root: Path,
    run_id: str,
    *,
    cfg: dict[str, object] | None = None,
    rounds: list[dict[str, object]] | None = None,
    status: str = "ok",
    attempt: int = 1,
    code_version: str = CODE_VERSION,
    dirty: bool = False,
    env_match: bool = True,
    machine: str = MACHINE,
    final_eval_split: str | None = None,
    final_acc: float = 0.9,
) -> Path:
    """Một attempt trong `root`/runs/<run_id>/attempt-<k>/; status ok, diverged, failed hay None (bị ngắt)."""
    cfg = cfg if cfg is not None else config()
    n = int(cfg["num_nodes"])
    rounds = rounds if rounds is not None else curve([0.5, 0.7, 0.8, 0.85], nodes=n)
    path = root / "runs" / run_id / f"attempt-{attempt}"
    path.mkdir(parents=True)
    manifest = {
        "run_id": run_id,
        "code_version": code_version,
        "git_commit": "a" * 40,
        "dirty": dirty,
        "env_match": env_match,
        "attempt": attempt,
        "config": cfg,
        "seed": cfg["seed"],
        "child_seeds": {"partition": 1},
        "partition_retries": 0,
        "versions": {"python": "3.12.3"},
        "machine": machine,
        "cpu_threads": cfg["threads"],
        "created_at": "2026-10-16T00:00:00+00:00",
        "spectral_gap": 0.5,
        "objective_weights": [1 / n] * n,
        "objective_shift_tv": 0.0,
        "node_class_counts": [[10] * 10 for _ in range(n)],
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (path / "rounds.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rounds), encoding="utf-8")
    last = rounds[-1] if rounds else {"messages_cum": 0, "bytes_cum": 0}
    split = final_eval_split or cfg["eval"]["split"]
    summaries: dict[str, dict[str, object]] = {
        "ok": {
            "status": "ok",
            "messages_total": last["messages_cum"],
            "bytes_total": last["bytes_cum"],
            "final_eval_split": split,
            "final_eval_size": 10_000 if split == "test" else 5_000,
            "final_acc_per_node": [final_acc] * n,
            "final_acc_local_per_node": [final_acc] * n,
            "final_f1_macro_per_node": [final_acc] * n,
            "final_acc_avg_model": final_acc,
        },
        "diverged": {
            "status": "diverged",
            "diverged_at_round": len(rounds) + 1,
            "reason": "mất mát huấn luyện của node 0 không hữu hạn (nan)",
            "messages_total": last["messages_cum"],
            "bytes_total": last["bytes_cum"],
        },
        "failed": {"status": "failed", "error": "MemoryError: giả", "messages_total": 0, "bytes_total": 0},
    }
    if status is not None:
        (path / "summary.json").write_text(json.dumps(summaries[status]), encoding="utf-8")
    return path


def write_index(root: Path, experiment: str, versions: dict[str, list[str]]) -> Path:
    """results/experiments/<experiment>.json theo contracts.ExperimentIndex."""
    path = root / "experiments" / f"{experiment}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = {
        cv: {"run_ids": sorted(ids), "git_commit": "a" * 40, "machine": MACHINE, "updated_at": "2026-10-16"}
        for cv, ids in versions.items()
    }
    path.write_text(json.dumps({"experiment": experiment, "versions": entries}), encoding="utf-8")
    return path

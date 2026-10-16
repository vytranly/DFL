"""Đọc kết quả của các lượt và gom thành bảng, mỗi lượt một dòng (module analysis; T15a; SPEC §5.3, INT-2,
INT-3; ADR-008).

- Chỉ import `dfl.contracts` (contract `analysis-reads-only`) và tự đọc JSON/JSONL, nên sửa code đọc ở đây
  không làm đổi `code_version`.
- Lượt được chọn qua chỉ mục `results/experiments/<exp>.json`, theo đúng một `code_version`; không quét cả
  thư mục. `scan` đọc mọi `runs/*` của một thư mục chỉ dành cho `--draft` (smoke, demo), vì `run --out` không
  ghi chỉ mục.
- Mỗi run_id có nhiều nhất một attempt đã xong (`ok` hay `diverged`); attempt `failed` và attempt bị ngắt
  (không có summary.json) được đếm, không bị bỏ qua (INT-3). Lượt thiếu hay phân kỳ vẫn có dòng trong bảng.
- File kết quả không bao giờ chứa NaN hay vô cực (ghi với `allow_nan=False`), nên gặp chúng lúc đọc là file
  hỏng: báo lỗi ngay, không để NaN lọt vào thống kê (INT-3).

json.loads (parse_constant nhận '-Infinity', 'Infinity', 'NaN'): https://docs.python.org/3.12/library/json.html#json.loads
pandas.DataFrame, DataFrame.to_csv: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from dfl.contracts import ExperimentIndex, ManifestRecord, RoundRecord

FINISHED = ("ok", "diverged")


class AnalysisError(ValueError):
    """Kết quả không dùng được cho phân tích: vi phạm INT-2, INT-3, INT-6 hay file hỏng."""


def _reject_constant(name: str) -> float:
    raise AnalysisError(
        f"giá trị {name} trong file kết quả: kết quả không bao giờ chứa NaN hay vô cực (INT-3)"
    )


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except AnalysisError as err:
        raise AnalysisError(f"{path}: {err}") from None


def read_lines(path: Path) -> list[Any]:
    try:
        return [
            json.loads(line, parse_constant=_reject_constant)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    except AnalysisError as err:
        raise AnalysisError(f"{path}: {err}") from None


@dataclass(frozen=True)
class Run:
    """Attempt đã xong (`ok` hay `diverged`) của một run_id."""

    run_id: str
    path: Path
    manifest: ManifestRecord
    rounds: list[RoundRecord]
    summary: dict[str, Any]

    @property
    def status(self) -> str:
        return str(self.summary["status"])

    @property
    def config(self) -> dict[str, Any]:
        return self.manifest["config"]

    @property
    def code_version(self) -> str:
        return self.manifest["code_version"]

    @property
    def seed(self) -> int:
        return int(self.config["seed"])


@dataclass(frozen=True)
class Attempts:
    """Mọi attempt của một run_id: attempt đã xong (nếu có), attempt `failed` và attempt bị ngắt."""

    run_id: str
    finished: Run | None
    failed: list[tuple[int, str]]
    interrupted: list[int]


def _attempt_number(path: Path) -> int:
    return int(path.name.removeprefix("attempt-"))


def read_run(run_dir: Path) -> Attempts:
    """Các attempt trong `run_dir` (results/runs/<run_id>); thư mục không có thì run_id chưa chạy."""
    run_id = run_dir.name
    finished: list[Run] = []
    failed: list[tuple[int, str]] = []
    interrupted: list[int] = []
    attempts = sorted(run_dir.glob("attempt-*"), key=_attempt_number) if run_dir.is_dir() else []
    for path in attempts:
        number = _attempt_number(path)
        if not (path / "summary.json").exists():
            interrupted.append(number)
            continue
        summary = read_json(path / "summary.json")
        if summary["status"] == "failed":
            failed.append((number, str(summary["error"])))
            continue
        finished.append(
            Run(run_id, path, read_json(path / "manifest.json"), read_lines(path / "rounds.jsonl"), summary)
        )
    if len(finished) > 1:
        raise AnalysisError(
            f"run_id {run_id} có {len(finished)} attempt đã xong ({[r.path.name for r in finished]}); "
            "mỗi run_id chỉ được có một (SPEC §5.3)"
        )
    return Attempts(run_id, finished[0] if finished else None, failed, interrupted)


def read_experiment(results: Path, experiment: str, code_version: str | None = None) -> list[Attempts]:
    """Các run_id của nhóm `experiment` dưới đúng một `code_version`, theo chỉ mục do sweep ghi."""
    path = Path(results) / "experiments" / f"{experiment}.json"
    if not path.exists():
        raise AnalysisError(f"không có chỉ mục của nhóm {experiment} ({path}); nhóm này chưa được sweep")
    index: ExperimentIndex = read_json(path)
    versions = index["versions"]
    if code_version is None:
        if len(versions) != 1:
            raise AnalysisError(
                f"chỉ mục {experiment} có {len(versions)} code_version ({sorted(versions)}); chọn một, "
                "vì một so sánh không được trộn nhiều code_version (INT-2)"
            )
        code_version = next(iter(versions))
    if code_version not in versions:
        raise AnalysisError(f"chỉ mục {experiment} không có code_version {code_version}")
    return [read_run(Path(results) / "runs" / run_id) for run_id in versions[code_version]["run_ids"]]


def scan(results: Path) -> list[Attempts]:
    """Mọi run_id trong `results`/runs, sắp theo tên; chỉ cho `--draft` (thư mục ngoài results/ không có chỉ
    mục)."""
    return [read_run(path) for path in sorted((Path(results) / "runs").glob("*")) if path.is_dir()]


def _config_columns(config: dict[str, Any] | None) -> dict[str, Any]:
    c = config or {}
    algorithm, evaluation = c.get("algorithm", {}), c.get("eval", {})
    partition, topology = c.get("partition", {}), c.get("topology", {})
    return {
        "seed": c.get("seed"),
        "dataset": c.get("dataset"),
        "num_nodes": c.get("num_nodes"),
        "model": c.get("model"),
        "augmentation": c.get("augmentation"),
        "topology": topology.get("kind"),
        "er_p": topology.get("p"),
        "mixing": c.get("mixing"),
        "scheme": partition.get("scheme"),
        "alpha": partition.get("alpha"),
        "k": partition.get("k"),
        "beta": partition.get("beta"),
        "tau": algorithm.get("tau"),
        "lr": algorithm.get("lr"),
        "mu": algorithm.get("mu"),
        "batch_size": algorithm.get("batch_size"),
        "local_steps": c.get("local_steps"),
        "every_steps": evaluation.get("every_steps"),
        "eval_split": evaluation.get("split"),
        "subset_size": evaluation.get("subset_size"),
        "threads": c.get("threads"),
        "runtime": c.get("runtime"),
    }


def row(attempts: Attempts) -> dict[str, Any]:
    """Một dòng của bảng: định danh, cấu hình, trạng thái, độ đo cuối và số attempt failed/bị ngắt."""
    run = attempts.finished
    summary = run.summary if run is not None else {}
    manifest = run.manifest if run is not None else {}
    final_acc = summary.get("final_acc_per_node")
    final_f1 = summary.get("final_f1_macro_per_node")
    return {
        "run_id": attempts.run_id,
        "code_version": manifest.get("code_version"),
        "machine": manifest.get("machine"),
        "dirty": manifest.get("dirty"),
        "env_match": manifest.get("env_match"),
        **_config_columns(manifest.get("config")),
        "status": run.status if run is not None else "missing",
        "diverged_at_round": summary.get("diverged_at_round"),
        "final_eval_split": summary.get("final_eval_split"),
        "final_acc_mean": sum(final_acc) / len(final_acc) if final_acc else None,
        "final_acc_min": min(final_acc) if final_acc else None,
        "final_acc_avg_model": summary.get("final_acc_avg_model"),
        "final_f1_macro_mean": sum(final_f1) / len(final_f1) if final_f1 else None,
        "messages_total": summary.get("messages_total"),
        "bytes_total": summary.get("bytes_total"),
        "failed_attempts": len(attempts.failed),
        "interrupted_attempts": len(attempts.interrupted),
    }


def frame(attempts: Sequence[Attempts]) -> pd.DataFrame:
    """Bảng mỗi run_id một dòng, theo thứ tự đầu vào; lượt thiếu và lượt phân kỳ vẫn có dòng (INT-3)."""
    return pd.DataFrame([row(a) for a in attempts])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m analysis.aggregate", description="Gom kết quả thành CSV, mỗi lượt một dòng."
    )
    parser.add_argument("--results", type=Path, default=Path("results"), help="thư mục kết quả")
    parser.add_argument("--out", type=Path, required=True, help="thư mục ghi CSV, ví dụ build/analysis")
    parser.add_argument("--experiment", action="append", default=[], help="tên nhóm; mặc định mọi nhóm")
    parser.add_argument("--code-version", default=None, help="code_version khi chỉ mục có nhiều mục")
    parser.add_argument("--scan", action="store_true", help="đọc mọi runs/* không qua chỉ mục (--draft)")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.scan:
        frame(scan(args.results)).to_csv(args.out / "runs.csv", index=False)
        print(f"đã ghi {args.out / 'runs.csv'}")
        return 0
    names = args.experiment or sorted(p.stem for p in (args.results / "experiments").glob("*.json"))
    for name in names:
        target = args.out / f"{name}.csv"
        frame(read_experiment(args.results, name, args.code_version)).to_csv(target, index=False)
        print(f"đã ghi {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

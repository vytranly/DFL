"""Các phép kiểm toàn vẹn trước khi so sánh hay vẽ (module analysis; T15a; SPEC §9.3 INT-2, INT-3, INT-6;
CONSTRAINTS R4, R5, R9).

Mỗi luật một hàm, báo `AnalysisError` với câu nói rõ luật bị vi phạm và lượt nào vi phạm. Hàm vẽ hình và lệnh
sanity gọi các hàm này trên đúng tập lượt của một hình hay một so sánh.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.aggregate import AnalysisError, Attempts, Run

__all__ = ["AnalysisError"]

# Biến cố định của mọi so sánh giữa các lượt DFL (SPEC INT-2, REQ-12): chỉ biến là trục của hình mới được
# khác.
FIXED_VARIABLES = {
    "dataset": lambda run: run.config["dataset"],
    "num_nodes": lambda run: run.config["num_nodes"],
    "model": lambda run: run.config["model"],
    "lr": lambda run: run.config["algorithm"]["lr"],
    "batch_size": lambda run: run.config["algorithm"]["batch_size"],
    "local_steps": lambda run: run.config["local_steps"],
    "every_steps": lambda run: run.config["eval"]["every_steps"],
    "eval_split": lambda run: run.config["eval"]["split"],
    "subset_size": lambda run: run.config["eval"]["subset_size"],
    "threads": lambda run: run.config["threads"],
    "machine": lambda run: run.manifest["machine"],
}
# Baseline được vẽ làm đường tham chiếu có nhãn thì miễn kiểm biến cố định: B1 và B1-4k (central), B2
# (local_only). Đường server (B3, E1c, E4) là ô được so sánh nên không bao giờ được miễn.
REFERENCE_KINDS = frozenset({"central", "local_only"})


def single_code_version(runs: Iterable[Run]) -> str:
    """code_version chung của các lượt; nhiều code_version trong một so sánh thì báo lỗi (INT-2, R5)."""
    versions = sorted({run.code_version for run in runs})
    if len(versions) != 1:
        raise AnalysisError(f"một so sánh trộn {len(versions)} code_version: {versions} (INT-2)")
    return versions[0]


def check_clean(runs: Iterable[Run], *, draft: bool) -> None:
    """Lượt `dirty` hay `env_match: false` chỉ được dùng ở chế độ --draft (INT-1, INT-2)."""
    if draft:
        return
    bad = [run.run_id for run in runs if run.manifest["dirty"] or not run.manifest["env_match"]]
    if bad:
        raise AnalysisError(
            f"lượt chạy trên cây bẩn hay môi trường lệch requirements.lock: {bad}; chỉ dùng với --draft"
        )


def check_fixed(runs: Sequence[Run], *, axes: Collection[str] = (), references: Collection[str] = ()) -> None:
    """Các lượt được so sánh phải trùng mọi biến cố định trừ `axes`; `references` là run_id của baseline
    vẽ làm đường tham chiếu, được miễn kiểm (INT-2)."""
    for run in runs:
        if run.run_id not in references:
            continue
        kind = run.config["topology"]["kind"]
        if kind == "server":
            raise AnalysisError(
                f"lượt {run.run_id} đi đường server (B3, E1c, E4): nó là ô được so sánh, không phải đường "
                "tham chiếu, nên phải qua kiểm biến cố định (INT-2)"
            )
        if kind not in REFERENCE_KINDS:
            raise AnalysisError(
                f"lượt {run.run_id} ({kind}) không phải baseline B1, B1-4k hay B2, nên không vẽ làm đường "
                "tham chiếu được (INT-2)"
            )
    compared = [run for run in runs if run.run_id not in references]
    for name, value_of in FIXED_VARIABLES.items():
        if name in axes:
            continue
        values = {run.run_id: value_of(run) for run in compared}
        if len({repr(v) for v in values.values()}) > 1:
            raise AnalysisError(
                f"biến cố định {name} khác nhau giữa các lượt được so sánh: {values}; chỉ trục của hình mới "
                "được khác (INT-2)"
            )


def select_seeds(
    groups: Mapping[str, Sequence[Run]], declared: Collection[int] | None
) -> tuple[dict[str, list[Run]], list[tuple[str, int]]]:
    """Lượt của mỗi điều kiện theo tập seed chung, và các cặp (điều kiện, seed) thiếu (INT-2, INT-3).

    Không khai báo tập seed thì mọi điều kiện phải có cùng tập seed. Có khai báo (trong thẻ tiền đăng ký) thì
    chỉ giữ các seed đó; seed khai báo mà thiếu lượt được trả về để báo cáo, không bị lặng lẽ bỏ.
    """
    if declared is None:
        seed_sets = {name: sorted(run.seed for run in group) for name, group in groups.items()}
        if len({tuple(seeds) for seeds in seed_sets.values()}) > 1:
            raise AnalysisError(
                f"tập seed khác nhau giữa các điều kiện: {seed_sets}; khai báo tập seed chung trong thẻ tiền "
                "đăng ký (INT-2)"
            )
        return {name: sorted(group, key=lambda run: run.seed) for name, group in groups.items()}, []
    chosen = {
        name: sorted((run for run in group if run.seed in declared), key=lambda run: run.seed)
        for name, group in groups.items()
    }
    missing = [
        (name, seed)
        for name, group in chosen.items()
        for seed in sorted(declared)
        if seed not in {run.seed for run in group}
    ]
    return chosen, missing


def check_no_nan(table: pd.DataFrame) -> None:
    """Lượt `ok` không được có NaN ở cột số nào; lượt phân kỳ hay thiếu thì để trống có chủ đích (INT-3)."""
    finished = table[table["status"] == "ok"].select_dtypes("number")
    bad = finished.isna()
    if bad.to_numpy().any():
        rows = table.loc[bad.any(axis=1)[bad.any(axis=1)].index, "run_id"].tolist()
        columns = [c for c in bad.columns if bad[c].any()]
        raise AnalysisError(
            f"NaN trong lượt đã xong {rows}, cột {columns}: không lấy trung bình qua NaN (INT-3)"
        )


def attempt_report(attempts: Iterable[Attempts]) -> list[dict[str, Any]]:
    """Mọi attempt `failed` và `diverged` kèm lý do, cho phụ lục báo cáo (INT-3)."""
    report = []
    for entry in attempts:
        for number, error in entry.failed:
            report.append({"run_id": entry.run_id, "attempt": number, "status": "failed", "reason": error})
        run = entry.finished
        if run is not None and run.status == "diverged":
            number = int(run.path.name.removeprefix("attempt-"))
            report.append(
                {
                    "run_id": entry.run_id,
                    "attempt": number,
                    "status": "diverged",
                    "reason": run.summary["reason"],
                }
            )
    return sorted(report, key=lambda item: (item["run_id"], item["attempt"]))


def check_documented(code_versions: Collection[str], ledger: Path) -> None:
    """Mọi code_version có lượt được dùng phải có một dòng trong docs/experiments/CODE_VERSIONS.md (INT-6,
    R9)."""
    if not ledger.exists():
        raise AnalysisError(
            f"thiếu {ledger} (CODE_VERSIONS.md): mọi code_version có lượt phải có một dòng (INT-6)"
        )
    text = ledger.read_text(encoding="utf-8")
    missing = sorted(version for version in code_versions if version not in text)
    if missing:
        raise AnalysisError(f"code_version {missing} chưa có dòng trong {ledger} (INT-6)")

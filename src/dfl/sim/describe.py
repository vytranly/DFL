"""Mô tả dữ liệu của các cách chia, không huấn luyện (module sim; KH §3.2; SPEC REQ-02, QS-11; hình P7).

Mỗi cách chia của bảng KH §3.2, cùng P2' (shards, chỉ dùng ở đây), được chạy với seed 0 … S − 1 qua đúng
đường của một lượt: chia bằng luồng partition rồi gán vị trí node bằng luồng placement của seed đó. Nhờ vậy ma
trận node × lớp trùng với lượt chạy cùng seed. Lệch mục tiêu tính với π = 1/N, như bảng KH §3.2. #C = k cần
N ≥ 10 (G23), nên bị bỏ qua và ghi vào `skipped` khi N nhỏ hơn.

`check_req02` so JS trung bình với cột JS của bảng KH §3.2 (configs/reference/kh_js_table.yaml): mỗi giá trị
lệch không quá max(10%, 0.002), và JS tăng nghiêm ngặt IID < α100 < α1 < α0.5 < α0.1 < α0.05, #C2 < #C1.

csv.DictWriter: https://docs.python.org/3.12/library/csv.html#csv.DictWriter
statistics.fmean: https://docs.python.org/3.12/library/statistics.html#statistics.fmean
dataclasses.asdict: https://docs.python.org/3.12/library/dataclasses.html#dataclasses.asdict
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
import yaml

from dfl import seeding
from dfl.config import RunConfig, canonical_json
from dfl.contracts import Partitioner
from dfl.data.heterogeneity import summarize
from dfl.data.partition import (
    DirichletPartitioner,
    IIDPartitioner,
    LabelKPartitioner,
    QuantityPartitioner,
    ShardsPartitioner,
    assign_positions,
)
from dfl.seeding import Stream

DEFAULT_TABLE = Path("configs") / "reference" / "kh_js_table.yaml"
# Thứ tự JS phải tăng nghiêm ngặt (REQ-02).
ORDERS = (
    ("iid", "dirichlet_100", "dirichlet_1", "dirichlet_0.5", "dirichlet_0.1", "dirichlet_0.05"),
    ("label_2", "label_1"),
)
RELATIVE_TOLERANCE = 0.10
ABSOLUTE_TOLERANCE = 0.002
MEAN_FIELDS = ("js_mean", "tv_mean", "classes_ge_5pct_mean", "n_min", "n_max", "cv_n", "objective_shift_tv")


@dataclass(frozen=True, slots=True)
class Scheme:
    key: str
    label: str
    partitioner: Partitioner


def schemes(min_size: int) -> tuple[Scheme, ...]:
    """Các cách chia của bảng KH §3.2 theo thứ tự hàng của bảng, rồi P2'."""
    return (
        Scheme("iid", "IID", IIDPartitioner(min_size)),
        *(
            Scheme(f"dirichlet_{alpha:g}", f"Dirichlet α = {alpha:g}", DirichletPartitioner(alpha, min_size))
            for alpha in (100.0, 1.0, 0.5, 0.1, 0.05)
        ),
        *(Scheme(f"label_{k}", f"#C = {k}", LabelKPartitioner(k, min_size)) for k in (2, 1)),
        Scheme("quantity_0.5", "Quantity β = 0.5", QuantityPartitioner(0.5, min_size)),
        Scheme("shards", "Shards (P2')", ShardsPartitioner(min_size=min_size)),
    )


def describe(config: RunConfig, labels: np.ndarray, *, seeds: int) -> dict[str, Any]:
    """Báo cáo JSON được của mọi cách chia × seed 0 … `seeds` − 1, với N, min_size của `config`."""
    n = config.num_nodes
    report: dict[str, Any] = {
        "config": json.loads(canonical_json(config)),
        "train_size": len(labels),
        "seeds": list(range(seeds)),
    }
    described: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    for scheme in schemes(config.partition.min_size):
        if isinstance(scheme.partitioner, LabelKPartitioner) and n < 10:
            skipped[scheme.key] = f"#C = k cần N ≥ 10 để mọi nhãn có node sở hữu, nhận N = {n} (G23)"
            continue
        rows = []
        for seed in range(seeds):
            raw = scheme.partitioner(labels, n, seeding.numpy_rng(seed, Stream.PARTITION))
            placed = assign_positions(raw, seeding.numpy_rng(seed, Stream.PLACEMENT))
            stats = dataclasses.asdict(summarize(labels, placed.parts))
            rows.append({"seed": seed, "retries": raw.retries, **stats})
        mean = {name: fmean(float(row[name]) for row in rows) for name in (*MEAN_FIELDS, "retries")}
        described[scheme.key] = {"label": scheme.label, "mean": mean, "per_seed": rows}
    report["schemes"] = described
    report["skipped"] = skipped
    return report


def write(report: Mapping[str, Any], out: Path) -> None:
    """Ghi describe.json (đủ chi tiết, allow_nan=False) và describe.csv (trung bình qua seed)."""
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, allow_nan=False, sort_keys=True, ensure_ascii=False, indent=2)
    (out / "describe.json").write_text(text + "\n", encoding="utf-8")
    with (out / "describe.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["scheme", "label", *MEAN_FIELDS, "retries"])
        writer.writeheader()
        for key, entry in report["schemes"].items():
            writer.writerow({"scheme": key, "label": entry["label"], **entry["mean"]})


def load_table(path: Path) -> dict[str, float]:
    """Cột JS của bảng KH §3.2, đọc bằng yaml.safe_load."""
    table = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(table, dict) or not all(
        isinstance(v, float) and math.isfinite(v) for v in table.values()
    ):
        raise ValueError(f"{path} phải là mapping tên cách chia → JS (số thực)")
    return {str(k): float(v) for k, v in table.items()}


def check_req02(js: Mapping[str, float], table: Mapping[str, float]) -> list[str]:
    """Các vi phạm REQ-02 của JS trung bình `js` so với `table`; rỗng là đạt."""
    problems = []
    required = dict.fromkeys([*table, *(key for order in ORDERS for key in order)])
    for key in required:
        if key not in js:
            problems.append(f"REQ-02: thiếu cách chia {key}")
        elif key in table:
            tolerance = max(RELATIVE_TOLERANCE * table[key], ABSOLUTE_TOLERANCE)
            if abs(js[key] - table[key]) > tolerance:
                problems.append(
                    f"REQ-02: {key} có JS = {js[key]:.4f}, bảng KH §3.2 ghi {table[key]}, "
                    f"lệch quá {tolerance:.4f}"
                )
    for order in ORDERS:
        present = [key for key in order if key in js]
        for low, high in zip(present, present[1:], strict=False):
            if not js[low] < js[high]:
                problems.append(
                    f"REQ-02: JS không tăng nghiêm ngặt, {low} = {js[low]:.4f} ≥ {high} = {js[high]:.4f}"
                )
    return problems


def js_means(report: Mapping[str, Any]) -> dict[str, float]:
    """JS trung bình qua seed của từng cách chia đã mô tả."""
    return {key: float(entry["mean"]["js_mean"]) for key, entry in report["schemes"].items()}

"""Mô tả dữ liệu của các cách chia, không huấn luyện (module sim; KH §3.2; SPEC REQ-02, QS-11; hình P7).

Mỗi cách chia của bảng KH §3.2, cùng P2' (shards, chỉ dùng ở đây), được chạy với seed 0 … S − 1 qua đúng
đường của một lượt: chia bằng luồng partition rồi gán vị trí node bằng luồng placement của seed đó. Nhờ vậy ma
trận node × lớp trùng với lượt chạy cùng seed. Lệch mục tiêu tính với π = 1/N, như bảng KH §3.2. #C = k cần
N ≥ 10 (G23), nên bị bỏ qua và ghi vào `skipped` khi N nhỏ hơn.

`check_req02` so JS trung bình với cột JS của bảng KH §3.2 (configs/reference/kh_js_table.yaml): mỗi giá trị
lệch không quá max(10%, 0.002), và JS tăng nghiêm ngặt IID < α100 < α1 < α0.5 < α0.1 < α0.05, #C2 < #C1.

`describe_topologies` lập bảng topology của KH §3.3 với N của `config` (T08b; ADR-009): spectral gap của W
Metropolis–Hastings cho T1, T4 và T2 (đồ thị seed 0 … S − 1, cùng luồng graph với lượt chạy); với T3 và T3b
là **tốc độ trộn kỳ vọng** 1 − √λ₂(E[W]), không phải spectral gap, vì W đổi theo round (T3 dạng đóng, T3b ước
lượng từ `matchings` round của lịch seed 0). Số message/round = 2·|E| chỉ là thuộc tính của đồ thị; chi phí
thật chỉ do Transport đếm trong lượt chạy (INV-2).

csv.DictWriter: https://docs.python.org/3.12/library/csv.html#csv.DictWriter
statistics.fmean: https://docs.python.org/3.12/library/statistics.html#statistics.fmean
dataclasses.asdict: https://docs.python.org/3.12/library/dataclasses.html#dataclasses.asdict
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
import yaml

from dfl import seeding
from dfl.config import DEFAULT_ER_P, RunConfig, canonical_json
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
from dfl.network import mixing, topology
from dfl.seeding import Stream

DEFAULT_TABLE = Path("configs") / "reference" / "kh_js_table.yaml"
# Thứ tự JS phải tăng nghiêm ngặt (REQ-02).
ORDERS = (
    ("iid", "dirichlet_100", "dirichlet_1", "dirichlet_0.5", "dirichlet_0.1", "dirichlet_0.05"),
    ("label_2", "label_1"),
)
RELATIVE_TOLERANCE = 0.10
ABSOLUTE_TOLERANCE = 0.002
# Giá trị in cạnh bảng của lệnh, lấy từ bảng KH §3.3 (N = 20, MH, p = 0.3).
KH_3_3 = {
    "T1 ring": "0.033",
    "T2 ER": "TB 0.233 (0.149–0.340)",
    "T3 gossip": "0.312 (tốc độ trộn)",
    "T4 FC": "1.000",
}
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
    if seeds < 1:
        raise ValueError(f"cần seeds ≥ 1, nhận {seeds}")
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


def _static_row(build: Callable[[], Any]) -> dict[str, Any]:
    """Hàng của một đồ thị tĩnh; không dựng được với N này (ring cần N ≥ 3) thì ghi lý do, không dừng lệnh."""
    try:
        graph = build()
    except ValueError as error:
        return {"skipped": str(error)}
    return {
        "edges": graph.number_of_edges(),
        "messages_per_round": 2 * graph.number_of_edges(),
        "spectral_gap": mixing.spectral_gap(mixing.metropolis_hastings(graph)),
    }


def _mixing_rate(expected_w: np.ndarray) -> float:
    """1 − √λ₂(E[W]): với W của matching (W² = W), E‖x_{t+1} − x̄‖² co theo λ₂(E[W]) mỗi round (ADR-009)."""
    return float(1.0 - math.sqrt(max(np.sort(np.linalg.eigvalsh(expected_w))[-2], 0.0)))


def gossip_mixing_rate(num_nodes: int) -> float:
    """Dạng đóng cho T3: mỗi cặp được ghép với xác suất q = 1/(N − 1) (N chẵn) hay 1/N (N lẻ), nên E[W] có
    trị riêng 1 − qN/2 trên phần bù của 1; N = 20 cho 0.312 (KH §3.3).
    """
    q = 1 / (num_nodes - 1) if num_nodes % 2 == 0 else 1 / num_nodes
    return 1.0 - math.sqrt(1.0 - q * num_nodes / 2)


def describe_topologies(config: RunConfig, *, seeds: int, matchings: int = 10_000) -> dict[str, Any]:
    """Bảng topology của KH §3.3 với N của `config`; ER dùng p của config hay p mặc định.

    T0 là đường server của B3/E1c (KH §3.3 "Star + server"): server ảo id N nhận N snapshot và gửi lại N, nên
    2N message mỗi round, như engine đếm qua Transport. Topology `star` (hub là node 0, nằm trong N node) là
    một hàng riêng.
    """
    if seeds < 1 or matchings < 1:
        raise ValueError(f"cần seeds ≥ 1 và matchings ≥ 1, nhận {seeds}, {matchings}")
    n = config.num_nodes
    p = config.topology.p if config.topology.p is not None else DEFAULT_ER_P
    rows: dict[str, Any] = {
        "T0 server": {"messages_per_round": 2 * n},
        "star": _static_row(lambda: topology.star(n).graph_at(0)),
        "T1 ring": _static_row(lambda: topology.ring(n).graph_at(0)),
    }
    graphs = [
        topology.erdos_renyi(n, p, seeding.numpy_rng(s, Stream.GRAPH)).graph_at(0) for s in range(seeds)
    ]
    gaps = [mixing.spectral_gap(mixing.metropolis_hastings(graph)) for graph in graphs]
    rows["T2 ER"] = {
        "p": p,
        "spectral_gap": fmean(gaps),
        "spectral_gap_per_seed": gaps,
        "spectral_gap_min": min(gaps),
        "spectral_gap_max": max(gaps),
        "spectral_gap_seed0": gaps[0],
        "edges_seed0": graphs[0].number_of_edges(),
        "messages_per_round_seed0": 2 * graphs[0].number_of_edges(),
    }
    rows["T3 gossip"] = {"mixing_rate": gossip_mixing_rate(n), "messages_per_round": 2 * (n // 2)}
    schedule = topology.er_gossip(graphs[0], lambda r: seeding.numpy_rng(0, Stream.GOSSIP_ER, r))
    expected_w = np.zeros((n, n))
    messages = 0
    for r in range(matchings):
        graph = schedule.graph_at(r)
        expected_w += mixing.metropolis_hastings(graph)
        messages += 2 * graph.number_of_edges()
    rows["T3b ER gossip"] = {
        "mixing_rate": _mixing_rate(expected_w / matchings),
        "matchings": matchings,
        "messages_per_round": messages / matchings,
    }
    rows["T4 FC"] = _static_row(lambda: topology.fully_connected(n).graph_at(0))
    return rows


def topology_table(rows: Mapping[str, Any], config: RunConfig) -> list[str]:
    """Các dòng bảng topology để in. Cột KH §3.3 chỉ in khi N = 20 và p = 0.3, là thiết lập của bảng đó.

    Hàng ER của KH §3.3 được sửa theo đúng các đồ thị này (luồng `Stream.GRAPH`) ngày 2026-10-16; spike
    docs/spikes/spike_topology_partition.py tạo đồ thị theo seed cách khác nên ra số khác.
    """
    p = config.topology.p if config.topology.p is not None else DEFAULT_ER_P
    reference = KH_3_3 if config.num_nodes == 20 and p == DEFAULT_ER_P else {}
    lines = [
        f"Topology (N = {config.num_nodes}, MH, ER p = {p:g})",
        f"{'topology':<15} {'spectral gap':<30} {'tốc độ trộn':<12} {'message/round':<15} "
        + ("KH §3.3" if reference else "(KH §3.3 chỉ so được ở N = 20, p = 0.3)"),
    ]
    for name, row in rows.items():
        if "skipped" in row:
            lines.append(f"{name:<15} bỏ qua: {row['skipped']}")
            continue
        if "spectral_gap_per_seed" in row:
            gap = (
                f"TB {row['spectral_gap']:.3f} ({row['spectral_gap_min']:.3f}–{row['spectral_gap_max']:.3f})"
            )
            messages = f"{row['messages_per_round_seed0']} (seed 0)"
        else:
            gap = f"{row['spectral_gap']:.4f}" if "spectral_gap" in row else "—"
            messages = f"{row['messages_per_round']:g}"
        rate = f"{row['mixing_rate']:.3f}" if "mixing_rate" in row else "—"
        lines.append(f"{name:<15} {gap:<30} {rate:<12} {messages:<15} {reference.get(name, '')}".rstrip())
    lines.append("message/round = 2·|E| là thuộc tính của đồ thị; chi phí thật chỉ do Transport đếm (INV-2)")
    return lines


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

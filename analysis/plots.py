"""Vẽ hình từ kết quả: khung chung, P1, P7, P12–P17 (module analysis; T16a, T29c, T30d, T31b, T31c; SPEC
INT-1, INT-2, INT-6, REQ-10; KH §3.8; ADR-013, ADR-014, ADR-015).

- `python -m analysis.plots --results results/ --out figures/` là cách duy nhất sinh hình cho báo cáo
  (INT-1):
  - không có `--draft`: chỉ đọc lượt trong results/ của repo, qua chỉ mục nhóm thí nghiệm; từ chối khi
    analysis/ có thay đổi chưa commit; mỗi hình qua INT-2 (một code_version, cây sạch, biến cố định) và INT-6;
  - có `--draft`: đọc mọi lượt của thư mục (smoke, demo), nhận lượt bẩn và lịch đánh giá riêng của profile,
    bỏ qua INT-6, gắn nhãn DRAFT lên mọi hình, và không bao giờ được ghi vào figures/.
- Mỗi hình ghi `code_version` của nó lên hình và vào PROVENANCE.json, cùng run_id đã dùng. Hình chưa đủ dữ
  liệu (nhóm thí nghiệm chưa chạy) thì bỏ qua và in tên ra; hình cũ không bị ghi đè bằng hình rỗng.
- Bảng màu Okabe–Ito, an toàn cho người mù màu (REQ-10): https://jfly.uni-koeln.de/color/ . Heatmap dùng
  viridis. Trục có đơn vị.
- **P1:** learning curve, `acc_mean` (%) theo bước cục bộ, trung bình ± std (ddof = 0) qua seed; mỗi ô nhỏ
  một topology, màu theo mức Non-IID. Chỉ lượt DFL (không B1, B2, B3), τ = 20, μ = 0 và η chung (η phổ biến
  nhất, nên các lượt kiểm độ nhạy η/2, 2η của E0b không lọt vào). Lượt phân kỳ không được lấy trung bình: số
  lượt phân kỳ ghi trong chú giải (INT-3).
- **P7:** heatmap node × lớp kèm cột kích thước node, mỗi cách chia một ô, từ `node_class_counts` của manifest
  (lượt có seed nhỏ nhất của mỗi cách chia).
- **P12, P13:** chi phí giao tiếp theo loại (message, byte, thời gian), mức lượt và mức node (ADR-013). Lượt
  DFL và đường server (B1, B2 không có message), τ = 20, μ = 0, η chung như P1, một runtime mỗi hình; mỗi màu
  một topology, cùng màu ở mọi ô. Mỗi loại một ô riêng, không dùng trục kép.
  - P12: (a) message tích lũy, (b) byte tích lũy (trục log) theo bước cục bộ, trung bình qua seed; (c) giây
    mỗi round của từng pha, cột chồng, trung bình qua seed.
  - P13: phân bố qua node và seed của message, byte và giây (train, comm) mỗi round; server ảo là một điểm có
    nhãn. Cần lượt ghi sau ADR-013.
- **P14, P15** (E7, ADR-014): lượt `rare_classes`, mỗi vị trí (hub, edge, random) một ô, bậc tính từ
  `graph_edges` của manifest (analysis/centrality.py). Cần lượt ghi sau ADR-014.
  - P14: accuracy cuối của từng node × seed theo bậc; node giữ lớp 5–9 khác cả màu lẫn hình dạng điểm. Tiêu đề
    ô ghi độ đo của H9 (accuracy trung bình của node không giữ lớp hiếm).
  - P15: đồ thị mạng ở seed nhỏ nhất mà cả ba vị trí đều có; cùng bố cục cho cả ba ô (cùng đồ thị theo G9);
    màu node là accuracy cuối trên một thang tuần tự một sắc chung cho ba ô, cỡ node là tải byte cuối, node
    giữ lớp hiếm có viền đậm.
- **P16** (ADR-015; góp ý buổi 2: theo dõi từng node, so đường hội tụ của hub với leaf): lượt DFL như P1
  trên topology tĩnh; cột là topology, sắp theo bậc trung bình; hàng là tối đa hai mức dữ liệu (trong các mức
  có nhiều topology nhất: ít lệch nhất và lệch nhất theo JS). Đường mảnh là `acc_per_node` của một node ở một
  seed, tô theo loại bậc (hub ≥ 6, giữa 3–5, leaf 1–2); đường đậm là trung bình của mỗi loại, kiểu nét riêng.
  Bậc lấy từ `graph_edges`, hay suy từ tải message với lượt ghi trước ADR-014.
- **P16-E9, P17-E9:** cùng hình, lấy từ E9 (lattice, BA); `code_version` khác lưới lõi nên là hình riêng
  (INT-2).
- **P17** (ADR-015; góp ý buổi 2: so số round với thời gian thực): lượt như P12, ở một mức dữ liệu (mức có
  nhiều topology nhất, hòa thì mức lệch hơn theo JS); (a) accuracy theo round, (b) accuracy theo thời gian
  phân tán ước tính ở 10 Mbit/s, (c) thời gian ước tính để đạt 90% (trung vị qua seed) theo băng thông
  1–1,000 Mbit/s, trục log. Mô hình ước tính ở analysis/comm_cost.py.

matplotlib (backend Agg không cần màn hình; subplots; Axes.imshow; Figure.savefig):
https://matplotlib.org/3.11.2/users/explain/figure/backends.html
https://matplotlib.org/3.11.2/api/_as_gen/matplotlib.pyplot.subplots.html
https://matplotlib.org/3.11.2/api/_as_gen/matplotlib.axes.Axes.imshow.html
https://matplotlib.org/3.11.2/api/figure_api.html
Constrained layout chừa dải dưới cho dòng code_version (ConstrainedLayoutEngine.set(rect=…)):
https://matplotlib.org/3.11.2/api/layout_engine_api.html
Figure.colorbar (một thang màu cho nhiều ô): https://matplotlib.org/3.11.2/api/figure_api.html
networkx.spring_layout (seed cố định thì bố cục cố định):
https://networkx.org/documentation/networkx-3.7/reference/generated/networkx.drawing.layout.spring_layout.html
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from analysis import centrality, checks, comm_cost, convergence, hypotheses, provenance, stats
from analysis.aggregate import AnalysisError, Run, read_experiment, scan
from analysis.convergence import Reach, bytes_at, median_reach, oscillation, run_reach

REPO_ROOT = provenance.REPO_ROOT
RESULTS_ROOT = REPO_ROOT / "results"
FIGURES_ROOT = REPO_ROOT / "figures"
LEDGER = REPO_ROOT / "docs" / "experiments" / "CODE_VERSIONS.md"
OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
BASELINE_KINDS = ("central", "local_only", "server")
DPI = 150
# Phần hình dành cho các ô (trái, dưới, rộng, cao); dải dưới 5% cho dòng code_version · seed.
CONTENT_RECT = (0, 0.05, 1, 0.95)


def level(run: Run) -> str:
    """Nhãn mức Non-IID của cách chia, ví dụ "IID", "α = 0.1", "#C = 2", "β = 0.5"."""
    p = run.config["partition"]
    if p["scheme"] == "dirichlet":
        return f"α = {p['alpha']:g}"
    if p["scheme"] == "label_k":
        return f"#C = {p['k']}"
    if p["scheme"] == "quantity":
        return f"β = {p['beta']:g}"
    return "IID" if p["scheme"] == "iid" else str(p["scheme"])


def _kind(run: Run) -> str:
    return run.config["topology"]["kind"]


def _keep_common(runs: Sequence[Run]) -> list[Run]:
    """Chỉ giữ η và model phổ biến nhất: lượt độ nhạy η/2, 2η của E0b, hay lượt E8 trong một thư mục --draft,
    không lọt vào hình của lưới lõi."""
    kept = list(runs)
    for value in (lambda r: r.config["algorithm"]["lr"], lambda r: r.config["model"]):
        if kept:
            common = Counter(value(r) for r in kept).most_common(1)[0][0]
            kept = [r for r in kept if value(r) == common]
    return kept


def select_p1(runs: Sequence[Run], *, draft: bool) -> list[Run]:
    """Lượt DFL của learning curve; ở --draft nhận mọi τ của profile."""
    dfl = [r for r in runs if _kind(r) not in BASELINE_KINDS and r.config["algorithm"]["mu"] == 0]
    if not draft:
        dfl = [r for r in dfl if r.config["algorithm"]["tau"] == 20]
    return _keep_common(dfl)


def select_comm(runs: Sequence[Run], *, draft: bool) -> list[Run]:
    """Lượt có message của P12, P13: như P1 nhưng giữ đường server; ở --draft nhận mọi τ của profile."""
    sent = [r for r in runs if _kind(r) not in ("central", "local_only") and r.config["algorithm"]["mu"] == 0]
    if not draft:
        sent = [r for r in sent if r.config["algorithm"]["tau"] == 20]
    return _keep_common(sent)


def select_p7(runs: Sequence[Run], *, draft: bool) -> list[Run]:
    return _keep_common([r for r in runs if _kind(r) not in ("central",)])


def learning_curves(
    runs: Sequence[Run],
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    """(topology, mức) → (bước, trung bình, std qua seed, số seed) trên các lượt `ok`."""
    groups: dict[tuple[str, str], list[Run]] = defaultdict(list)
    for run in runs:
        if run.status == "ok":
            groups[(_kind(run), level(run))].append(run)
    curves = {}
    for key, group in groups.items():
        steps = np.array([r["local_steps"] for r in group[0].rounds])
        values = np.array([[r["acc_mean"] for r in run.rounds] for run in group])
        curves[key] = (steps, values.mean(axis=0), values.std(axis=0), len(group))
    return curves


def class_matrices(runs: Sequence[Run]) -> dict[str, np.ndarray]:
    """Mức → ma trận node × lớp của lượt có seed nhỏ nhất của mức đó."""
    first: dict[str, Run] = {}
    for run in sorted(runs, key=lambda r: r.seed):
        first.setdefault(level(run), run)
    return {name: np.array(run.manifest["node_class_counts"]) for name, run in first.items()}


def _figure(ncols: int, width: float, **kwargs: Any) -> tuple[Figure, np.ndarray]:
    figure, axes = plt.subplots(1, ncols, figsize=(width, 4), squeeze=False, layout="constrained", **kwargs)
    figure.get_layout_engine().set(rect=CONTENT_RECT)
    return figure, axes


def _stamp(figure: Figure, runs: Sequence[Run], *, draft: bool) -> None:
    version = checks.single_code_version(runs)
    seeds = sorted({r.seed for r in runs})
    figure.text(
        0.99, 0.005, f"code_version {version[:12]} · seed {seeds}", ha="right", va="bottom", fontsize=7
    )
    if draft:
        figure.text(
            0.5, 0.5, "DRAFT", ha="center", va="center", fontsize=60, color="red", alpha=0.25, rotation=30
        )


def draw_p1(runs: Sequence[Run], *, draft: bool) -> Figure | None:
    curves = learning_curves(runs)
    if not curves:
        return None
    topologies = sorted({topology for topology, _ in curves})
    levels = sorted({name for _, name in curves})
    diverged = Counter((_kind(r), level(r)) for r in runs if r.status == "diverged")
    figure, axes = _figure(len(topologies), 5 * len(topologies), sharey=True)
    for ax, topology in zip(axes[0], topologies, strict=True):
        for i, name in enumerate(levels):
            if (topology, name) not in curves:
                continue
            steps, mean, std, n = curves[(topology, name)]
            color = OKABE_ITO[i % len(OKABE_ITO)]
            extra = f", {diverged[(topology, name)]} phân kỳ" if diverged[(topology, name)] else ""
            ax.plot(steps, 100 * mean, color=color, label=f"{name} ({n} seed{extra})")
            ax.fill_between(steps, 100 * (mean - std), 100 * (mean + std), color=color, alpha=0.2)
        ax.set_title(topology)
        ax.set_xlabel("bước cục bộ")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0][0].set_ylabel("accuracy trung bình qua node (%)")
    figure.suptitle("P1 · learning curve (trung bình ± std qua seed)")
    _stamp(figure, runs, draft=draft)
    return figure


PANELS_PER_ROW = 5


def _class_matrix_figure(matrices: dict[str, np.ndarray], title: str) -> Figure:
    """Mỗi cách chia một heatmap node × lớp kèm cột kích thước node; tối đa PANELS_PER_ROW ô mỗi hàng."""
    names = list(matrices)
    cols = min(PANELS_PER_ROW, len(names))
    rows = math.ceil(len(names) / cols)
    figure, axes = plt.subplots(
        rows,
        2 * cols,
        figsize=(5.5 * cols, 4 * rows),
        squeeze=False,
        layout="constrained",
        gridspec_kw={"width_ratios": [4, 1] * cols},
    )
    figure.get_layout_engine().set(rect=CONTENT_RECT)
    for ax in axes.flat:
        ax.set_axis_off()
    for i, name in enumerate(names):
        counts = matrices[name]
        heat, sizes = axes[i // cols][2 * (i % cols)], axes[i // cols][2 * (i % cols) + 1]
        heat.set_axis_on()
        sizes.set_axis_on()
        image = heat.imshow(counts, aspect="auto", cmap="viridis")
        heat.set_title(name)
        heat.set_xlabel("lớp")
        heat.set_ylabel("node")
        heat.set_xticks(range(counts.shape[1]))
        heat.set_yticks(range(0, len(counts), max(1, len(counts) // 10)))
        figure.colorbar(image, ax=heat, label="số mẫu")
        sizes.barh(range(len(counts)), counts.sum(axis=1), color=OKABE_ITO[4])
        sizes.invert_yaxis()
        sizes.set_xlabel("kích thước\nnode (mẫu)")
        sizes.set_yticks([])
    figure.suptitle(title)
    return figure


def draw_p7(runs: Sequence[Run], *, draft: bool) -> Figure | None:
    """P7 từ node_class_counts của manifest (lượt của lưới)."""
    matrices = class_matrices(runs)
    if not matrices:
        return None
    figure = _class_matrix_figure(dict(sorted(matrices.items())), "P7 · phân bố lớp theo node (từ manifest)")
    _stamp(figure, runs, draft=draft)
    return figure


def describe_matrices(report: dict[str, Any], *, seed: int = 0) -> dict[str, np.ndarray]:
    """Nhãn cách chia → ma trận node × lớp ở `seed`, từ describe.json (T05b), theo JS trung bình tăng dần."""
    if seed not in report["seeds"]:
        raise AnalysisError(f"describe.json không có seed {seed} (có {report['seeds']})")
    if not report["schemes"]:
        raise AnalysisError("describe.json không có cách chia nào")
    matrices = {}
    for entry in sorted(report["schemes"].values(), key=lambda entry: entry["mean"]["js_mean"]):
        rows = [row for row in entry["per_seed"] if row["seed"] == seed]
        if not rows:
            raise AnalysisError(f"cách chia {entry['label']} thiếu seed {seed} trong describe.json")
        matrices[entry["label"]] = np.array(rows[0]["node_class_counts"])
    return matrices


def draw_p7_describe(report: dict[str, Any], *, seed: int = 0) -> Figure:
    """P7 từ describe.json: mọi cách chia của KH §3.2 ở một seed, trước khi có lượt nào (T05b, M2)."""
    config = report["config"]
    figure = _class_matrix_figure(
        describe_matrices(report, seed=seed),
        f"P7 · phân bố lớp theo node ({config['dataset']}, N = {config['num_nodes']}, seed {seed})",
    )
    figure.text(
        0.99, 0.005, f"describe · {report['train_size']:,} ảnh train", ha="right", va="bottom", fontsize=7
    )
    return figure


def topology_label(run: Run) -> str:
    """Tên topology, kèm p của Erdős–Rényi để hai mức p không gộp làm một."""
    p = run.config["topology"].get("p")
    return f"{_kind(run)} p = {p:g}" if p is not None else _kind(run)


# Màu topology: Okabe–Ito bỏ vàng (#F0E442 quá sáng cho đường trên nền trắng) và đen (dành cho server ảo).
# Đã chạy validator của skill dataviz: đạt độ sáng, chroma, tách biệt khi mù màu (ΔE ≥ 15.8); độ tương phản
# thấp với nền được bù bằng nhãn hiển thị (chú giải, nhãn trục).
TOPOLOGY_COLORS = [c for c in OKABE_ITO if c not in ("#F0E442", "#000000")]
# Kiểu nét theo topology, mã hóa thứ hai cạnh màu: hai topology cùng số message (ring và server ở N = 4 đều
# 2N mỗi round) thì đường nét đứt vẫn hiện trên đường liền.
LINESTYLES = ["-", "--", ":", "-."]
# Pha của thời gian mỗi round, theo thứ tự chồng từ dưới lên; "khác" là wall time ngoài mọi pha.
PHASE_LABELS = {"train": "train", "comm": "comm", "combine": "combine", "eval": "eval", "other": "khác"}
PHASE_COLORS = dict(zip(PHASE_LABELS, OKABE_ITO[4:7] + OKABE_ITO[0:2], strict=True))
MEDIAN = {"color": "#000000", "linewidth": 1.5}
MB = 1e6


def _by_topology(runs: Sequence[Run]) -> tuple[dict[str, list[Run]], dict[str, str], Counter[str]]:
    """Lượt `ok` theo topology, màu cố định theo thứ tự tên (cùng màu ở mọi ô), và số lượt phân kỳ."""
    groups: dict[str, list[Run]] = defaultdict(list)
    for run in runs:
        if run.status == "ok":
            groups[topology_label(run)].append(run)
    names = sorted(groups)
    colors = {name: TOPOLOGY_COLORS[i % len(TOPOLOGY_COLORS)] for i, name in enumerate(names)}
    diverged = Counter(topology_label(r) for r in runs if r.status == "diverged")
    return {name: groups[name] for name in names}, colors, diverged


def _seeds_label(name: str, group: Sequence[Run], diverged: Counter[str]) -> str:
    extra = f", {diverged[name]} phân kỳ" if diverged[name] else ""
    return f"{name} ({len(group)} seed{extra})"


def _runtime_label(runtime: str) -> str:
    """Nhãn trục x của ô thời gian; ở inprocess nói rõ comm = 0 là vì runtime (ADR-010)."""
    if runtime == "inprocess":
        return "topology (runtime inprocess: comm = 0, Transport trong bộ nhớ không chờ)"
    return f"topology (runtime {runtime})"


def _headroom(ax: Any, top: float) -> None:
    """Chừa phần trên của ô cho chú giải, để nó không đè lên cột hay hộp."""
    ax.set_ylim(0, 1.4 * top if top > 0 else 1)


def _jitter(center: float, count: int, spread: float = 0.12) -> np.ndarray:
    """Hoành độ rải đều quanh `center`, tất định (không dùng RNG), để các điểm cùng giá trị không chồng."""
    return center + (np.linspace(-spread, spread, count) if count > 1 else np.zeros(count))


def draw_p12(runs: Sequence[Run], *, draft: bool) -> Figure | None:
    """P12: message, byte tích lũy và thời gian mỗi round theo pha, mỗi topology một màu (ADR-013)."""
    groups, colors, diverged = _by_topology(runs)
    if not groups:
        return None
    runtime = comm_cost.single_runtime(runs)
    figure, axes = _figure(3, 16)
    messages_ax, bytes_ax, time_ax = axes[0]
    for i, (name, group) in enumerate(groups.items()):
        series = [comm_cost.cost_series(run) for run in group]
        steps = series[0].steps
        # Đường vẽ trước dày hơn, nên vẫn lộ viền khi đường sau trùng hệt nó.
        width = max(1.5, 3.5 - 0.7 * i)
        style = {"color": colors[name], "linestyle": LINESTYLES[i % len(LINESTYLES)], "linewidth": width}
        label = _seeds_label(name, group, diverged)
        messages_ax.plot(steps, np.mean([s.messages for s in series], axis=0), label=label, **style)
        bytes_ax.plot(steps, np.mean([s.bytes for s in series], axis=0) / MB, label=label, **style)
    for ax, ylabel in ((messages_ax, "message tích lũy"), (bytes_ax, "byte tích lũy (MB)")):
        ax.set_xlabel("bước cục bộ")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    bytes_ax.set_yscale("log")
    messages_ax.legend(fontsize=7)
    names = list(groups)
    bottom = np.zeros(len(names))
    for phase, label in PHASE_LABELS.items():
        heights = np.array(
            [np.mean([comm_cost.phase_seconds_per_round(run)[phase] for run in groups[n]]) for n in names]
        )
        time_ax.bar(
            names,
            heights,
            bottom=bottom,
            color=PHASE_COLORS[phase],
            label=label,
            edgecolor="white",
            linewidth=1,
        )
        bottom += heights
    time_ax.set_ylabel("thời gian mỗi round (s)")
    time_ax.set_xlabel(_runtime_label(runtime), fontsize=8)
    time_ax.tick_params(axis="x", labelrotation=20)
    time_ax.grid(alpha=0.3, axis="y")
    _headroom(time_ax, float(bottom.max()))
    time_ax.legend(fontsize=7, loc="upper center", ncol=len(PHASE_LABELS))
    messages_ax.set_title("(a) message")
    bytes_ax.set_title("(b) byte")
    time_ax.set_title("(c) thời gian, theo pha")
    figure.suptitle("P12 · chi phí giao tiếp theo loại (trung bình qua seed)")
    _stamp(figure, runs, draft=draft)
    return figure


def _node_boxes(
    ax: Any, data: list[np.ndarray], positions: np.ndarray, colors: list[str], width: float
) -> None:
    """Hộp qua node và seed, cùng mỗi node × seed một điểm, để phân bố không độ rộng (mọi node cùng bậc) vẫn
    mang màu."""
    boxes = ax.boxplot(data, positions=positions, widths=width, patch_artist=True, medianprops=MEDIAN)
    for patch, values, x, color in zip(boxes["boxes"], data, positions, colors, strict=True):
        patch.set(facecolor=color, alpha=0.4)
        ax.scatter(_jitter(x, len(values), width / 4), values, s=10, color=color, zorder=3)


def draw_p13(runs: Sequence[Run], *, draft: bool) -> Figure | None:
    """P13: phân bố qua node và seed của tải và thời gian mỗi round; server ảo là một điểm (ADR-013)."""
    groups, colors, _ = _by_topology(runs)
    if not groups:
        return None
    runtime = comm_cost.single_runtime(runs)
    loads = {name: [comm_cost.node_load(run) for run in group] for name, group in groups.items()}
    seconds = {
        name: [comm_cost.node_seconds_per_round(run) for run in group] for name, group in groups.items()
    }
    figure, axes = _figure(3, 16)
    names = list(groups)
    positions = np.arange(len(names))
    for ax, field, scale, ylabel in (
        (axes[0][0], "messages", 1, "message mỗi round"),
        (axes[0][1], "bytes", MB, "byte mỗi round (MB)"),
    ):
        data = [np.concatenate([getattr(load, field) for load in loads[n]]) / scale for n in names]
        _node_boxes(ax, data, positions, [colors[n] for n in names], 0.5)
        servers = [
            (i, getattr(load, f"server_{field}") / scale)
            for i, n in enumerate(names)
            for load in loads[n]
            if getattr(load, f"server_{field}") is not None
        ]
        if servers:
            xs, ys = zip(*servers, strict=True)
            ax.scatter(xs, ys, marker="X", s=64, color=OKABE_ITO[7], label="server ảo", zorder=4)
            ax.legend(fontsize=7)
        ax.set_xticks(positions, names, rotation=20)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3, axis="y")
    time_ax = axes[0][2]
    top = 0.0
    for offset, index, phase in ((-0.2, 0, "train"), (0.2, 1, "comm")):
        data = [np.concatenate([pair[index] for pair in seconds[n]]) for n in names]
        _node_boxes(time_ax, data, positions + offset, [PHASE_COLORS[phase]] * len(names), 0.3)
        time_ax.plot([], [], color=PHASE_COLORS[phase], linewidth=6, label=PHASE_LABELS[phase])
        top = max(top, *(float(values.max()) for values in data))
    time_ax.set_xticks(positions, names, rotation=20)
    time_ax.set_ylabel("thời gian mỗi round (s)")
    time_ax.set_xlabel(_runtime_label(runtime), fontsize=8)
    time_ax.grid(alpha=0.3, axis="y")
    _headroom(time_ax, top)
    time_ax.legend(fontsize=7, loc="upper center", ncol=2)
    axes[0][0].set_title("(a) tải message (gửi + nhận)")
    axes[0][1].set_title("(b) tải byte (gửi + nhận)")
    time_ax.set_title("(c) thời gian của từng node")
    figure.suptitle("P13 · chi phí giao tiếp theo node (phân bố qua node và seed)")
    _stamp(figure, runs, draft=draft)
    return figure


def select_e7(runs: Sequence[Run], *, draft: bool) -> list[Run]:
    """Lượt E7 của P14, P15: cách chia rare_classes với một trong ba vị trí (ADR-014)."""
    return [
        r
        for r in runs
        if r.config["partition"]["scheme"] == "rare_classes"
        and centrality.placement(r) in centrality.PLACEMENTS
    ]


# P14: hai nhóm node, khác cả màu lẫn hình dạng điểm (mã hóa thứ hai cho người mù màu). Cặp xanh–cam này kề
# nhau trong TOPOLOGY_COLORS, đã qua validator của skill dataviz ở T29c.
HOLDER_STYLE = {"color": OKABE_ITO[5], "marker": "D", "s": 40, "edgecolors": "white", "linewidths": 0.8}
OTHER_STYLE = {"color": OKABE_ITO[4], "marker": "o", "s": 24, "edgecolors": "white", "linewidths": 0.5}
# P15: thang tuần tự một sắc, nhạt → đậm theo accuracy.
ACCURACY_CMAP = "Blues"
EDGE_COLOR = "#B0B0B0"


def _by_placement(runs: Sequence[Run]) -> dict[str, list[Run]]:
    groups: dict[str, list[Run]] = defaultdict(list)
    for run in runs:
        if run.status == "ok":
            groups[centrality.placement(run)].append(run)
    return {
        name: sorted(groups[name], key=lambda r: r.seed) for name in centrality.PLACEMENTS if groups[name]
    }


def draw_p14(runs: Sequence[Run], *, draft: bool) -> Figure | None:
    """P14: accuracy cuối của từng node theo bậc, mỗi vị trí một ô (ADR-014)."""
    groups = _by_placement(runs)
    if not groups:
        return None
    figure, axes = _figure(len(groups), 5 * len(groups), sharey=True)
    for ax, (name, group) in zip(axes[0], groups.items(), strict=True):
        for i, run in enumerate(group):
            # Dời ngang theo seed, tất định, để các node cùng bậc của nhiều seed không chồng hẳn lên nhau.
            x = centrality.degrees(run) + 0.08 * (i - (len(group) - 1) / 2)
            y = 100 * centrality.final_accuracy(run)
            held = centrality.holders(run)
            ax.scatter(x[~held], y[~held], label="không giữ lớp 5–9" if i == 0 else None, **OTHER_STYLE)
            ax.scatter(x[held], y[held], label="giữ lớp 5–9" if i == 0 else None, zorder=3, **HOLDER_STYLE)
        measure = 100 * np.mean([centrality.non_holder_accuracy(run) for run in group])
        ax.set_title(f"{name} · node không giữ lớp 5–9: {measure:.1f}% ({len(group)} seed)", fontsize=9)
        ax.set_xlabel("bậc của node")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("accuracy cuối trên đủ tập test (%)")
    axes[0][0].legend(fontsize=7)
    figure.suptitle("P14 · accuracy cuối của từng node theo bậc (E7: lớp 5–9 chỉ ở K node, đặt theo vị trí)")
    _stamp(figure, [run for group in groups.values() for run in group], draft=draft)
    return figure


def draw_p15(runs: Sequence[Run], *, draft: bool) -> Figure | None:
    """P15: đồ thị mạng của một seed chung, màu theo accuracy, cỡ theo tải byte (ADR-014)."""
    groups = _by_placement(runs)
    if not groups:
        return None
    common = set.intersection(*({r.seed for r in group} for group in groups.values()))
    if not common:
        raise AnalysisError(f"P15 cần một seed có ở mọi vị trí {list(groups)}; không có seed chung")
    seed = min(common)
    chosen = {name: next(r for r in group if r.seed == seed) for name, group in groups.items()}
    accuracy = {name: 100 * centrality.final_accuracy(run) for name, run in chosen.items()}
    loads = {name: centrality.final_byte_load(run) for name, run in chosen.items()}
    low = min(float(a.min()) for a in accuracy.values())
    high = max(float(a.max()) for a in accuracy.values())
    heaviest = max(float(load.max()) for load in loads.values()) or 1.0
    figure, axes = _figure(len(chosen), 5 * len(chosen) + 1)
    mappable = None
    for ax, (name, run) in zip(axes[0], chosen.items(), strict=True):
        degree = centrality.degrees(run)
        graph = nx.Graph()
        graph.add_nodes_from(range(len(degree)))
        graph.add_edges_from(centrality.edges(run))
        layout = nx.spring_layout(graph, seed=0)
        nx.draw_networkx_edges(graph, layout, ax=ax, edge_color=EDGE_COLOR, width=1)
        xy = np.array([layout[i] for i in range(len(degree))])
        held = centrality.holders(run)
        mappable = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=accuracy[name],
            cmap=ACCURACY_CMAP,
            vmin=low,
            vmax=high,
            s=60 + 540 * loads[name] / heaviest,
            edgecolors=np.where(held, "#000000", "white"),
            linewidths=np.where(held, 2.5, 0.8),
            zorder=3,
        )
        # Chữ trắng trên nửa đậm của thang màu, chữ đen trên nửa nhạt, để số hiệu node luôn đọc được.
        middle = (low + high) / 2
        for i, (x, y) in enumerate(xy):
            ink = "white" if accuracy[name][i] > middle else "black"
            ax.annotate(str(i), (x, y), fontsize=6, ha="center", va="center", color=ink, zorder=4)
        held_degrees = sorted(degree[held].tolist())
        ax.set_title(f"{name} · seed {seed} · bậc của node giữ lớp 5–9: {held_degrees}", fontsize=9)
        ax.set_axis_off()
    figure.colorbar(mappable, ax=list(axes[0]), label="accuracy cuối (%)", shrink=0.8)
    figure.suptitle(
        f"P15 · đồ thị mạng ở seed {seed}: màu = accuracy cuối, cỡ = tải byte cuối, "
        "viền đen = node giữ lớp 5–9"
    )
    _stamp(figure, list(chosen.values()), draft=draft)
    return figure


# P16: màu và kiểu nét của ba loại bậc (mã hóa thứ hai); ba màu Okabe–Ito, phân biệt được khi mù màu.
DEGREE_STYLE = {"hub": (OKABE_ITO[5], "-"), "giữa": (OKABE_ITO[2], "--"), "leaf": (OKABE_ITO[4], ":")}
DEGREE_LABELS = {"hub": "hub, bậc ≥ 6", "giữa": "giữa, bậc 3–5", "leaf": "leaf, bậc 1–2"}


def select_p16(runs: Sequence[Run], *, draft: bool) -> list[Run]:
    """Lượt DFL của P1 trên topology tĩnh (gossip không có bậc cố định)."""
    return [r for r in select_p1(runs, draft=draft) if _kind(r) in centrality.STATIC_KINDS]


def p16_levels(runs: Sequence[Run]) -> list[str]:
    """Trong các mức có nhiều topology nhất: mức ít lệch nhất và mức lệch nhất theo JS (một mức nếu chỉ có
    một)."""
    topologies: dict[str, set[str]] = defaultdict(set)
    js: dict[str, list[float]] = defaultdict(list)
    for run in runs:
        topologies[level(run)].add(topology_label(run))
        js[level(run)].append(centrality.js_mean(run))
    widest = max(len(names) for names in topologies.values())
    ordered = sorted(
        (n for n in topologies if len(topologies[n]) == widest), key=lambda n: (np.mean(js[n]), n)
    )
    return ordered if len(ordered) <= 2 else [ordered[0], ordered[-1]]


def draw_p16(runs: Sequence[Run], *, draft: bool) -> Figure | None:
    """P16: accuracy của từng node theo bước, tô theo hub/giữa/leaf; mỗi ô một topology × một mức."""
    ok = [r for r in runs if r.status == "ok"]
    if not ok:
        return None
    levels = p16_levels(ok)
    shown = [r for r in ok if level(r) in levels]
    mean_degree: dict[str, list[float]] = defaultdict(list)
    for run in shown:
        mean_degree[topology_label(run)].append(float(centrality.degrees(run).mean()))
    topologies = sorted(mean_degree, key=lambda name: (np.mean(mean_degree[name]), name))
    figure, axes = plt.subplots(
        len(levels),
        len(topologies),
        figsize=(4 * len(topologies), 3.4 * len(levels)),
        squeeze=False,
        sharey=True,
        layout="constrained",
    )
    figure.get_layout_engine().set(rect=CONTENT_RECT)
    for row, name in enumerate(levels):
        for col, topology in enumerate(topologies):
            ax = axes[row][col]
            ax.set_title(f"{topology} · {name}", fontsize=9)
            group = [r for r in shown if level(r) == name and topology_label(r) == topology]
            curves: dict[str, list[np.ndarray]] = defaultdict(list)
            steps: list[int] = []
            for run in group:
                steps = [int(r["local_steps"]) for r in run.rounds]
                accuracy = 100 * np.array([r["acc_per_node"] for r in run.rounds])
                for node, degree in enumerate(centrality.degrees(run)):
                    kind = centrality.degree_class(int(degree))
                    ax.plot(steps, accuracy[:, node], color=DEGREE_STYLE[kind][0], alpha=0.25, linewidth=0.7)
                    curves[kind].append(accuracy[:, node])
            for kind in centrality.DEGREE_CLASSES:
                if curves[kind]:
                    color, style = DEGREE_STYLE[kind]
                    label = f"{DEGREE_LABELS[kind]} ({len(curves[kind])} node·seed)"
                    ax.plot(
                        steps,
                        np.mean(curves[kind], axis=0),
                        color=color,
                        linestyle=style,
                        linewidth=2.5,
                        label=label,
                    )
            if group:
                ax.legend(fontsize=7, loc="best")
            ax.grid(alpha=0.3)
            if row == len(levels) - 1:
                ax.set_xlabel("bước cục bộ")
        axes[row][0].set_ylabel("accuracy của node (%)")
    figure.suptitle(
        "P16 · đường hội tụ của từng node (mảnh: một node ở một seed; đậm: trung bình theo loại bậc)"
    )
    _stamp(figure, shown, draft=draft)
    return figure


# P17: ngưỡng của thời gian để đạt (REQ-04 x ∈ {90, 95}; 90 để nhiều topology đạt hơn ở mức lệch) và dải băng
# thông, từ mạng IoT chậm tới LAN (Mbit/s).
P17_TARGET = 0.90
BANDWIDTHS_MBIT = np.logspace(0, 3, 31)


def pick_level(runs: Sequence[Run]) -> str:
    """Mức dữ liệu có nhiều topology nhất; hòa thì mức lệch hơn theo JS trung bình, rồi theo tên."""
    topologies: dict[str, set[str]] = defaultdict(set)
    js: dict[str, list[float]] = defaultdict(list)
    for run in runs:
        topologies[level(run)].add(topology_label(run))
        js[level(run)].append(centrality.js_mean(run))
    return max(topologies, key=lambda name: (len(topologies[name]), float(np.mean(js[name])), name))


def draw_p17(runs: Sequence[Run], *, draft: bool) -> Figure | None:
    """P17: accuracy theo round và theo thời gian phân tán ước tính; thời gian để đạt 90% theo băng thông."""
    if not runs:
        return None
    chosen = pick_level(runs)
    groups, colors, diverged = _by_topology([r for r in runs if level(r) == chosen])
    if not groups:
        return None
    comm_cost.single_runtime([run for group in groups.values() for run in group])
    figure, axes = _figure(3, 16)
    rounds_ax, time_ax, bandwidth_ax = axes[0]
    unreached = []
    reached_any = False
    for i, (name, group) in enumerate(groups.items()):
        series = [comm_cost.time_series(run) for run in group]
        accuracy = 100 * np.mean([[r["acc_mean"] for r in run.rounds] for run in group], axis=0)
        style = {"color": colors[name], "linestyle": LINESTYLES[i % len(LINESTYLES)], "linewidth": 2}
        label = _seeds_label(name, group, diverged)
        rounds_ax.plot(series[0].rounds, accuracy, label=label, **style)
        estimated = np.mean(
            [comm_cost.estimated_seconds(s, mbit=comm_cost.REFERENCE_MBIT) for s in series], axis=0
        )
        time_ax.plot(estimated, accuracy, label=label, **style)
        # R@x trên mốc của chính lượt (chỉ lượt ok), không kiểm lưới chuẩn: thời gian nội suy trên cùng mốc.
        reaches = [
            convergence.reach(s.steps, [r["acc_mean"] for r in run.rounds], P17_TARGET)
            for s, run in zip(series, group, strict=True)
        ]
        seconds = [
            [
                convergence.value_at(s.steps, comm_cost.estimated_seconds(s, mbit=b), reach)
                for b in BANDWIDTHS_MBIT
            ]
            for s, reach in zip(series, reaches, strict=True)
        ]
        median = np.median(seconds, axis=0)
        if np.all(np.isfinite(median)):
            bandwidth_ax.plot(BANDWIDTHS_MBIT, median, label=label, **style)
            reached_any = True
        else:
            unreached.append(name)
    rounds_ax.set_xlabel("round")
    rounds_ax.set_ylabel("accuracy trung bình qua node (%)")
    rounds_ax.legend(fontsize=7)
    time_ax.set_xlabel(f"thời gian phân tán ước tính (s), B = {comm_cost.REFERENCE_MBIT:g} Mbit/s")
    bandwidth_ax.axvline(comm_cost.REFERENCE_MBIT, color="#808080", linestyle=":", linewidth=1)
    bandwidth_ax.set_xscale("log")
    bandwidth_ax.set_yscale("log")
    bandwidth_ax.set_xlabel("băng thông B (Mbit/s)")
    bandwidth_ax.set_ylabel(f"thời gian ước tính để đạt {100 * P17_TARGET:.0f}% (s)")
    if reached_any:
        bandwidth_ax.legend(fontsize=7)
    if unreached:
        bandwidth_ax.text(
            0.02,
            0.02,
            f"không đạt {100 * P17_TARGET:.0f}% (trung vị qua seed): {', '.join(unreached)}",
            transform=bandwidth_ax.transAxes,
            fontsize=7,
        )
    for ax in axes[0]:
        ax.grid(alpha=0.3)
    rounds_ax.set_title("(a) theo round")
    time_ax.set_title("(b) theo thời gian ước tính")
    bandwidth_ax.set_title(f"(c) thời gian để đạt {100 * P17_TARGET:.0f}% theo băng thông")
    figure.suptitle(
        f"P17 · số round so với thời gian ({chosen}; ước tính = max huấn luyện + max byte của node / B, "
        "các node song song)"
    )
    _stamp(figure, [run for group in groups.values() for run in group], draft=draft)
    return figure


# ---------------------------------------------------------------- P2–P6, P8–P11 và bảng T1 (T16b, T16c, T27c)

TOPOLOGY_ORDER = ("ring", "erdos_renyi", "gossip", "er_gossip", "fully_connected", "server", "star")
LEVEL_ORDER = ("IID", "α = 100", "α = 1", "α = 0.5", "α = 0.1", "α = 0.05", "#C = 2", "#C = 1", "β = 0.5")
CORE = ("e0b_sanity", "e1_noniid", "e2_topology")


def topology_color(kind: str) -> str:
    """Màu cố định theo topology, cùng màu ở mọi hình P2–P6, P8, P9 (TOPOLOGY_COLORS đã qua validator)."""
    if kind not in TOPOLOGY_ORDER:
        return OKABE_ITO[7]
    return TOPOLOGY_COLORS[TOPOLOGY_ORDER.index(kind) % len(TOPOLOGY_COLORS)]


def _order(kind: str) -> int:
    return TOPOLOGY_ORDER.index(kind) if kind in TOPOLOGY_ORDER else len(TOPOLOGY_ORDER)


def _level_order(name: str) -> int:
    return LEVEL_ORDER.index(name) if name in LEVEL_ORDER else len(LEVEL_ORDER)


def _groups(runs: Sequence[Run], key: Callable[[Run], Any]) -> dict[Any, list[Run]]:
    groups: dict[Any, list[Run]] = defaultdict(list)
    for run in runs:
        if run.status == "ok":
            groups[key(run)].append(run)
    return dict(groups)


def final_acc(run: Run) -> float:
    values = run.summary["final_acc_per_node"]
    return sum(values) / len(values)


def _mean_half(values: Sequence[float]) -> tuple[float, float]:
    """Trung bình và nửa độ rộng CI 95% (t) qua seed; một seed thì nửa độ rộng 0."""
    if len(values) < 2:
        return float(np.mean(values)), 0.0
    estimate = stats.mean_ci(values)
    return estimate.mean, estimate.high - estimate.mean


def _every(run: Run) -> int:
    return run.config["eval"]["every_steps"] or run.config["algorithm"]["tau"]


def reach95(run: Run) -> Reach:
    return run_reach(run, 0.95, every=_every(run))


def bytes95(run: Run) -> float:
    steps = [float(r["local_steps"]) for r in run.rounds]
    return bytes_at(steps, [float(r["bytes_cum"]) for r in run.rounds], reach95(run))


def _mean_series(group: Sequence[Run], field: str) -> tuple[np.ndarray, np.ndarray]:
    steps = np.array([r["local_steps"] for r in group[0].rounds], dtype=float)
    return steps, np.mean([[float(r[field]) for r in run.rounds] for run in group], axis=0)


def select_core(runs: Sequence[Run], *, draft: bool) -> list[Run]:
    """Lượt DFedAvg của lưới lõi (như P1): τ = 20, μ = 0, MH, η chung; ở --draft nhận mọi τ."""
    return [r for r in select_p1(runs, draft=draft) if r.config["mixing"] == "metropolis_hastings"]


def draw_p2(runs: Sequence[Run], *, draft: bool) -> Figure | None:
    """P2: accuracy cuối (trung bình ± CI 95%, kèm điểm từng seed) theo JS đo được; mỗi đường một topology."""
    groups = _groups(runs, lambda r: (_kind(r), level(r)))
    if not groups:
        return None
    figure, axes = _figure(1, 9)
    ax = axes[0][0]
    labelled: set[str] = set()
    for rank, kind in enumerate(sorted({k for k, _ in groups}, key=_order)):
        color = topology_color(kind)
        width = max(1.2, 3.0 - 0.5 * rank)  # đường vẽ trước dày hơn, nên vẫn lộ viền khi trùng đường sau
        for schemes, marker, style, scale in (
            (("iid", "dirichlet"), "o", "-", "Dirichlet"),
            (("iid", "label_k"), "s", ":", "#C"),
        ):
            points = []
            for name in sorted({n for k, n in groups if k == kind}, key=_level_order):
                group = groups[(kind, name)]
                if group[0].config["partition"]["scheme"] not in schemes:
                    continue
                xs = [hypotheses.js_mean(r.manifest["node_class_counts"]) for r in group]
                ys = [100 * final_acc(r) for r in group]
                ax.scatter(xs, ys, s=10, color=color, alpha=0.35, marker=marker)
                mean, half = _mean_half(ys)
                points.append((float(np.mean(xs)), mean, half, name))
            if len(points) < 2 and scale == "#C":
                continue
            if points:
                x, y, h, names = zip(*points, strict=True)
                ax.errorbar(
                    x,
                    y,
                    yerr=h,
                    color=color,
                    marker=marker,
                    linestyle=style,
                    linewidth=width,
                    capsize=3,
                    label=f"{kind} · {scale}",
                )
                for xi, yi, name in zip(x, y, names, strict=True):
                    if name not in labelled:
                        ax.annotate(name, (xi, yi), textcoords="offset points", xytext=(4, 6), fontsize=7)
                        labelled.add(name)
    ax.set_xlabel("JS đo được của partition (bit)")
    ax.set_ylabel("accuracy cuối trên đủ tập test (%)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title("trung bình ± CI 95% qua seed; điểm nhạt là từng seed")
    figure.suptitle("P2 · accuracy cuối theo mức Non-IID")
    _stamp(figure, runs, draft=draft)
    return figure


def _heatmap(
    ax: Any, matrix: np.ndarray, rows: Sequence[str], cols: Sequence[str], text: Callable[[float], str]
) -> Any:
    image = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap="viridis")
    finite = matrix[np.isfinite(matrix)]
    low, high = (finite.min(), finite.max()) if finite.size else (0.0, 0.0)
    ax.set_xticks(range(len(cols)), cols, rotation=30)
    ax.set_yticks(range(len(rows)), rows)
    for i in range(len(rows)):
        for j in range(len(cols)):
            value = matrix[i, j]
            if math.isnan(value):
                continue
            # viridis sáng dần về phía vàng: nửa trên của thang dùng chữ đen.
            bright = high > low and (value - low) / (high - low) > 0.5
            ax.text(
                j, i, text(value), ha="center", va="center", fontsize=7, color="black" if bright else "white"
            )
    return image


def draw_p3(runs: Sequence[Run], *, draft: bool) -> Figure | None:
    """P3: heatmap topology × mức Non-IID của (a) accuracy cuối, (b) trung vị R@95 theo bước (H2)."""
    groups = _groups(runs, lambda r: (_kind(r), level(r)))
    if not groups:
        return None
    kinds = sorted({k for k, _ in groups}, key=_order)
    names = sorted({n for _, n in groups}, key=_level_order)
    acc = np.full((len(kinds), len(names)), np.nan)
    r95 = np.full((len(kinds), len(names)), np.nan)
    unreached = np.zeros((len(kinds), len(names)), dtype=bool)
    for (kind, name), group in groups.items():
        i, j = kinds.index(kind), names.index(name)
        acc[i, j] = 100 * float(np.mean([final_acc(r) for r in group]))
        step = float(np.median([reach95(r).step for r in group]))
        unreached[i, j] = math.isinf(step)
        r95[i, j] = step
    figure, axes = _figure(2, 14)
    image = _heatmap(axes[0][0], acc, kinds, names, lambda v: f"{v:.1f}")
    figure.colorbar(image, ax=axes[0][0], label="accuracy cuối (%)")
    finite = np.where(unreached, np.nan, r95)
    image = _heatmap(axes[0][1], finite, kinds, names, lambda v: f"{v:.0f}")
    for i, j in zip(*np.nonzero(unreached), strict=True):
        axes[0][1].text(j, i, "> ngân\nsách", ha="center", va="center", fontsize=6)
    figure.colorbar(image, ax=axes[0][1], label="trung vị R@95 (bước cục bộ)")
    axes[0][0].set_title("(a) accuracy cuối, trung bình qua seed")
    axes[0][1].set_title("(b) R@95, trung vị qua seed")
    figure.suptitle("P3 · topology × mức Non-IID")
    _stamp(figure, runs, draft=draft)
    return figure


P4_LEVELS = ("IID", "α = 1", "α = 0.1", "#C = 2")


def draw_p4(runs: Sequence[Run], *, draft: bool) -> Figure | None:
    """P4: phân bố qua node (và seed) của Acc_i và Acc^loc_i ở đánh giá cuối, theo topology × mức."""
    groups = _groups(runs, lambda r: (_kind(r), level(r)))
    keys = sorted((k for k in groups if k[1] in P4_LEVELS), key=lambda k: (_level_order(k[1]), _order(k[0])))
    if not keys:
        return None
    figure, axes = _figure(2, max(10, 0.9 * len(keys) + 4), sharey=True)
    positions = np.arange(len(keys))
    for ax, field, title in (
        (axes[0][0], "final_acc_per_node", "(a) Acc_i trên tập test chung"),
        (axes[0][1], "final_acc_local_per_node", "(b) Acc^loc_i theo phân bố nhãn của node"),
    ):
        data = [np.concatenate([100 * np.asarray(r.summary[field]) for r in groups[k]]) for k in keys]
        _node_boxes(ax, data, positions, [topology_color(k[0]) for k in keys], 0.6)
        ax.set_xticks(positions, [f"{k[0]}\n{k[1]}" for k in keys], rotation=90, fontsize=7)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
    axes[0][0].set_ylabel("accuracy của node (%)")
    figure.suptitle("P4 · phương sai giữa node (mỗi điểm là một node × seed)")
    _stamp(figure, runs, draft=draft)
    return figure


def draw_p5(runs: Sequence[Run], *, draft: bool) -> Figure | None:
    """P5: drift trước tổng hợp và consensus distance theo bước, theo topology, IID so với α = 0.1 (H3)."""
    groups = _groups(runs, lambda r: (_kind(r), level(r)))
    keys = sorted((k for k in groups if k[1] in ("IID", "α = 0.1")), key=lambda k: (_order(k[0]), k[1]))
    if not keys:
        return None
    figure, axes = _figure(2, 14, sharey=True)
    for ax, field, title in (
        (axes[0][0], "drift_pre_combine", "(a) drift Ξ½ trước tổng hợp"),
        (axes[0][1], "consensus_distance", "(b) consensus distance Ξ sau tổng hợp"),
    ):
        for kind, name in keys:
            steps, values = _mean_series(groups[(kind, name)], field)
            style = "--" if name == "IID" else "-"
            ax.plot(
                steps,
                np.maximum(values, 1e-12),
                color=topology_color(kind),
                linestyle=style,
                label=f"{kind} · {name}",
            )
        ax.set_yscale("log")
        ax.set_xlabel("bước cục bộ")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("khoảng cách RMS tới trung bình (chuẩn L2 của tham số)")
    axes[0][1].legend(fontsize=7)
    figure.suptitle("P5 · drift và consensus distance (trung bình qua seed; nét đứt: IID, liền: α = 0.1)")
    _stamp(figure, runs, draft=draft)
    return figure


def select_p6(runs: Sequence[Run], *, draft: bool) -> list[Run]:
    """P6, P9: lượt DFL với MH và η chung ở α = 0.1 (ở --draft: mọi mức), mọi τ và μ."""
    kept = [
        r
        for r in runs
        if _kind(r) not in BASELINE_KINDS
        and r.config["mixing"] == "metropolis_hastings"
        and (draft or level(r) == "α = 0.1")
    ]
    return _keep_common(kept)


def _config_label(run: Run) -> str:
    algorithm = run.config["algorithm"]
    return f"τ = {algorithm['tau']}, μ = {algorithm['mu']:g}"


def _bytes95_label(group: Sequence[Run]) -> str:
    values = [bytes95(r) for r in group]
    reached = sum(math.isfinite(v) for v in values)
    middle = float(np.median(values))
    shown = "> ngân sách" if math.isinf(middle) else f"{middle / MB:.0f} MB"
    return f"bytes@95 {shown}, {reached}/{len(values)} đạt"


def _accuracy_vs_bytes(ax: Any, group: Sequence[Run], **style: Any) -> None:
    _, acc = _mean_series(group, "acc_mean")
    _, cost = _mean_series(group, "bytes_cum")
    ax.plot(cost / MB, 100 * acc, **style)


def draw_p6(runs: Sequence[Run], *, draft: bool) -> Figure | None:
    """P6: accuracy theo byte tích lũy (trục log), (a) theo topology ở τ = 20, (b) theo τ trên ring và FC."""
    by_topology = _groups(
        [r for r in runs if r.config["algorithm"]["mu"] == 0],
        lambda r: (_kind(r), r.config["algorithm"]["tau"]),
    )
    if not by_topology:
        return None
    figure, axes = _figure(2, 15, sharey=True)
    taus = sorted({tau for _, tau in by_topology})
    reference = 20 if 20 in taus else taus[0]
    for (kind, tau), group in sorted(by_topology.items(), key=lambda item: (_order(item[0][0]), item[0][1])):
        label = f"{kind} ({_bytes95_label(group)})"
        if tau == reference:
            _accuracy_vs_bytes(axes[0][0], group, color=topology_color(kind), label=label)
        if kind in ("ring", "fully_connected"):
            style = LINESTYLES[taus.index(tau) % len(LINESTYLES)]
            _accuracy_vs_bytes(
                axes[0][1],
                group,
                color=topology_color(kind),
                linestyle=style,
                label=f"{kind}, τ = {tau} ({_bytes95_label(group)})",
            )
    for ax, title in ((axes[0][0], f"(a) topology, τ = {reference}"), (axes[0][1], "(b) τ trên ring và FC")):
        ax.set_xscale("log")
        ax.set_xlabel("byte tích lũy của cả mạng (MB, trục log)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6)
    axes[0][0].set_ylabel("accuracy trung bình qua node (%)")
    figure.suptitle("P6 · accuracy theo chi phí giao tiếp (trung bình qua seed; DFedAvg, μ = 0)")
    _stamp(figure, runs, draft=draft)
    return figure


def expected_gossip_rate(num_nodes: int) -> float:
    """Tốc độ trộn kỳ vọng của gossip trên K_N, 1 − √λ₂(E[W]) (ADR-009); 0.312 với N = 20."""
    lam = 0.5 - 1 / (2 * (num_nodes - 1)) if num_nodes % 2 == 0 else 0.5
    return 1 - math.sqrt(lam)


def draw_p8(runs: Sequence[Run], *, draft: bool) -> Figure | None:
    """P8: mức suy giảm ghép cặp (IID − α = 0.1) theo spectral gap của từng lượt; gossip vẽ riêng (RQ2)."""
    groups = _groups(runs, lambda r: (_kind(r), level(r), r.seed))
    pairs = [
        (kind, seed, groups[(kind, "IID", seed)][0], groups[(kind, "α = 0.1", seed)][0])
        for kind, name, seed in groups
        if name == "IID" and (kind, "α = 0.1", seed) in groups
    ]
    if not pairs:
        return None
    figure, axes = _figure(2, 13, sharey=True, gridspec_kw={"width_ratios": [3, 1]})
    static, dynamic = axes[0]
    for kind in sorted({p[0] for p in pairs}, key=_order):
        points = [
            (skewed.manifest["spectral_gap"], 100 * (final_acc(iid) - final_acc(skewed)))
            for k, _, iid, skewed in pairs
            if k == kind
        ]
        if points[0][0] is None:
            xs = [TOPOLOGY_ORDER.index(kind)] * len(points)
            dynamic.scatter(xs, [y for _, y in points], color=topology_color(kind), label=kind)
            continue
        static.scatter([x for x, _ in points], [y for _, y in points], color=topology_color(kind), label=kind)
    static.set_xscale("log")
    static.set_xlabel("spectral gap của đồ thị của lượt (trục log)")
    static.set_ylabel("mức suy giảm: Acc IID − Acc α = 0.1 (điểm %)")
    static.set_title("(a) topology tĩnh, mỗi điểm một seed")
    dynamic_kinds = sorted({p[0] for p in pairs if p[3].manifest["spectral_gap"] is None}, key=_order)
    n = pairs[0][3].config["num_nodes"]
    dynamic.set_xticks(
        [TOPOLOGY_ORDER.index(k) for k in dynamic_kinds],
        [
            f"{k}\n(tốc độ trộn kỳ vọng {expected_gossip_rate(n):.3f})" if k == "gossip" else k
            for k in dynamic_kinds
        ],
        fontsize=7,
    )
    dynamic.set_title("(b) gossip: không có spectral gap cố định")
    for ax in (static, dynamic):
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
    figure.suptitle(
        "P8 · suy giảm do Non-IID theo spectral gap (chỉ mô tả: ba cụm giá trị, lẫn với bậc và byte)"
    )
    _stamp(figure, runs, draft=draft)
    return figure


def draw_p9(runs: Sequence[Run], *, draft: bool) -> Figure | None:
    """P9: quét τ và μ trên ring và FC: accuracy theo bước cục bộ (hàng trên) và theo byte (hàng dưới)."""
    groups = _groups(
        [r for r in runs if _kind(r) in ("ring", "fully_connected")], lambda r: (_kind(r), _config_label(r))
    )
    if not groups:
        return None
    configs = sorted({c for _, c in groups}, key=lambda c: [float(v.split("= ")[1]) for v in c.split(", ")])
    figure, axes = plt.subplots(2, 2, figsize=(14, 8), squeeze=False, layout="constrained", sharey=True)
    figure.get_layout_engine().set(rect=CONTENT_RECT)
    for col, kind in enumerate(("ring", "fully_connected")):
        for i, config_name in enumerate(configs):
            group = groups.get((kind, config_name))
            if not group:
                continue
            color = TOPOLOGY_COLORS[i % len(TOPOLOGY_COLORS)]
            steps, acc = _mean_series(group, "acc_mean")
            axes[0][col].plot(steps, 100 * acc, color=color, label=f"{config_name} ({len(group)} seed)")
            _accuracy_vs_bytes(axes[1][col], group, color=color, label=config_name)
        axes[0][col].set_title(kind)
        axes[0][col].set_xlabel("bước cục bộ")
        axes[1][col].set_xscale("log")
        axes[1][col].set_xlabel("byte tích lũy của cả mạng (MB, trục log)")
        if axes[0][col].get_legend_handles_labels()[0]:
            axes[0][col].legend(fontsize=7)
    for ax in axes.flat:
        ax.grid(alpha=0.3)
    for row in axes:
        row[0].set_ylabel("accuracy trung bình qua node (%)")
    figure.suptitle("P9 · quét τ và μ ở cùng τ·R (trung bình qua seed)")
    _stamp(figure, runs, draft=draft)
    return figure


def select_e8(runs: Sequence[Run], *, draft: bool) -> list[Run]:
    """P10, P11: lượt E8 (process runtime) của thang model."""
    return [r for r in runs if r.config["runtime"] == "processes" and r.config["model"] in hypotheses.PARAMS]


def _per_round(run: Run, field: str) -> float:
    previous, values = 0, []
    for record in run.timing:
        values.append(float(record[field]) / (int(record["round"]) - previous))
        previous = int(record["round"])
    return float(np.median(values))


def draw_p10(runs: Sequence[Run], *, draft: bool) -> Figure | None:
    """P10: byte mỗi round, thời gian giao tiếp và tỉ lệ giao tiếp trên tính toán theo số tham số (H8)."""
    groups = _groups([r for r in runs if r.timing], lambda r: (_kind(r), r.config["model"]))
    if not groups:
        return None
    figure, axes = _figure(3, 17)
    kinds = sorted({k for k, _ in groups}, key=_order)
    for rank, kind in enumerate(kinds):
        models = sorted((m for k, m in groups if k == kind), key=hypotheses.PARAMS.__getitem__)
        params = [hypotheses.PARAMS[m] for m in models]
        per_round = [
            np.mean([r.rounds[-1]["bytes_cum"] / r.rounds[-1]["round"] for r in groups[(kind, m)]])
            for m in models
        ]
        comm = [np.mean([_per_round(r, "comm_s") for r in groups[(kind, m)]]) for m in models]
        train = [np.mean([_per_round(r, "train_s") for r in groups[(kind, m)]]) for m in models]
        style = {
            "color": topology_color(kind),
            "marker": "o",
            "label": kind,
            "linestyle": LINESTYLES[rank % len(LINESTYLES)],
            "linewidth": max(1.2, 3.0 - 0.6 * rank),
        }
        axes[0][0].plot(params, np.asarray(per_round) / MB, **style)
        axes[0][1].plot(params, comm, **style)
        axes[0][1].plot(params, train, color=topology_color(kind), marker="x", linestyle=":")
        axes[0][2].plot(params, np.asarray(comm) / (np.asarray(comm) + np.asarray(train)), **style)
    for ax, ylabel, title in (
        (axes[0][0], "byte mỗi round của cả mạng (MB)", "(a) byte (tất định)"),
        (
            axes[0][1],
            "giây mỗi round (trung vị qua mốc)",
            "(b) comm (tròn) và train (×), lớn nhất qua node",
        ),
        (axes[0][2], "comm / (comm + train)", "(c) tỉ lệ giao tiếp trên tính toán"),
    ):
        ax.set_xscale("log")
        ax.set_xlabel("số tham số của model (trục log)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0][0].set_yscale("log")
    axes[0][1].set_yscale("log")
    axes[0][0].legend(fontsize=7)
    figure.suptitle(
        f"P10 · chi phí giao tiếp theo độ phức tạp model (runtime processes, {_runtime_note(runs)})"
    )
    _stamp(figure, runs, draft=draft)
    return figure


def _runtime_note(runs: Sequence[Run]) -> str:
    return f"N = {runs[0].config['num_nodes']}, một máy, loopback"


def draw_p11(runs: Sequence[Run], *, draft: bool) -> Figure | None:
    """P11: accuracy theo byte tích lũy, mỗi đường một model; mỗi ô một topology (E8)."""
    groups = _groups(runs, lambda r: (_kind(r), r.config["model"], level(r)))
    if not groups:
        return None
    kinds = sorted({k for k, _, _ in groups}, key=_order)
    figure, axes = _figure(len(kinds), 5 * len(kinds) + 1, sharey=True)
    models = sorted({m for _, m, _ in groups}, key=hypotheses.PARAMS.__getitem__)
    for ax, kind in zip(axes[0], kinds, strict=True):
        for i, model in enumerate(models):
            for name, style in (("IID", "-"), ("α = 0.1", "--")):
                group = groups.get((kind, model, name))
                if group:
                    label = f"{model} ({hypotheses.PARAMS[model]:,} tham số)" if name == "IID" else None
                    _accuracy_vs_bytes(
                        ax,
                        group,
                        color=TOPOLOGY_COLORS[i % len(TOPOLOGY_COLORS)],
                        linestyle=style,
                        label=label,
                    )
        ax.set_xscale("log")
        ax.set_xlabel("byte tích lũy của cả mạng (MB, trục log)")
        ax.set_title(kind)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("accuracy trung bình qua node (%)")
    axes[0][0].legend(fontsize=7)
    figure.suptitle(
        "P11 · accuracy theo byte, mỗi đường một model (liền: IID, đứt: α = 0.1; η không chọn lại theo model)"
    )
    _stamp(figure, runs, draft=draft)
    return figure


def select_t1(runs: Sequence[Run], *, draft: bool) -> list[Run]:
    """Bảng T1: mọi ô chính (trừ B1 một node, vốn không cùng N); chỉ lượt η chung."""
    return _keep_common([r for r in runs if _kind(r) != "central"])


def _summary_mean(runs: Sequence[Run], field: str, reduce: Callable[..., Any] = np.mean) -> float:
    """100 × trung bình qua lượt của `reduce` (trung bình, hay std ddof = 0) của một dãy theo node."""
    return 100 * float(np.mean([reduce(r.summary[field]) for r in runs]))


def table_t1(runs: Sequence[Run]) -> pd.DataFrame:
    """T1: acc, acc^loc, macro-F1, R@95 (trung vị, k/n), bytes@95, độ dao động, std giữa node cho mọi ô."""

    def key(run: Run) -> tuple[Any, ...]:
        a = run.config["algorithm"]
        return (_kind(run), level(run), a["tau"], a["mu"], run.config["mixing"])

    cells: dict[tuple[Any, ...], list[Run]] = defaultdict(list)
    for run in runs:
        cells[key(run)].append(run)
    rows = []
    for (kind, name, tau, mu, mixing), group in sorted(
        cells.items(), key=lambda item: (_order(item[0][0]), _level_order(item[0][1]), item[0][2:])
    ):
        ok = [r for r in group if r.status == "ok"]
        row: dict[str, Any] = {
            "topology": kind,
            "mức": name,
            "τ": tau,
            "μ": mu,
            "trộn": mixing,
            "seed": len(group),
            "phân kỳ": len(group) - len(ok),
        }
        if ok:
            acc = [100 * final_acc(r) for r in ok]
            mean, half = _mean_half(acc)
            reaches = [reach95(r) for r in ok]
            middle, label = median_reach(reaches)
            costs = [bytes95(r) for r in ok]
            swings = [
                oscillation(np.array([x["acc_per_node"] for x in r.rounds])) for r in ok if len(r.rounds) >= 3
            ]
            row |= {
                "acc (%)": f"{mean:.2f} ± {half:.2f}",
                "acc^loc (%)": f"{_summary_mean(ok, 'final_acc_local_per_node'):.2f}",
                "macro-F1 (%)": f"{_summary_mean(ok, 'final_f1_macro_per_node'):.2f}",
                "R@95 (bước)": ("> ngân sách" if math.isinf(middle) else f"{middle:.0f}") + f" ({label})",
                "bytes@95 (MB)": "> ngân sách"
                if math.isinf(np.median(costs))
                else f"{np.median(costs) / MB:.0f}",
                "độ dao động": f"{np.mean(swings):.4f}" if swings else "",
                "std giữa node (%)": f"{_summary_mean(ok, 'final_acc_per_node', np.std):.2f}",
            }
        rows.append(row)
    return pd.DataFrame(rows)


# Bảng → (nhóm thí nghiệm, hàm chọn lượt, hàm lập bảng); ghi ra <tên>.csv và <tên>.md.
TABLES: dict[str, tuple[tuple[str, ...], Callable[..., list[Run]], Callable[..., pd.DataFrame]]] = {
    "T1": (
        (
            "e0b_sanity",
            "e1_noniid",
            "e2_topology",
            "e3a_algorithms",
            "e4a_quantity",
            "e1c_objective",
            "e2b_gossip",
            "e3b_tau_mu",
            "e4_quantity",
        ),
        select_t1,
        table_t1,
    ),
}
# Trục được phép khác giữa các lượt của một hình (INT-2): E8 đổi model.
FIGURE_AXES: dict[str, tuple[str, ...]] = {"P10": ("model",), "P11": ("model",)}


# Tên hình → (nhóm thí nghiệm cấp dữ liệu khi không có --draft, hàm chọn lượt, hàm vẽ).
FIGURES: dict[str, tuple[tuple[str, ...], Callable[..., list[Run]], Callable[..., Figure | None]]] = {
    "P1": (("e1_noniid", "e0b_sanity"), select_p1, draw_p1),
    "P2": (CORE, select_core, draw_p2),
    "P3": (CORE, select_core, draw_p3),
    "P4": (CORE, select_core, draw_p4),
    "P5": (CORE, select_core, draw_p5),
    "P6": (("e1_noniid", "e2_topology", "e3a_algorithms", "e3b_tau_mu"), select_p6, draw_p6),
    "P7": (("e1_noniid",), select_p7, draw_p7),
    "P8": (CORE, select_core, draw_p8),
    "P9": (("e1_noniid", "e3a_algorithms", "e3b_tau_mu"), select_p6, draw_p9),
    "P10": (("e8_model_complexity",), select_e8, draw_p10),
    "P11": (("e8_model_complexity",), select_e8, draw_p11),
    "P12": (("e1_noniid", "e2_topology", "e0b_sanity"), select_comm, draw_p12),
    "P13": (("e1_noniid", "e2_topology", "e0b_sanity"), select_comm, draw_p13),
    "P14": (("e7_centrality",), select_e7, draw_p14),
    "P15": (("e7_centrality",), select_e7, draw_p15),
    "P16": (("e1_noniid", "e2_topology"), select_p16, draw_p16),
    "P16-E9": (("e9_topology_shapes",), select_p16, draw_p16),
    "P17": (("e1_noniid", "e2_topology", "e0b_sanity"), select_comm, draw_p17),
    "P17-E9": (("e9_topology_shapes",), select_comm, draw_p17),
}
# Ở --draft mọi hình đọc cùng một thư mục, nên bản -E9 trùng hệt bản gốc: bỏ qua.
DRAFT_DUPLICATES = frozenset({"P16-E9", "P17-E9"})


def _markdown(table: pd.DataFrame) -> str:
    """Bảng Markdown tự viết (DataFrame.to_markdown cần gói tabulate, không có trong requirements.lock)."""
    header = "| " + " | ".join(str(c) for c in table.columns) + " |"
    rule = "|" + "|".join("---" for _ in table.columns) + "|"
    body = [
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
        for row in table.itertuples(index=False)
    ]
    return "\n".join([header, rule, *body]) + "\n"


def _inside(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def _runs_of(results: Path, experiments: Sequence[str], *, draft: bool) -> list[Run]:
    if draft:
        attempts = scan(results) if (results / "runs").is_dir() else []
    else:
        attempts = [
            a
            for name in experiments
            if (results / "experiments" / f"{name}.json").exists()
            for a in read_experiment(results, name)
        ]
    unique = {a.run_id: a.finished for a in attempts if a.finished is not None}
    return list(unique.values())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m analysis.plots", description="Vẽ hình từ kết quả (INT-1)."
    )
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("figures"))
    parser.add_argument("--draft", action="store_true", help="smoke/demo: nhãn DRAFT, không ghi vào figures/")
    parser.add_argument("--describe", type=Path, help="vẽ P7 từ describe.json (T05b), chỉ ra ngoài figures/")
    parser.add_argument("--seed", type=int, default=0, help="seed của P7 từ describe.json")
    args = parser.parse_args(argv)
    if args.describe is not None:
        if _inside(args.out, FIGURES_ROOT):
            raise AnalysisError(
                "P7 từ describe.json chỉ ghi ra ngoài figures/ (INT-1); chọn --out như build/describe"
            )
        report = json.loads(args.describe.read_text(encoding="utf-8"))
        figure = draw_p7_describe(report, seed=args.seed)
        args.out.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.out / "P7-describe.png", dpi=DPI)
        plt.close(figure)
        print(f"đã vẽ {args.out / 'P7-describe.png'}")
        return 0
    if args.draft and _inside(args.out, FIGURES_ROOT):
        raise AnalysisError(
            "--draft không bao giờ được ghi vào figures/ (INT-1); chọn --out khác, ví dụ build/"
        )
    if not args.draft:
        if not _inside(args.results, RESULTS_ROOT):
            raise AnalysisError(
                f"không có --draft thì chỉ đọc lượt trong results/ của repo (INT-1), nhận {args.results}"
            )
        if provenance.analysis_dirty():
            raise AnalysisError(
                "analysis/ có thay đổi chưa commit; commit trước khi vẽ hình cho báo cáo (INT-1)"
            )
    command = "python -m analysis.plots " + " ".join(shlex.quote(a) for a in (argv or sys.argv[1:]))
    entries: dict[str, dict[str, Any]] = {}
    for name, (experiments, select, draw) in FIGURES.items():
        if args.draft and name in DRAFT_DUPLICATES:
            continue
        runs = select(_runs_of(args.results, experiments, draft=args.draft), draft=args.draft)
        if runs:
            # --draft chỉ nới lượt bẩn, lịch đánh giá riêng và INT-6 (INT-1); biến cố định và một code_version
            # mỗi hình vẫn được kiểm.
            version = checks.single_code_version(runs)
            checks.check_fixed(runs, axes=FIGURE_AXES.get(name, ()))
            if not args.draft:
                checks.check_clean(runs, draft=False)
                checks.check_documented({version}, LEDGER)
        figure = draw(runs, draft=args.draft) if runs else None
        if figure is None:
            print(f"bỏ qua {name}: chưa đủ dữ liệu")
            continue
        args.out.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.out / f"{name}.png", dpi=DPI)
        plt.close(figure)
        entries[name] = {
            "code_version": checks.single_code_version(runs),
            "run_ids": sorted(r.run_id for r in runs),
        }
        print(f"đã vẽ {args.out / f'{name}.png'}")
    for name, (experiments, select, tabulate) in TABLES.items():
        runs = select(_runs_of(args.results, experiments, draft=args.draft), draft=args.draft)
        if not runs:
            print(f"bỏ qua {name}: chưa đủ dữ liệu")
            continue
        version = checks.single_code_version(runs)
        checks.check_fixed(runs)
        if not args.draft:
            checks.check_clean(runs, draft=False)
            checks.check_documented({version}, LEDGER)
        table = tabulate(runs)
        args.out.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out / f"{name}.csv", index=False)
        (args.out / f"{name}.md").write_text(_markdown(table), encoding="utf-8")
        entries[name] = {"code_version": version, "run_ids": sorted(r.run_id for r in runs)}
        print(f"đã lập {args.out / f'{name}.csv'}")
    if entries or (args.out / "PROVENANCE.json").exists():
        args.out.mkdir(parents=True, exist_ok=True)
        provenance.update(args.out / "PROVENANCE.json", command=command, draft=args.draft, figures=entries)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AnalysisError as err:
        print(f"lỗi: {err}", file=sys.stderr)
        raise SystemExit(1) from None

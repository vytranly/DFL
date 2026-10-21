"""Vẽ hình từ kết quả: khung chung, P1, P7, P12, P13, P14 và P15 (module analysis; T16a, T29c, T30d; SPEC
INT-1, INT-2, INT-6, REQ-10; KH §3.8; ADR-013, ADR-014).

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
from matplotlib.figure import Figure

from analysis import centrality, checks, comm_cost, provenance
from analysis.aggregate import AnalysisError, Run, read_experiment, scan

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


def select_p1(runs: Sequence[Run], *, draft: bool) -> list[Run]:
    """Lượt DFL của learning curve; ở --draft nhận mọi τ của profile."""
    dfl = [r for r in runs if _kind(r) not in BASELINE_KINDS and r.config["algorithm"]["mu"] == 0]
    if not draft:
        dfl = [r for r in dfl if r.config["algorithm"]["tau"] == 20]
    if not dfl:
        return []
    common = Counter(r.config["algorithm"]["lr"] for r in dfl).most_common(1)[0][0]
    return [r for r in dfl if r.config["algorithm"]["lr"] == common]


def select_comm(runs: Sequence[Run], *, draft: bool) -> list[Run]:
    """Lượt có message của P12, P13: như P1 nhưng giữ đường server; ở --draft nhận mọi τ của profile."""
    sent = [r for r in runs if _kind(r) not in ("central", "local_only") and r.config["algorithm"]["mu"] == 0]
    if not draft:
        sent = [r for r in sent if r.config["algorithm"]["tau"] == 20]
    if not sent:
        return []
    common = Counter(r.config["algorithm"]["lr"] for r in sent).most_common(1)[0][0]
    return [r for r in sent if r.config["algorithm"]["lr"] == common]


def select_p7(runs: Sequence[Run], *, draft: bool) -> list[Run]:
    return [r for r in runs if _kind(r) not in ("central",)]


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
        graph.add_edges_from(run.manifest["graph_edges"])
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


# Tên hình → (nhóm thí nghiệm cấp dữ liệu khi không có --draft, hàm chọn lượt, hàm vẽ).
FIGURES: dict[str, tuple[tuple[str, ...], Callable[..., list[Run]], Callable[..., Figure | None]]] = {
    "P1": (("e1_noniid", "e0b_sanity"), select_p1, draw_p1),
    "P7": (("e1_noniid",), select_p7, draw_p7),
    "P12": (("e1_noniid", "e2_topology", "e0b_sanity"), select_comm, draw_p12),
    "P13": (("e1_noniid", "e2_topology", "e0b_sanity"), select_comm, draw_p13),
    "P14": (("e7_centrality",), select_e7, draw_p14),
    "P15": (("e7_centrality",), select_e7, draw_p15),
}


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
        runs = select(_runs_of(args.results, experiments, draft=args.draft), draft=args.draft)
        if runs:
            # --draft chỉ nới lượt bẩn, lịch đánh giá riêng và INT-6 (INT-1); biến cố định và một code_version
            # mỗi hình vẫn được kiểm.
            version = checks.single_code_version(runs)
            checks.check_fixed(runs)
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

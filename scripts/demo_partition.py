"""Demo chia dữ liệu từng bước (T31f; ADR-015; góp ý buổi 2 của GVHD).

GVHD muốn thấy trên một dataset cụ thể: từng bước mỗi cách chia làm gì; tham số (α của Dirichlet, k của
Pathological) đổi phân bố dữ liệu ra sao; ảnh hưởng của α có giống nhau giữa các dataset không. Script làm lại
từng bước của Dirichlet (NIID-Bench `noniid-labeldir`) và #C = k (NIID-Bench `noniid-#label{k}`) với đúng thứ
tự rút số ngẫu nhiên của src/dfl/data/partition.py, in các bước trung gian, rồi kiểm kết quả trùng từng chỉ số
với partitioner của dự án. Sau đó quét α và k, mô tả các cách chia khác, và so MNIST với CIFAR-10 (nếu đã
tải).

Nằm ngoài src/ nên không đổi code_version. Chạy từ gốc repo bằng `-m` để import được `analysis`:
    .venv/bin/python -m scripts.demo_partition --out build/demo_partition
Kết quả: demo_partition.md (mọi bảng, cũng in ra màn hình), alpha_effect.png, P7-describe.png.

numpy.random.Generator.dirichlet: https://numpy.org/doc/2.5/reference/random/generated/numpy.random.Generator.dirichlet.html
numpy.array_split: https://numpy.org/doc/2.5/reference/generated/numpy.array_split.html
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis import plots
from dfl import seeding
from dfl.config import CIFAR10_INPUT, DATASET_INPUTS, RunConfig
from dfl.data import datasets
from dfl.data.heterogeneity import summarize
from dfl.data.partition import (
    MAX_RETRIES,
    NUM_CLASSES,
    DirichletPartitioner,
    LabelKPartitioner,
    QuantityPartitioner,
    RareClassPartitioner,
    ShardsPartitioner,
)
from dfl.seeding import Stream
from dfl.sim import describe

ALPHAS = (1000.0, 100.0, 10.0, 5.0, 1.0, 0.5, 0.3, 0.1, 0.05)
KS = (1, 2, 3, 5, 10)
BETAS = (0.5, 1.0, 5.0)
TRACE_ALPHAS = (10.0, 0.5, 0.1)
TRACE_K = 2
JS_LEVELS = (100.0, 1.0, 0.5, 0.1, 0.05)


@dataclass(frozen=True)
class ClassStep:
    """Một lớp của Dirichlet: tỉ lệ rút từ Dir_N(α), node đã đầy (đạt cỡ trung bình), số mẫu mỗi node nhận."""

    label: int
    size: int
    drawn: np.ndarray
    full: np.ndarray
    counts: np.ndarray


def dirichlet_steps(
    labels: np.ndarray, num_nodes: int, alpha: float, rng: np.random.Generator, min_size: int
) -> tuple[list[np.ndarray], int, list[ClassStep]]:
    """Từng bước của DirichletPartitioner, cùng thứ tự rút số; trả về các phần, số lần lấy mẫu lại và vết của
    lần thử thành công."""
    cap = len(labels) / num_nodes
    class_indices = [np.flatnonzero(labels == k) for k in range(NUM_CLASSES)]
    for attempt in range(MAX_RETRIES + 1):
        parts: list[list[int]] = [[] for _ in range(num_nodes)]
        trace: list[ClassStep] = []
        for label, idx_k in enumerate(class_indices):
            idx = rng.permutation(idx_k)  # bước 1: xáo mẫu của lớp
            drawn = rng.dirichlet(np.full(num_nodes, alpha))  # bước 2: tỉ lệ p ~ Dir_N(α)
            full = np.array([len(nodes) >= cap for nodes in parts])  # bước 3: node đã đủ cỡ trung bình
            masked = np.where(full, 0.0, drawn)
            if masked.sum() == 0:
                break
            cuts = (np.cumsum(masked / masked.sum()) * len(idx)).astype(int)[:-1]  # bước 4: điểm cắt
            chunks = np.split(idx, cuts)
            for node, chunk in enumerate(chunks):
                parts[node].extend(chunk.tolist())
            trace.append(ClassStep(label, len(idx), drawn, full, np.array([len(c) for c in chunks])))
        else:
            if min(len(nodes) for nodes in parts) >= min_size:  # bước 5: node nào quá nhỏ thì rút lại cả lượt
                return [np.array(nodes, dtype=np.int64) for nodes in parts], attempt, trace
    raise ValueError(f"Dirichlet α = {alpha}: không chia được sau {MAX_RETRIES:,} lần lấy mẫu lại")


def label_k_steps(
    labels: np.ndarray, num_nodes: int, k: int, rng: np.random.Generator
) -> tuple[list[np.ndarray], list[list[int]], dict[int, list[int]]]:
    """Từng bước của LabelKPartitioner: nhãn của mỗi node, rồi chia đều mẫu mỗi nhãn cho các node giữ nó."""
    owned: list[list[int]] = []
    for node in range(num_nodes):
        node_labels = [node % NUM_CLASSES]  # nhãn đầu cố định: mọi nhãn đều có chủ
        while len(node_labels) < k:
            c = int(rng.integers(NUM_CLASSES))  # k − 1 nhãn còn lại rút ngẫu nhiên, không trùng
            if c not in node_labels:
                node_labels.append(c)
        owned.append(node_labels)
    parts: list[list[int]] = [[] for _ in range(num_nodes)]
    chunks_of: dict[int, list[int]] = {}
    for c in range(NUM_CLASSES):
        owners = [node for node in range(num_nodes) if c in owned[node]]
        idx = rng.permutation(np.flatnonzero(labels == c))
        chunks = np.array_split(idx, len(owners))  # chia đều: các chủ chênh nhau tối đa 1 mẫu
        chunks_of[c] = [len(chunk) for chunk in chunks]
        for node, chunk in zip(owners, chunks, strict=True):
            parts[node].extend(chunk.tolist())
    return [np.array(nodes, dtype=np.int64) for nodes in parts], owned, chunks_of


def same_parts(mine: Sequence[np.ndarray], theirs: Sequence[np.ndarray]) -> bool:
    return len(mine) == len(theirs) and all(np.array_equal(a, b) for a, b in zip(mine, theirs, strict=True))


def theory_cv(num_nodes: int, alpha: float) -> float:
    """CV của phần một lớp mà một node nhận khi chưa có giới hạn cỡ: p_i ~ Beta(α, (N − 1)α), nên
    Var = (1/N)(1 − 1/N)/(Nα + 1) và CV = √((N − 1)/(Nα + 1)). Chỉ phụ thuộc N và tích Nα."""
    return math.sqrt((num_nodes - 1) / (num_nodes * alpha + 1))


def _config(dataset: str, num_nodes: int, min_size: int) -> RunConfig:
    """Cấu hình cho describe (P7): dataset của nhãn, model khớp shape ảnh (ADR-012), N và min_size của "
    "demo."""
    model = "cnn_niidbench" if DATASET_INPUTS[dataset] == CIFAR10_INPUT else "smallcnn"
    config = RunConfig(dataset=dataset, model=model, num_nodes=num_nodes)
    return dataclasses.replace(config, partition=dataclasses.replace(config.partition, min_size=min_size))


def _yes(value: bool) -> str:
    return "có" if value else "KHÔNG"


def _rng(seed: int) -> np.random.Generator:
    return seeding.numpy_rng(seed, Stream.PARTITION)


def _row_stats(labels: np.ndarray, parts: Sequence[np.ndarray]) -> dict[str, float]:
    stats = summarize(labels, parts)
    counts = np.array(stats.node_class_counts)
    return {
        "js": stats.js_mean,
        "classes": stats.classes_ge_5pct_mean,
        "empty": float((counts == 0).mean()),
        "n_min": stats.n_min,
        "n_max": stats.n_max,
        "cv_n": stats.cv_n,
    }


def _mean_stats(labels: np.ndarray, splits: Sequence[Sequence[np.ndarray]]) -> dict[str, float]:
    rows = [_row_stats(labels, parts) for parts in splits]
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def _matrix(labels: np.ndarray, parts: Sequence[np.ndarray]) -> list[str]:
    counts = np.array(summarize(labels, parts).node_class_counts)
    lines = ["| node | " + " | ".join(str(c) for c in range(NUM_CLASSES)) + " | tổng |", "|---" * 12 + "|"]
    for node, row in enumerate(counts):
        lines.append(f"| {node} | " + " | ".join(f"{v:,}" for v in row) + f" | {row.sum():,} |")
    return lines


def dirichlet_section(labels: np.ndarray, name: str, n: int, min_size: int) -> tuple[list[str], bool]:
    lines = [f"## 1. Dirichlet từng bước ({name}, N = {n}, seed 0)", ""]
    lines += [
        "Với mỗi lớp c (theo thứ tự 0 … 9): (1) xáo các mẫu của lớp; (2) rút tỉ lệ p ~ Dir_N(α), mỗi node "
        "một phần; (3) node đã đủ cỡ trung bình (tổng số mẫu / N) không nhận thêm: tỉ lệ của nó về 0, phần "
        "còn lại chuẩn hóa lại (quy tắc `balance` của NIID-Bench); (4) cắt dãy mẫu theo tỉ lệ tích lũy. Hết "
        f"10 lớp, nếu có node nhỏ hơn {min_size} mẫu thì rút lại cả lượt. Số mẫu của một lớp ở một node vì "
        "vậy là **ngẫu nhiên**, và α càng nhỏ thì tỉ lệ càng dồn vào vài node.",
        "",
    ]
    all_same = True
    for alpha in TRACE_ALPHAS:
        parts, retries, trace = dirichlet_steps(labels, n, alpha, _rng(0), min_size)
        same = same_parts(parts, DirichletPartitioner(alpha, min_size)(labels, n, _rng(0)).parts)
        all_same &= same
        first = trace[0]
        top = np.argsort(first.drawn)[::-1][:5]
        lines += [
            f"### α = {alpha:g} (Nα = {n * alpha:g}; lấy mẫu lại {retries} lần; trùng partitioner: "
            f"{'có' if same else 'KHÔNG'})",
            "",
            f"- Lớp {first.label} có {first.size:,} mẫu. Năm tỉ lệ lớn nhất rút từ Dir_{n}({alpha:g}): "
            + ", ".join(f"node {i}: {first.drawn[i]:.3f}" for i in top)
            + f"; tổng năm tỉ lệ này là {first.drawn[top].sum():.2f}.",
            f"- Số node nhận ít nhất một mẫu của lớp {first.label}: {int((first.counts > 0).sum())}/{n}; "
            f"nhiều nhất một node nhận {first.counts.max():,} mẫu.",
            "- Qua 10 lớp: "
            + "; ".join(
                f"lớp {s.label}: {int((s.counts > 0).sum())} node nhận, {int(s.full.sum())} node đã đầy"
                for s in trace
            )
            + ".",
            "",
        ]
        if alpha == 0.5:
            lines += [f"Ma trận node × lớp với α = {alpha:g}:", "", *_matrix(labels, parts), ""]
    return lines, all_same


def label_k_section(labels: np.ndarray, name: str, n: int, min_size: int) -> tuple[list[str], bool]:
    parts, owned, chunks_of = label_k_steps(labels, n, TRACE_K, _rng(0))
    same = same_parts(parts, LabelKPartitioner(TRACE_K, min_size)(labels, n, _rng(0)).parts)
    lines = [
        f"## 2. Pathological #C = k từng bước ({name}, N = {n}, k = {TRACE_K}, seed 0; trùng partitioner: "
        f"{'có' if same else 'KHÔNG'})",
        "",
        "Bước 1: node i giữ nhãn i mod 10 (để nhãn nào cũng có chủ), cộng k − 1 nhãn rút ngẫu nhiên, không "
        "trùng. Bước 2: với mỗi nhãn c, mẫu của c được xáo rồi **chia đều** cho các node giữ c. Vì vậy, trả "
        "lời câu hỏi của GVHD: hai node cùng giữ một nhãn nhận số mẫu của nhãn đó **gần như bằng nhau** "
        "(chênh tối đa 1), không ngẫu nhiên; cái ngẫu nhiên là **node nào giữ nhãn nào**, nên số node giữ "
        "mỗi nhãn khác nhau, và nhãn ít chủ thì mỗi chủ nhận phần lớn hơn. Kích thước node vì vậy không đều.",
        "",
        "| node | nhãn giữ | số mẫu mỗi nhãn |",
        "|---|---|---|",
    ]
    owners = {c: [node for node in range(n) if c in owned[node]] for c in range(NUM_CLASSES)}
    for node in range(n):
        sizes = [chunks_of[c][owners[c].index(node)] for c in owned[node]]
        lines.append(
            f"| {node} | {', '.join(map(str, owned[node]))} | {', '.join(f'{s:,}' for s in sizes)} |"
        )
    lines += [
        "",
        "| nhãn | số node giữ | số mẫu mỗi node giữ (min–max) |",
        "|---|---|---|",
        *(
            f"| {c} | {len(owners[c])} | {min(chunks_of[c]):,}–{max(chunks_of[c]):,} |"
            for c in range(NUM_CLASSES)
        ),
        "",
    ]
    return lines, same


def alpha_section(labels: np.ndarray, name: str, n: int, seeds: int, min_size: int) -> tuple[list[str], dict]:
    lines = [
        f"## 3. Ảnh hưởng của α ({name}, N = {n}, trung bình {seeds} seed)",
        "",
        "CV lý thuyết là độ lệch tương đối của phần một lớp mà một node nhận, √((N − 1)/(Nα + 1)): chỉ phụ "
        "thuộc **Nα**. Nα ≫ 1 thì mọi node nhận gần đúng 1/N mỗi lớp (gần IID); Nα ≈ 1 trở xuống thì mỗi lớp "
        "dồn vào vài node. Cùng α nhưng N khác thì mức lệch khác.",
        "",
        "| α | Nα | CV lý thuyết | JS TB (bit) | số lớp ≥ 5% mỗi node | % ô node×lớp rỗng | n_min | n_max | "
        "lấy mẫu lại TB |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    js_of: dict[float, float] = {}
    for alpha in ALPHAS:
        try:
            results = [DirichletPartitioner(alpha, min_size)(labels, n, _rng(s)) for s in range(seeds)]
        except ValueError as error:
            lines.append(
                f"| {alpha:g} | {n * alpha:g} | {theory_cv(n, alpha):.2f} | không chia được: {error} |"
            )
            continue
        stats = _mean_stats(labels, [r.parts for r in results])
        js_of[alpha] = stats["js"]
        retries = np.mean([r.retries for r in results])
        lines.append(
            f"| {alpha:g} | {n * alpha:g} | {theory_cv(n, alpha):.2f} | {stats['js']:.3f} | "
            f"{stats['classes']:.1f} | {100 * stats['empty']:.0f}% | {stats['n_min']:,.0f} | "
            f"{stats['n_max']:,.0f} | {retries:.1f} |"
        )
    lines += [""]
    return lines, js_of


def k_section(labels: np.ndarray, name: str, n: int, seeds: int, min_size: int) -> tuple[list[str], dict]:
    lines = [
        f"## 4. Ảnh hưởng của k trong #C = k ({name}, N = {n}, trung bình {seeds} seed)",
        "",
        "k là tham số rời rạc: mỗi node có **đúng** k nhãn, nên không tạo được mức lệch nằm giữa hai giá "
        "trị k liền nhau, khác với α liên tục. k = 10 là mọi node giữ mọi nhãn, tức gần IID.",
        "",
        "| k | JS TB (bit) | số lớp ≥ 5% mỗi node | n_min | n_max | CV kích thước |",
        "|---|---|---|---|---|---|",
    ]
    js_of: dict[int, float] = {}
    for k in KS:
        results = [LabelKPartitioner(k, min_size)(labels, n, _rng(s)) for s in range(seeds)]
        stats = _mean_stats(labels, [r.parts for r in results])
        js_of[k] = stats["js"]
        lines.append(
            f"| {k} | {stats['js']:.3f} | {stats['classes']:.1f} | {stats['n_min']:,.0f} | "
            f"{stats['n_max']:,.0f} | {stats['cv_n']:.2f} |"
        )
    lines += [""]
    return lines, js_of


def other_section(labels: np.ndarray, name: str, n: int, seeds: int, min_size: int) -> list[str]:
    lines = [
        f"## 5. Các cách chia khác ({name}, N = {n}, trung bình {seeds} seed)",
        "",
        "- **Quantity skew** (β): nhãn IID, chỉ kích thước node lệch, q ~ Dir_N(β). Lệch về số lượng, "
        "không lệch nhãn, nên JS gần 0.",
        "- **Shards** (McMahan 2017): sắp mẫu theo nhãn, cắt N × 2 mảnh, mỗi node nhận 2 mảnh ngẫu nhiên; "
        "kích thước đều, số nhãn mỗi node thường là 2 nhưng có thể 1 hay hơn 2, nên không trùng #C = 2.",
        "- **rare_classes** (E7, ADR-014): lớp 0–4 chia đều cho mọi node, lớp 5–9 chỉ cho K node.",
        "",
        "| cách chia | JS TB (bit) | số lớp ≥ 5% mỗi node | n_min | n_max | CV kích thước |",
        "|---|---|---|---|---|---|",
    ]
    schemes = [(f"quantity β = {b:g}", QuantityPartitioner(b, min_size)) for b in BETAS]
    schemes += [
        ("shards (2 mảnh)", ShardsPartitioner(2, min_size)),
        ("rare_classes K = 2", RareClassPartitioner(2)),
    ]
    for label, partitioner in schemes:
        stats = _mean_stats(labels, [partitioner(labels, n, _rng(s)).parts for s in range(seeds)])
        lines.append(
            f"| {label} | {stats['js']:.3f} | {stats['classes']:.1f} | {stats['n_min']:,.0f} | "
            f"{stats['n_max']:,.0f} | {stats['cv_n']:.2f} |"
        )
    lines += [""]
    return lines


def dataset_section(names: Sequence[str], alpha_js: dict, k_js: dict, class_sizes: dict, n: int) -> list[str]:
    lines = [
        f"## 6. Cùng α trên các dataset khác nhau (N = {n})",
        "",
        "Độ lệch nhãn của một cách chia chỉ phụ thuộc số lớp, mức cân bằng giữa các lớp và N (cùng Nα), "
        "không phụ thuộc nội dung ảnh. MNIST và CIFAR-10 cùng 10 lớp gần cân bằng, nên cùng α cho JS gần "
        "như nhau. Cái khác giữa hai dataset là **ảnh hưởng tới accuracy**: CIFAR-10 khó hơn, nên cùng một "
        "mức lệch nhãn làm giảm accuracy nhiều hơn (cần thí nghiệm để đo, không suy từ bảng này).",
        "",
        "| dataset | số mẫu của node | mẫu mỗi lớp (min–max) | "
        + " | ".join(f"JS α = {a:g}" for a in JS_LEVELS)
        + " | JS #C = 2 | JS #C = 1 |",
        "|---" * (5 + len(JS_LEVELS)) + "|",
    ]
    for name in names:
        sizes = class_sizes[name]
        cells = [f"{alpha_js[name].get(a, float('nan')):.3f}" for a in JS_LEVELS]
        lines.append(
            f"| {name} | {sizes.sum():,} | {sizes.min():,}–{sizes.max():,} | "
            + " | ".join(cells)
            + f" | {k_js[name][2]:.3f} | {k_js[name][1]:.3f} |"
        )
    lines += [""]
    return lines


def alpha_figure(alpha_js: dict, k_js: dict, n: int) -> plt.Figure:
    figure, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for i, (name, values) in enumerate(alpha_js.items()):
        alphas = sorted(values)
        color = plots.TOPOLOGY_COLORS[i]
        ax.plot(
            alphas,
            [values[a] for a in alphas],
            marker="o",
            color=color,
            linewidth=2,
            label=f"{name}: Dirichlet",
        )
        for k, style in ((2, "--"), (1, ":")):
            ax.axhline(k_js[name][k], color=color, linestyle=style, linewidth=1, label=f"{name}: #C = {k}")
    ax.set_xscale("log")
    ax.set_xlabel(f"α (log; N = {n}, nên Nα = {n} × α)")
    ax.set_ylabel("JS trung bình giữa node và toàn cục (bit)")
    ax.set_title("Ảnh hưởng của α: α nhỏ → lệch nhãn mạnh; #C = k là các mức rời rạc")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    return figure


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Demo chia dữ liệu từng bước (T31f).")
    parser.add_argument("--out", type=Path, default=Path("build/demo_partition"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--datasets", default="mnist,cifar10", help="dataset đầu tiên dùng cho phần từng bước"
    )
    parser.add_argument("--nodes", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--min-size", type=int, default=64)
    args = parser.parse_args(argv)
    n, seeds, min_size = args.nodes, args.seeds, args.min_size
    labels_of: dict[str, np.ndarray] = {}
    for name in args.datasets.split(","):
        try:
            labels_of[name] = datasets.load(name, root=args.data_root, with_test=False).train.y.numpy()
        except OSError as error:  # dataset chưa tải về data/
            print(f"bỏ qua {name}: {error}")
    if not labels_of:
        raise SystemExit("không nạp được dataset nào")
    first = next(iter(labels_of))
    labels = labels_of[first]
    sizes = ", ".join(f"{name} {len(values):,} mẫu" for name, values in labels_of.items())
    lines = [
        "# Demo chia dữ liệu từng bước",
        "",
        "Sinh bằng `python -m scripts.demo_partition` trên nhãn thật của phần train dành cho node (sau khi "
        f"tách validation): {sizes}. Số ngẫu nhiên lấy từ luồng partition của seed như lượt chạy.",
        "",
    ]
    dirichlet_lines, dirichlet_same = dirichlet_section(labels, first, n, min_size)
    label_lines, label_same = label_k_section(labels, first, n, min_size)
    lines += dirichlet_lines + label_lines
    alpha_js: dict[str, dict] = {}
    k_js: dict[str, dict] = {}
    for name, values in labels_of.items():
        section, alpha_js[name] = alpha_section(values, name, n, seeds, min_size)
        lines += section
        section, k_js[name] = k_section(values, name, n, seeds, min_size)
        lines += section
    lines += other_section(labels, first, n, seeds, min_size)
    class_sizes = {name: np.bincount(values, minlength=NUM_CLASSES) for name, values in labels_of.items()}
    lines += dataset_section(list(labels_of), alpha_js, k_js, class_sizes, n)
    lines += [
        "## 7. Kiểm chứng",
        "",
        f"- Các bước Dirichlet ở trên trùng từng chỉ số với `DirichletPartitioner`: {_yes(dirichlet_same)}.",
        f"- Các bước #C = {TRACE_K} ở trên trùng từng chỉ số với `LabelKPartitioner`: {_yes(label_same)}.",
        "- Bảng JS cùng định nghĩa với `python -m dfl describe` (bảng KH §3.2, REQ-02).",
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    (args.out / "demo_partition.md").write_text(text, encoding="utf-8")
    figure = alpha_figure(alpha_js, k_js, n)
    figure.savefig(args.out / "alpha_effect.png", dpi=plots.DPI)
    plt.close(figure)
    report = describe.describe(_config(first, n, min_size), labels, seeds=1)
    figure = plots.draw_p7_describe(report, seed=0)
    figure.savefig(args.out / "P7-describe.png", dpi=plots.DPI)
    plt.close(figure)
    print(text)
    print(f"đã ghi {args.out / 'demo_partition.md'}, alpha_effect.png, P7-describe.png")
    return 0 if dirichlet_same and label_same else 1


if __name__ == "__main__":
    raise SystemExit(main())

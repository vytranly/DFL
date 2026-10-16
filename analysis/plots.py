"""Vẽ hình từ kết quả: khung chung, P1 và P7 (module analysis; T16a; SPEC INT-1, INT-2, INT-6, REQ-10; KH
§3.8).

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

matplotlib (backend Agg không cần màn hình; subplots; Axes.imshow; Figure.savefig):
https://matplotlib.org/3.11.2/users/explain/figure/backends.html
https://matplotlib.org/3.11.2/api/_as_gen/matplotlib.pyplot.subplots.html
https://matplotlib.org/3.11.2/api/_as_gen/matplotlib.axes.Axes.imshow.html
https://matplotlib.org/3.11.2/api/figure_api.html
Constrained layout chừa dải dưới cho dòng code_version (ConstrainedLayoutEngine.set(rect=…)):
https://matplotlib.org/3.11.2/api/layout_engine_api.html
"""

from __future__ import annotations

import argparse
import shlex
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from analysis import checks, provenance
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


def draw_p7(runs: Sequence[Run], *, draft: bool) -> Figure | None:
    matrices = class_matrices(runs)
    if not matrices:
        return None
    names = sorted(matrices)
    figure, axes = _figure(
        2 * len(names), 5.5 * len(names), gridspec_kw={"width_ratios": [4, 1] * len(names)}
    )
    for i, name in enumerate(names):
        counts = matrices[name]
        heat, sizes = axes[0][2 * i], axes[0][2 * i + 1]
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
    figure.suptitle("P7 · phân bố lớp theo node (từ manifest)")
    _stamp(figure, runs, draft=draft)
    return figure


# Tên hình → (nhóm thí nghiệm cấp dữ liệu khi không có --draft, hàm chọn lượt, hàm vẽ).
FIGURES: dict[str, tuple[tuple[str, ...], Callable[..., list[Run]], Callable[..., Figure | None]]] = {
    "P1": (("e1_noniid", "e0b_sanity"), select_p1, draw_p1),
    "P7": (("e1_noniid",), select_p7, draw_p7),
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
    args = parser.parse_args(argv)
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

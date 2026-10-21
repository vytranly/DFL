"""Kết luận của H1–H8 và RQ4 theo đúng phép so sánh chính đã đăng ký (module analysis; T17a, T18a, T19a, T27c,
T17b, T19b, T20b; ADR-007; SPEC EXP-4; thẻ docs/experiments/E1.md … E8.md).

- Mỗi giả thuyết một phép so sánh chính: một đại lượng theo seed, chiều kỳ vọng và quy tắc của `stats` (5 seed
  hay 3 seed; kiểm định dấu cho đại lượng bị kiểm duyệt). Mọi phép so khác chỉ để khám phá và không ở đây.
- Định nghĩa dùng chung (docs/experiments/E1.md): accuracy cuối = trung bình `final_acc_per_node` (đủ tập
  test); JS = trung bình qua node của JS (log₂) giữa phân bố nhãn của node và phân bố gộp, từ
  `node_class_counts`, cùng định nghĩa với `dfl.data.heterogeneity.js_mean` (analysis không import được
  `dfl.data`, contract `analysis-reads-only`); drift theo thời gian = trung bình `drift_pre_combine` qua mốc.
- Lượt lấy theo nhóm thí nghiệm (chỉ mục do sweep ghi) rồi theo ô. Mỗi phép so: một `code_version`, cây sạch,
  trùng biến cố định trừ trục của nó (INT-2); thiếu ô hay seed là AnalysisError, không âm thầm thu nhỏ phép so
  (INT-3); có lượt phân kỳ thì "chưa kết luận" kèm "k/n lượt phân kỳ" (ADR-007).
- `python -m analysis.hypotheses --results results/ [--only H1 H3 …]` in giá trị theo seed, trung bình kèm CI
  95%, kết luận, khuyến nghị (REQ-13) và số run ID; phép so chưa đủ lượt thì in lý do.

Độ dốc OLS lấy từ `stats.h2_slopes`, cùng công thức với H2.
statistics.median: https://docs.python.org/3.12/library/statistics.html#statistics.median
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np

from analysis import checks, stats
from analysis.aggregate import AnalysisError, Run, read_experiment
from analysis.convergence import Reach, bytes_at, run_reach

ETA = 0.1  # η chung của lưới (ADR-007, E0a)
SEEDS5, SEEDS3 = (0, 1, 2, 3, 4), (0, 1, 2)
MH = "metropolis_hastings"
RING, FC, ER, GOSSIP = "ring", "fully_connected", "erdos_renyi", "gossip"
IID: tuple[str, float | None] = ("iid", None)
A01: tuple[str, float | None] = ("dirichlet", 0.1)
DIRICHLET_LADDER: tuple[tuple[str, float | None], ...] = (
    IID,
    ("dirichlet", 100.0),
    ("dirichlet", 1.0),
    ("dirichlet", 0.5),
    A01,
    ("dirichlet", 0.05),
)
# Số tham số của thang model (ADR-011, test số tham số ở T09c).
PARAMS = {"logreg": 7_850, "smallcnn": 46_730, "mlp2nn": 199_210, "cnn_mcmahan": 1_663_370}
GROUPS = (
    "e0b_sanity",
    "e1_noniid",
    "e2_topology",
    "e3a_algorithms",
    "e4a_quantity",
    "e8_model_complexity",
    "e1c_objective",
    "e2b_gossip",
    "e3b_tau_mu",
    "e4_quantity",
)


def js_mean(counts: Sequence[Sequence[int]]) -> float:
    """JS trung bình (bit) giữa phân bố nhãn của từng node và phân bố gộp."""
    array = np.asarray(counts, dtype=float)
    dists = array / array.sum(axis=1, keepdims=True)
    pooled = array.sum(axis=0) / array.sum()

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    def js(p: np.ndarray) -> float:
        m = 0.5 * (p + pooled)
        return 0.5 * kl(p, m) + 0.5 * kl(pooled, m)

    return float(np.mean([js(d) for d in dists]))


def level(run: Run) -> tuple[str, float | None]:
    p = run.config["partition"]
    value = {"dirichlet": p["alpha"], "label_k": p["k"], "quantity": p["beta"]}.get(p["scheme"])
    return p["scheme"], None if value is None else float(value)


def final_accuracy(run: Run) -> float:
    values = run.summary["final_acc_per_node"]
    return sum(values) / len(values)


def mean_drift(run: Run) -> float:
    return sum(float(r["drift_pre_combine"]) for r in run.rounds) / len(run.rounds)


@dataclass(frozen=True)
class Cell:
    """Một ô của lưới: η luôn là η chung, để lượt độ nhạy của E0b không lọt vào."""

    kind: str
    level: tuple[str, float | None]
    tau: int = 20
    mu: float = 0.0
    mixing: str = MH
    model: str = "smallcnn"

    def matches(self, run: Run) -> bool:
        cfg, algorithm = run.config, run.config["algorithm"]
        return (
            cfg["topology"]["kind"] == self.kind
            and level(run) == self.level
            and algorithm["tau"] == self.tau
            and float(algorithm["mu"]) == self.mu
            and float(algorithm["lr"]) == ETA
            and cfg["mixing"] == self.mixing
            and cfg["model"] == self.model
        )


def cell(runs: Sequence[Run], spec: Cell, seeds: Sequence[int]) -> dict[int, Run]:
    """Lượt của ô `spec` theo seed; thiếu seed hay hai lượt cùng seed là lỗi."""
    found: dict[int, Run] = {}
    for run in runs:
        if run.seed in seeds and spec.matches(run):
            if run.seed in found:
                raise AnalysisError(
                    f"ô {spec} có hai lượt ở seed {run.seed}: {found[run.seed].run_id}, {run.run_id}"
                )
            found[run.seed] = run
    missing = [s for s in seeds if s not in found]
    if missing:
        raise AnalysisError(f"thiếu ô {spec} ở seed {missing} (tập seed khai báo {list(seeds)})")
    return found


@dataclass(frozen=True)
class Outcome:
    name: str
    statistic: str
    seeds: tuple[int, ...]
    values: tuple[float, ...]
    verdict: str
    detail: str
    run_ids: tuple[str, ...]

    def lines(self) -> list[str]:
        per_seed = ", ".join(f"s{s}: {v:+.4g}" for s, v in zip(self.seeds, self.values, strict=True))
        return [
            f"{self.name}: {self.verdict} — {self.recommendation}",
            f"  {self.statistic}",
            f"  theo seed: {per_seed}" if self.values else "  theo seed: (không tính được)",
            f"  {self.detail}; {len(self.run_ids)} run ID",
        ]

    @property
    def recommendation(self) -> str:
        return stats.recommendation(self.verdict)


def _used(cells: Sequence[dict[int, Run]]) -> list[Run]:
    return [run for c in cells for run in c.values()]


def _guard(runs: Sequence[Run], *, axes: Sequence[str] = ()) -> int:
    """INT-2 trên các lượt của một phép so; trả về số lượt phân kỳ."""
    checks.single_code_version(runs)
    checks.check_clean(runs, draft=False)
    checks.check_fixed(runs, axes=axes)
    return sum(run.status == "diverged" for run in runs)


def _paired(
    name: str,
    statistic: str,
    cells: Sequence[dict[int, Run]],
    seeds: Sequence[int],
    value: Callable[[int], float],
    *,
    expected: int,
    axes: Sequence[str] = (),
) -> Outcome:
    """Đại lượng `value(seed)` trên các ô `cells`, quyết định bằng `stats.decide`."""
    runs = _used(cells)
    diverged = _guard(runs, axes=axes)
    ids = tuple(sorted(run.run_id for run in runs))
    if diverged:
        detail = f"{diverged}/{len(runs)} lượt phân kỳ"
        return Outcome(name, statistic, tuple(seeds), (), stats.OPEN, detail, ids)
    values = tuple(value(s) for s in seeds)
    estimate = stats.mean_ci(values)
    verdict = stats.decide(values, expected=expected)
    interval = f"[CI 95% {estimate.low:+.4g}, {estimate.high:+.4g}]"
    detail = f"trung bình {estimate.mean:+.4g} {interval}, {len(seeds)} seed"
    return Outcome(name, statistic, tuple(seeds), values, verdict, detail, ids)


def _slope(points: Sequence[tuple[float, float]]) -> float:
    return stats.h2_slopes({0: points})[0]


def h1(runs: Sequence[Run]) -> Outcome:
    ladders = {kind: [cell(runs, Cell(kind, lv), SEEDS5) for lv in DIRICHLET_LADDER] for kind in (RING, FC)}

    def value(s: int) -> float:
        slopes = [
            _slope(
                [(js_mean(c[s].manifest["node_class_counts"]), final_accuracy(c[s])) for c in ladders[kind]]
            )
            for kind in (RING, FC)
        ]
        return sum(slopes) / len(slopes)

    statistic = (
        "b_s = trung bình (ring, FC) của độ dốc OLS accuracy cuối theo JS, thang Dirichlet + IID; kỳ vọng < 0"
    )
    return _paired("H1", statistic, [c for k in ladders.values() for c in k], SEEDS5, value, expected=-1)


def h2(runs: Sequence[Run]) -> Outcome:
    fc = [cell(runs, Cell(FC, lv), SEEDS5) for lv in DIRICHLET_LADDER]
    ring = [cell(runs, Cell(RING, lv), SEEDS5) for lv in DIRICHLET_LADDER]

    def value(s: int) -> float:
        return _slope(
            [
                (js_mean(f[s].manifest["node_class_counts"]), final_accuracy(f[s]) - final_accuracy(r[s]))
                for f, r in zip(fc, ring, strict=True)
            ]
        )

    statistic = "b_s = độ dốc OLS của Acc_FC − Acc_ring theo JS, thang Dirichlet + IID (ADR-007); kỳ vọng > 0"
    return _paired("H2", statistic, fc + ring, SEEDS5, value, expected=+1)


def h3(runs: Sequence[Run]) -> Outcome:
    kinds = (RING, ER, GOSSIP, FC)
    iid = {k: cell(runs, Cell(k, IID), SEEDS5) for k in kinds}
    skewed = {k: cell(runs, Cell(k, A01), SEEDS5) for k in kinds}

    def value(s: int) -> float:
        return min(math.log(mean_drift(skewed[k][s]) / mean_drift(iid[k][s])) for k in kinds)

    statistic = "m_s = min theo topology (T1–T4) của ln(drift α = 0.1 / drift IID); kỳ vọng > 0"
    return _paired("H3", statistic, [*iid.values(), *skewed.values()], SEEDS5, value, expected=+1)


@dataclass(frozen=True)
class Cost:
    """bytes@x của một lượt: `bytes` là giá trị khi đạt, cận trên khi "≤ 100", +∞ khi không đạt."""

    reach: Reach
    bytes: float

    @property
    def censored(self) -> str:
        if self.reach.label == "≤ 100":
            return "low"
        return "" if self.reach.reached else "high"


def cost(run: Run, x: float = 0.95) -> Cost:
    reached = run_reach(run, x)
    steps = [float(r["local_steps"]) for r in run.rounds]
    return Cost(reached, bytes_at(steps, [float(r["bytes_cum"]) for r in run.rounds], reached))


def compare_cost(a: Cost, b: Cost) -> int:
    """−1 khi a chắc chắn tốn ít byte hơn b, +1 khi chắc chắn nhiều hơn, 0 khi hòa hay không xác định.

    "≤ 100" là cận trên (byte ở mốc 100 bước): nó chỉ nhỏ hơn một giá trị đạt được lớn hơn cận đó.
    """
    if a.censored == "high" or b.censored == "high":
        return 0 if a.censored == b.censored else (1 if a.censored == "high" else -1)
    if a.censored == "low" and b.censored == "low":
        return 0
    if a.censored == "low":
        return -1 if b.bytes > a.bytes else 0
    if b.censored == "low":
        return 1 if a.bytes > b.bytes else 0
    return -1 if a.bytes < b.bytes else 1 if a.bytes > b.bytes else 0


def h4(runs: Sequence[Run]) -> Outcome:
    ring, fc = cell(runs, Cell(RING, A01), SEEDS5), cell(runs, Cell(FC, A01), SEEDS5)
    used = _used([ring, fc])
    diverged = _guard(used)
    ids = tuple(sorted(run.run_id for run in used))
    statistic = (
        "thứ tự bytes@95 ở α = 0.1, ring so với FC (kiểm định dấu, kiểm duyệt hai phía); kỳ vọng ring rẻ hơn"
    )
    if diverged:
        return Outcome("H4", statistic, SEEDS5, (), stats.OPEN, f"{diverged}/{len(used)} lượt phân kỳ", ids)
    orders = [compare_cost(cost(ring[s]), cost(fc[s])) for s in SEEDS5]
    count = f"{sum(o < 0 for o in orders)}/{len(orders)} cặp đúng chiều"
    if all(o < 0 for o in orders):
        verdict = stats.SUPPORTED
    elif all(o > 0 for o in orders):
        verdict = stats.REJECTED
    else:
        verdict = stats.OPEN
    medians = ", ".join(
        f"{name} trung vị {median([cost(c[s]).bytes for s in SEEDS5]) / 1e6:.1f} MB"
        for name, c in (("ring", ring), ("FC", fc))
    )
    return Outcome(
        "H4", statistic, SEEDS5, tuple(float(o) for o in orders), verdict, f"{count}; {medians}", ids
    )


def _mean_over(kinds: Sequence[str], value: Callable[[str], float]) -> float:
    return sum(value(k) for k in kinds) / len(kinds)


def h5(runs: Sequence[Run]) -> Outcome:
    prox = {k: cell(runs, Cell(k, A01, mu=0.5), SEEDS3) for k in (RING, FC)}
    avg = {k: cell(runs, Cell(k, A01), SEEDS3) for k in (RING, FC)}

    def value(s: int) -> float:
        return _mean_over((RING, FC), lambda k: final_accuracy(prox[k][s]) - final_accuracy(avg[k][s]))

    statistic = (
        "d_s = trung bình (ring, FC) của Acc(DFedProx μ = 0.5) − Acc(DFedAvg), α = 0.1, τ = 20; kỳ vọng > 0"
    )
    return _paired("H5", statistic, [*prox.values(), *avg.values()], SEEDS3, value, expected=+1)


def h6(runs: Sequence[Run]) -> Outcome:
    server = cell(runs, Cell("server", A01, mixing="size_weighted"), SEEDS3)
    fc = cell(runs, Cell(FC, A01), SEEDS3)
    statistic = "e_s = Acc(B3, trọng số n_k/n) − Acc(FC + MH, 1/N), α = 0.1; kỳ vọng > 0"
    return _paired(
        "H6",
        statistic,
        [server, fc],
        SEEDS3,
        lambda s: final_accuracy(server[s]) - final_accuracy(fc[s]),
        expected=+1,
    )


def h7(runs: Sequence[Run]) -> Outcome:
    fast = {k: cell(runs, Cell(k, A01, tau=5), SEEDS3) for k in (RING, FC)}
    slow = {k: cell(runs, Cell(k, A01, tau=50), SEEDS3) for k in (RING, FC)}

    def value(s: int) -> float:
        return _mean_over((RING, FC), lambda k: final_accuracy(fast[k][s]) - final_accuracy(slow[k][s]))

    statistic = "h_s = trung bình (ring, FC) của Acc(τ = 5) − Acc(τ = 50), μ = 0, α = 0.1; kỳ vọng > 0"
    return _paired("H7", statistic, [*fast.values(), *slow.values()], SEEDS3, value, expected=+1)


def comm_per_round(run: Run) -> float:
    """Trung vị qua các dòng timing.jsonl của `comm_s` chia số round kể từ dòng trước (ADR-011)."""
    if not run.timing:
        raise AnalysisError(f"lượt {run.run_id} không có timing.jsonl")
    previous, values = 0, []
    for record in run.timing:
        values.append(float(record["comm_s"]) / (int(record["round"]) - previous))
        previous = int(record["round"])
    return median(values)


E8_COMBOS = tuple((kind, lv) for kind in (RING, GOSSIP, FC) for lv in (IID, A01))


def h8(runs: Sequence[Run]) -> Outcome:
    cells = {
        (kind, lv, model): cell(runs, Cell(kind, lv, model=model), SEEDS3)
        for kind, lv in E8_COMBOS
        for model in PARAMS
    }

    def value(s: int) -> float:
        slopes = [
            _slope(
                [(math.log(p), math.log(comm_per_round(cells[(kind, lv, m)][s]))) for m, p in PARAMS.items()]
            )
            for kind, lv in E8_COMBOS
        ]
        return sum(slopes) / len(slopes)

    statistic = (
        "c_s = trung bình 6 tổ hợp (topology, mức) của độ dốc ln(comm mỗi round) "
        "theo ln(số tham số); kỳ vọng > 0"
    )
    return _paired("H8", statistic, list(cells.values()), SEEDS3, value, expected=+1, axes=("model",))


def e8_checks(runs: Sequence[Run]) -> list[str]:
    """Kiểm tất định của thẻ E8: byte = message × 4 × số tham số, và tải theo node khớp số tổng (ADR-013)."""
    problems = []
    for run in runs:
        params = PARAMS[run.config["model"]]
        for record in run.rounds:
            if int(record["bytes_cum"]) != int(record["messages_cum"]) * 4 * params:
                problems.append(
                    f"{run.run_id}, round {record['round']}: bytes_cum {record['bytes_cum']} khác "
                    f"message × 4 × số tham số = {int(record['messages_cum']) * 4 * params}"
                )
                break
            if "node_bytes_cum" in record and (
                max(record["node_bytes_cum"]) != record["max_node_bytes_cum"]
                or sum(record["node_messages_cum"]) != 2 * record["messages_cum"]
            ):
                problems.append(f"{run.run_id}, round {record['round']}: tải theo node không khớp số tổng")
                break
    return problems


def rq4(runs: Sequence[Run]) -> Outcome:
    mh = cell(runs, Cell(ER, A01), SEEDS3)
    uniform = cell(runs, Cell(ER, A01, mixing="uniform"), SEEDS3)
    statistic = (
        "u_s = Acc(ER + MH) − Acc(ER + uniform), α = 0.1 (khuyến nghị trọng số trộn, thẻ E4); kỳ vọng > 0"
    )
    return _paired(
        "RQ4",
        statistic,
        [mh, uniform],
        SEEDS3,
        lambda s: final_accuracy(mh[s]) - final_accuracy(uniform[s]),
        expected=+1,
    )


COMPARISONS: dict[str, Callable[[Sequence[Run]], Outcome]] = {
    "H1": h1,
    "H2": h2,
    "H3": h3,
    "H4": h4,
    "H5": h5,
    "H6": h6,
    "H7": h7,
    "H8": h8,
    "RQ4": rq4,
}


def load(results: Path, groups: Sequence[str] = GROUPS) -> list[Run]:
    """Lượt đã xong của các nhóm có chỉ mục; một run_id dùng chung giữa các nhóm chỉ lấy một lần."""
    unique: dict[str, Run] = {}
    for name in groups:
        if (Path(results) / "experiments" / f"{name}.json").exists():
            for attempts in read_experiment(results, name):
                if attempts.finished is not None:
                    unique[attempts.run_id] = attempts.finished
    return list(unique.values())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m analysis.hypotheses",
        description="Kết luận H1–H8 và RQ4 theo thẻ thí nghiệm (ADR-007).",
    )
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--only", nargs="*", default=list(COMPARISONS), choices=list(COMPARISONS))
    args = parser.parse_args(argv)
    runs = load(args.results)
    report: list[str] = []
    for name in args.only:
        try:
            report += COMPARISONS[name](runs).lines()
        except AnalysisError as err:
            report.append(f"{name}: chưa đủ dữ liệu ({err})")
    if "H8" in args.only:
        problems = e8_checks(load(args.results, ["e8_model_complexity"]))
        report.append(
            "E8 kiểm tất định: " + ("đạt" if not problems else f"{len(problems)} lượt sai: {problems[:3]}")
        )
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AnalysisError as err:
        print(f"lỗi: {err}")
        raise SystemExit(1) from None

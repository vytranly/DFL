"""Quy tắc chọn η của ADR-007 trên kết quả E0a, chỉ từ validation (module analysis; T14e; G8, G25; SPEC
QS-2).

- "Accuracy validation" của một lượt là trung bình `acc_mean` ở 5 mốc cuối trên tập con validation; lượt phân
  kỳ tính là 0. "AUC validation" là AUC của đường `acc_mean` (analysis.convergence); lượt phân kỳ có AUC 0.
- **η chung** (τ = 20, ô FC-IID, ring-α0.1, ring-#C2): η có accuracy validation nhỏ nhất qua các ô là lớn
  nhất. Các η cách giá trị tốt nhất không quá 0.2 điểm % là hòa; khi hòa, chọn η có AUC validation lớn nhất ở
  ring-α0.1 (hòa cả AUC thì lấy η nhỏ hơn, để kết quả tất định).
- **τ = 1 và τ = 50** (ô FC-IID, ring-α0.1): η tốt nhất theo đúng quy tắc trên; nếu η chung cách giá trị tốt
  nhất không quá 0.2 điểm % thì η chung được coi là tốt nhất. η tốt nhất khác η chung thì kích hoạt 12 lượt
  bổ sung.
- **η riêng của B1:** accuracy validation lớn nhất trong 3 lượt B1. ADR-007 không định nghĩa "hòa" cho B1,
  nên dùng cùng ngưỡng 0.2 điểm % như η chung, và hòa thì lấy AUC lớn hơn.
- **μ của DFedProx** (QS-2): giữ cường độ η·μ·τ ∈ {0.1, 1} ở τ = 20, nên μ = 0.1/(20η) và 1/(20η).
- Lượt có `final_eval_split` khác validation, hay cấu hình đánh giá trên test, bị từ chối: η không bao giờ
  được chọn bằng kết quả trên tập test.

Dòng lệnh: `python -m analysis.select_lr --results results/ --experiment e0a_lr`.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from analysis import checks, convergence
from analysis.aggregate import AnalysisError, Run, read_experiment

TIE = 0.002  # 0.2 điểm %
LAST_MARKS = 5
QS2_STRENGTHS = {"weak": 0.1, "strong": 1.0}
QS2_TAU = 20
COMMON_TAU = 20


def _kind(run: Run) -> str:
    return run.config["topology"]["kind"]


def _partition(run: Run) -> tuple[str, object]:
    p = run.config["partition"]
    return p["scheme"], p["alpha"] if p["scheme"] == "dirichlet" else p["k"]


# Ô của E0a, nhận diện từ cấu hình trong manifest (ADR-007).
CELLS: dict[str, Callable[[Run], bool]] = {
    "FC-IID": lambda r: _kind(r) == "fully_connected" and _partition(r)[0] == "iid",
    "ring-α0.1": lambda r: _kind(r) == "ring" and _partition(r) == ("dirichlet", 0.1),
    "ring-#C2": lambda r: _kind(r) == "ring" and _partition(r) == ("label_k", 2),
}
AUC_CELL = "ring-α0.1"


def _check_validation(run: Run) -> None:
    split = run.config["eval"]["split"]
    final = run.summary.get("final_eval_split", split)
    if split != "validation" or final != "validation":
        raise AnalysisError(
            f"lượt {run.run_id} đánh giá trên {final}; η chỉ được chọn trên validation (G8, G25, ADR-007)"
        )


def validation_accuracy(run: Run) -> float:
    """Trung bình acc_mean ở 5 mốc cuối; 0 với lượt phân kỳ."""
    _check_validation(run)
    if run.status == "diverged":
        return 0.0
    convergence.check_grid(run)
    tail = run.rounds[-LAST_MARKS:]
    return sum(float(r["acc_mean"]) for r in tail) / len(tail)


def validation_auc(run: Run) -> float:
    _check_validation(run)
    value = convergence.run_auc(run)
    return 0.0 if value is None else value


@dataclass(frozen=True)
class Choice:
    eta: float
    # η → accuracy validation nhỏ nhất qua các ô.
    worst: dict[float, float]
    tie: bool
    extra_runs: bool = False
    candidates: tuple[float, ...] = field(default=())


def _table(runs: Sequence[Run], cells: Sequence[str], tau: int) -> dict[float, dict[str, Run]]:
    selected = [r for r in runs if r.config["algorithm"]["tau"] == tau and r.config["algorithm"]["mu"] == 0]
    etas = sorted({r.config["algorithm"]["lr"] for r in selected})
    table: dict[float, dict[str, Run]] = {}
    for eta in etas:
        row = {}
        for cell in cells:
            matches = [r for r in selected if r.config["algorithm"]["lr"] == eta and CELLS[cell](r)]
            if len(matches) != 1:
                raise AnalysisError(
                    f"thiếu lượt (hay có {len(matches)} lượt) cho η = {eta}, ô {cell}, τ = {tau}"
                )
            row[cell] = matches[0]
        table[eta] = row
    if not table:
        raise AnalysisError(f"thiếu lượt E0a với τ = {tau}")
    return table


def _choose(table: dict[float, dict[str, Run]]) -> Choice:
    worst = {eta: min(validation_accuracy(run) for run in row.values()) for eta, row in table.items()}
    best = max(worst.values())
    candidates = tuple(eta for eta, value in worst.items() if value >= best - TIE)
    if len(candidates) == 1:
        return Choice(candidates[0], worst, tie=False, candidates=candidates)
    aucs = {eta: validation_auc(table[eta][AUC_CELL]) for eta in candidates}
    top = max(aucs.values())
    return Choice(min(eta for eta in candidates if aucs[eta] == top), worst, tie=True, candidates=candidates)


def common_eta(runs: Sequence[Run]) -> Choice:
    """η chung, chọn ở τ = 20 trên ba ô của E0a."""
    return _choose(_table(runs, list(CELLS), COMMON_TAU))


def eta_for_tau(runs: Sequence[Run], *, common: float, tau: int | None = None) -> Choice:
    """η tốt nhất cho τ = 1 hay τ = 50 trên hai ô; η chung trong 0.2 điểm % thì giữ η chung."""
    taus = sorted({r.config["algorithm"]["tau"] for r in runs} - {COMMON_TAU})
    tau = tau if tau is not None else (taus[0] if len(taus) == 1 else None)
    if tau is None:
        raise AnalysisError(f"cần chỉ rõ τ trong {taus}")
    choice = _choose(_table(runs, ["FC-IID", AUC_CELL], tau))
    if common in choice.worst and choice.worst[common] >= max(choice.worst.values()) - TIE:
        return Choice(common, choice.worst, tie=choice.tie, extra_runs=False, candidates=choice.candidates)
    return Choice(choice.eta, choice.worst, tie=choice.tie, extra_runs=True, candidates=choice.candidates)


def b1_eta(runs: Sequence[Run]) -> float:
    """η riêng của B1: accuracy validation lớn nhất trong các lượt B1; hòa thì AUC lớn hơn."""
    b1 = [r for r in runs if _kind(r) == "central" and r.config["local_steps"] != 4_000]
    if not b1:
        raise AnalysisError("thiếu lượt B1 của E0a")
    accuracy = {r.config["algorithm"]["lr"]: validation_accuracy(r) for r in b1}
    best = max(accuracy.values())
    candidates = [r for r in b1 if accuracy[r.config["algorithm"]["lr"]] >= best - TIE]
    top = max(validation_auc(r) for r in candidates)
    return min(r.config["algorithm"]["lr"] for r in candidates if validation_auc(r) == top)


def mu_for(eta: float) -> dict[str, float]:
    """μ giữ η·μ·τ ∈ {0.1, 1} ở τ = 20 (QS-2)."""
    return {name: strength / (eta * QS2_TAU) for name, strength in QS2_STRENGTHS.items()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m analysis.select_lr", description="Chọn η theo ADR-007 (E0a)."
    )
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--experiment", default="e0a_lr")
    args = parser.parse_args(argv)
    runs = [a.finished for a in read_experiment(args.results, args.experiment) if a.finished is not None]
    checks.single_code_version(runs)
    checks.check_clean(runs, draft=False)
    common = common_eta(runs)
    print(f"η chung: {common.eta} (trường hợp xấu nhất theo η: {common.worst}; hòa: {common.tie})")
    for tau in sorted({r.config["algorithm"]["tau"] for r in runs if _kind(r) != "central"} - {COMMON_TAU}):
        choice = eta_for_tau(
            [r for r in runs if r.config["algorithm"]["tau"] == tau], common=common.eta, tau=tau
        )
        print(
            f"τ = {tau}: η tốt nhất {choice.eta}; chạy thêm 12 lượt: {'có' if choice.extra_runs else 'không'}"
        )
    print(f"η riêng của B1: {b1_eta(runs)}")
    mu = mu_for(common.eta)
    print(f"μ của DFedProx (QS-2): {mu['weak']:.4g} (η·μ·τ = 0.1) và {mu['strong']:.4g} (η·μ·τ = 1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

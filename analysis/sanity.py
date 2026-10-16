"""Lệnh kiểm sanity SAN-1…5 trên results/ (module analysis; T14a; SPEC §9.3; CONSTRAINTS R7).

- Mỗi SAN trả `Verdict` với một trong ba trạng thái: "đạt", "không đạt", "thiếu lượt". Thiếu lượt (ô chưa
  chạy, thiếu seed, lượt chưa có attempt xong) không bao giờ được tính là đạt.
- Ngưỡng là của CONSTRAINTS.md R7 (nhóm chốt ở P1.8, P1.9); đổi chúng phải có quyết định của nhóm, và không
  bao giờ đổi vì kết quả trên tập test.
- Lượt được nhận diện từ cấu hình trong manifest: B1 là `central` với hơn 4,000 bước, B1-4k là `central` với
  đúng 4,000 bước, B2 là `local_only`, DFedAvg-FC-IID là FC + IID + MH với τ = 20, μ = 0.
- Accuracy của một lượt là trung bình `final_acc_per_node` trên đủ tập test (với FC mọi node bằng nhau; với
  B1 chỉ có một node). Lượt phân kỳ không có accuracy, nên SAN chứa nó là "không đạt".
- SAN-5 là điều kiện cấu trúc trên `rounds.jsonl`, kiểm được trên lượt của bất kỳ nhóm nào.

Dòng lệnh, ví dụ: `python -m analysis.sanity --results results/ --e0b e0b_sanity --e1 e1_noniid`, hay
`--san5 e2_topology`. Mã thoát 0 khi mọi SAN được yêu cầu đều đạt, 1 khi có SAN không đạt hay thiếu lượt.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from analysis import checks
from analysis.aggregate import AnalysisError, Run, read_experiment

# CONSTRAINTS.md R7.
SAN1_MIN = 0.985
SAN2_MAX_GAP = 0.010
SAN3_MIN_GAP = 0.20
SAN4_MAX_GAP = 0.010
SAN5_FC_RATIO = 1e-6
SAN5_REL_TOL = 1e-9
BASELINE_SEEDS = (0, 1, 2)
GRID_BUDGET = 4_000
PASSED, FAILED, MISSING = "đạt", "không đạt", "thiếu lượt"


@dataclass(frozen=True)
class Verdict:
    san: str
    status: str
    detail: str

    @property
    def passed(self) -> bool:
        return self.status == PASSED


def _kind(run: Run) -> str:
    return run.config["topology"]["kind"]


def _alpha(run: Run) -> float | None:
    partition = run.config["partition"]
    return partition["alpha"] if partition["scheme"] == "dirichlet" else None


def _is_iid(run: Run) -> bool:
    return run.config["partition"]["scheme"] == "iid"


def _is_b1(run: Run) -> bool:
    return _kind(run) == "central" and run.config["local_steps"] > GRID_BUDGET


def _is_b1_4k(run: Run) -> bool:
    return _kind(run) == "central" and run.config["local_steps"] == GRID_BUDGET


def _is_dfedavg(run: Run) -> bool:
    algorithm = run.config["algorithm"]
    return run.config["mixing"] == "metropolis_hastings" and algorithm["tau"] == 20 and algorithm["mu"] == 0


def _accuracy(run: Run) -> float | None:
    """Accuracy cuối trên đủ tập test; None với lượt phân kỳ."""
    if run.status != "ok":
        return None
    if run.summary["final_eval_split"] != "test":
        raise AnalysisError(
            f"lượt {run.run_id} đánh giá cuối trên {run.summary['final_eval_split']}; SAN đo trên test"
        )
    values = run.summary["final_acc_per_node"]
    return sum(values) / len(values)


def _cell(runs: Sequence[Run], keep: Callable[[Run], bool], seeds: Sequence[int]) -> list[Run] | None:
    """Các lượt của ô theo đúng `seeds`; None nếu thiếu seed nào."""
    by_seed = {run.seed: run for run in runs if keep(run)}
    if any(seed not in by_seed for seed in seeds):
        return None
    return [by_seed[seed] for seed in seeds]


def _mean(cell: Sequence[Run]) -> float | None:
    values = [_accuracy(run) for run in cell]
    return None if any(v is None for v in values) else sum(values) / len(values)


def _missing(san: str, what: str, seeds: Sequence[int]) -> Verdict:
    return Verdict(san, MISSING, f"thiếu lượt {what} ở seed {list(seeds)}")


def _diverged(san: str, what: str) -> Verdict:
    return Verdict(san, FAILED, f"{what} có lượt phân kỳ, không có accuracy cuối")


def san1(runs: Sequence[Run], seeds: Sequence[int] = BASELINE_SEEDS) -> Verdict:
    """SAN-1: B1 (η riêng) đạt ≥ 98.5% trên đủ tập test, trung bình qua seed."""
    cell = _cell(runs, _is_b1, seeds)
    if cell is None:
        return _missing("SAN-1", "B1", seeds)
    mean = _mean(cell)
    if mean is None:
        return _diverged("SAN-1", "B1")
    status = PASSED if mean >= SAN1_MIN else FAILED
    return Verdict("SAN-1", status, f"B1 trung bình {mean:.4f}, ngưỡng ≥ {SAN1_MIN}")


def san2(runs: Sequence[Run], seeds: Sequence[int] = BASELINE_SEEDS) -> Verdict:
    """SAN-2: DFedAvg-FC-IID kém B1-4k không quá 1.0 điểm % (trung bình seed 0, 1, 2)."""
    fc = _cell(runs, lambda r: _kind(r) == "fully_connected" and _is_iid(r) and _is_dfedavg(r), seeds)
    b14k = _cell(runs, _is_b1_4k, seeds)
    if fc is None or b14k is None:
        return _missing("SAN-2", "DFedAvg-FC-IID" if fc is None else "B1-4k", seeds)
    fc_mean, b_mean = _mean(fc), _mean(b14k)
    if fc_mean is None or b_mean is None:
        return _diverged("SAN-2", "DFedAvg-FC-IID hay B1-4k")
    gap = b_mean - fc_mean
    status = PASSED if gap <= SAN2_MAX_GAP else FAILED
    return Verdict(
        "SAN-2", status, f"B1-4k {b_mean:.4f} − FC-IID {fc_mean:.4f} = {gap:.4f}, ngưỡng ≤ {SAN2_MAX_GAP}"
    )


def san3(runs: Sequence[Run], seeds: Sequence[int] = BASELINE_SEEDS) -> Verdict:
    """SAN-3: B2 ở α = 0.1 thấp hơn B2 ở IID ít nhất 20 điểm %."""
    iid = _cell(runs, lambda r: _kind(r) == "local_only" and _is_iid(r), seeds)
    skew = _cell(runs, lambda r: _kind(r) == "local_only" and _alpha(r) == 0.1, seeds)
    if iid is None or skew is None:
        return _missing("SAN-3", "B2-IID" if iid is None else "B2 α = 0.1", seeds)
    iid_mean, skew_mean = _mean(iid), _mean(skew)
    if iid_mean is None or skew_mean is None:
        return _diverged("SAN-3", "B2")
    gap = iid_mean - skew_mean
    status = PASSED if gap >= SAN3_MIN_GAP else FAILED
    return Verdict(
        "SAN-3", status, f"B2 IID {iid_mean:.4f} − α 0.1 {skew_mean:.4f} = {gap:.4f}, ngưỡng ≥ {SAN3_MIN_GAP}"
    )


def san4(runs: Sequence[Run]) -> Verdict:
    """SAN-4: trên cùng topology, accuracy ở α = 100 và ở IID chênh không quá 1.0 điểm % (cùng tập seed)."""
    topologies = sorted({_kind(r) for r in runs if _alpha(r) == 100.0 and _is_dfedavg(r)})
    if not topologies:
        return Verdict("SAN-4", MISSING, "thiếu lượt α = 100")
    lines, status = [], PASSED
    for kind in topologies:
        seeds = sorted(r.seed for r in runs if _kind(r) == kind and _alpha(r) == 100.0 and _is_dfedavg(r))
        a100 = _cell(runs, lambda r, k=kind: _kind(r) == k and _alpha(r) == 100.0 and _is_dfedavg(r), seeds)
        iid = _cell(runs, lambda r, k=kind: _kind(r) == k and _is_iid(r) and _is_dfedavg(r), seeds)
        if iid is None:
            return _missing("SAN-4", f"IID của {kind}", seeds)
        a_mean, i_mean = _mean(a100), _mean(iid)
        if a_mean is None or i_mean is None:
            return _diverged("SAN-4", kind)
        gap = abs(a_mean - i_mean)
        if gap > SAN4_MAX_GAP:
            status = FAILED
        lines.append(f"{kind}: |α100 {a_mean:.4f} − IID {i_mean:.4f}| = {gap:.4f}")
    return Verdict("SAN-4", status, "; ".join(lines) + f"; ngưỡng ≤ {SAN4_MAX_GAP}")


def _structure_problems(run: Run) -> list[str]:
    """Các điều kiện cấu trúc của SAN-5 mà `run` vi phạm (SPEC §9.3)."""
    kind, problems = _kind(run), []
    baseline = kind in ("central", "local_only")
    mh = run.config["mixing"] == "metropolis_hastings"
    for record in run.rounds:
        drift, consensus = record["drift_pre_combine"], record["consensus_distance"]
        if (kind == "fully_connected" and mh) or kind == "server":
            if consensus > SAN5_FC_RATIO * drift:
                problems.append(f"round {record['round']}: consensus {consensus} > 1e-6 × drift {drift}")
            if record["acc_std"] != 0:
                problems.append(
                    f"round {record['round']}: acc_std {record['acc_std']} ≠ 0 dù mọi node chung model"
                )
        if mh and kind != "server" and consensus > drift * (1 + SAN5_REL_TOL):
            problems.append(f"round {record['round']}: consensus {consensus} > drift {drift} với W đối xứng")
    if run.rounds:
        messages = run.rounds[0]["messages_cum"]
        if baseline and messages != 0:
            problems.append(f"baseline {kind} có {messages} message")
        if not baseline and messages <= 0:
            problems.append("lượt DFL không có message nào ở mốc đầu")
    return problems


def san5(runs: Sequence[Run]) -> Verdict:
    """SAN-5: bước tổng hợp có tác dụng thật, trên mọi lượt được đưa vào."""
    if not runs:
        return Verdict("SAN-5", MISSING, "không có lượt nào")
    problems = [f"{run.run_id}: {p}" for run in runs for p in _structure_problems(run)]
    if problems:
        return Verdict("SAN-5", FAILED, "; ".join(problems[:10]) + (" …" if len(problems) > 10 else ""))
    return Verdict("SAN-5", PASSED, f"{len(runs)} lượt đạt mọi điều kiện cấu trúc")


def _finished(results: Path, experiment: str) -> list[Run]:
    runs = [a.finished for a in read_experiment(results, experiment) if a.finished is not None]
    checks.single_code_version(runs)
    checks.check_clean(runs, draft=False)
    return runs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m analysis.sanity", description="Kiểm SAN-1…5 (R7).")
    parser.add_argument("--results", type=Path, default=Path("results"), help="thư mục kết quả")
    parser.add_argument("--e0b", help="nhóm E0b: SAN-1, SAN-2, SAN-3 và SAN-5")
    parser.add_argument("--e1", help="nhóm E1: SAN-4 (ô IID lấy từ --e0b) và SAN-5")
    parser.add_argument("--san5", nargs="*", default=[], help="nhóm khác chỉ kiểm SAN-5")
    args = parser.parse_args(argv)
    verdicts: list[Verdict] = []
    e0b = _finished(args.results, args.e0b) if args.e0b else []
    if args.e0b:
        verdicts += [san1(e0b), san2(e0b), san3(e0b), san5(e0b)]
    if args.e1:
        e1 = _finished(args.results, args.e1)
        verdicts += [san4(e1 + e0b), san5(e1)]
    for name in args.san5:
        verdicts.append(san5(_finished(args.results, name)))
    for verdict in verdicts:
        print(f"{verdict.san}: {verdict.status} ({verdict.detail})")
    return 0 if verdicts and all(v.passed for v in verdicts) else 1


if __name__ == "__main__":
    raise SystemExit(main())

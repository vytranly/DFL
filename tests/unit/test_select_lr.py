"""Test cho analysis/select_lr.py: quy tắc chọn η của ADR-007 trên E0a, chỉ từ validation (T14e; G8, G25;
SPEC QS-2).
"""

from pathlib import Path

import pytest

from analysis import aggregate, select_lr
from analysis.aggregate import AnalysisError
from tests.analysis_fakes import CODE_VERSION, config, curve, write_index, write_run

CELLS = {
    "fc-iid": {"topology__kind": "fully_connected"},
    "ring-a01": {"partition__scheme": "dirichlet", "partition__alpha": 0.1},
    "ring-c2": {"partition__scheme": "label_k", "partition__k": 2, "num_nodes": 10},
}
ETAS = (0.01, 0.05, 0.1)


def e0a(
    tmp_path: Path,
    accs: dict[tuple[float, str], list[float]],
    *,
    tau: int = 20,
    split: str = "validation",
    status: dict[tuple[float, str], str] | None = None,
) -> list[aggregate.Run]:
    """Lượt E0a giả: (η, ô) → chuỗi acc_mean ở 5 mốc (100 … 500 bước), đánh giá trên `split`."""
    ids = []
    for (eta, name), values in accs.items():
        run_id = f"{name}-{eta}-t{tau}"
        n = 10 if name == "ring-c2" else 4
        overrides = {
            **CELLS.get(name, {}),
            "algorithm__lr": eta,
            "algorithm__tau": tau,
            "local_steps": 500,
            "eval__split": split,
            "seed": 100,
        }
        if name == "b1":
            overrides = {
                "num_nodes": 1,
                "topology__kind": "central",
                "algorithm__lr": eta,
                "local_steps": 500,
                "eval__split": split,
                "seed": 100,
            }
            n = 1
        write_run(
            tmp_path,
            run_id,
            cfg=config(**overrides),
            rounds=curve(values, tau=tau, nodes=n),
            status=(status or {}).get((eta, name), "ok"),
        )
        ids.append(run_id)
    write_index(tmp_path, "e0a", {CODE_VERSION: ids})
    return [a.finished for a in aggregate.read_experiment(tmp_path, "e0a") if a.finished is not None]


def flat(value: float) -> list[float]:
    return [value] * 5


def test_common_eta_maximizes_the_worst_cell_on_validation(tmp_path: Path) -> None:
    accs = {
        (0.01, "fc-iid"): flat(0.95),
        (0.01, "ring-a01"): flat(0.80),
        (0.01, "ring-c2"): flat(0.70),
        (0.05, "fc-iid"): flat(0.97),
        (0.05, "ring-a01"): flat(0.85),
        (0.05, "ring-c2"): flat(0.78),
        (0.1, "fc-iid"): flat(0.98),
        (0.1, "ring-a01"): flat(0.86),
        (0.1, "ring-c2"): flat(0.60),
    }

    choice = select_lr.common_eta(e0a(tmp_path, accs))

    assert choice.eta == 0.05 and choice.worst[0.05] == pytest.approx(0.78) and not choice.tie


def test_a_tie_within_0_2_points_goes_to_the_larger_auc_on_ring_alpha_0_1(tmp_path: Path) -> None:
    # Trường hợp xấu nhất: η = 0.05 là 0.800, η = 0.1 là 0.799 (hòa trong 0.2 điểm %); AUC ở ring-α0.1 của
    # 0.1 (khoảng 0.849) lớn hơn của 0.05 (khoảng 0.803), nên chọn 0.1 dù accuracy của nó kém hơn một chút.
    accs = {
        (0.01, "fc-iid"): flat(0.90),
        (0.01, "ring-a01"): flat(0.70),
        (0.01, "ring-c2"): flat(0.70),
        (0.05, "fc-iid"): flat(0.97),
        (0.05, "ring-a01"): [0.75, 0.78, 0.81, 0.83, 0.83],
        (0.05, "ring-c2"): flat(0.80),
        (0.1, "fc-iid"): flat(0.97),
        (0.1, "ring-a01"): [0.84, 0.85, 0.85, 0.85, 0.85],
        (0.1, "ring-c2"): flat(0.799),
    }

    choice = select_lr.common_eta(e0a(tmp_path, accs))

    assert choice.tie and choice.eta == 0.1


def test_a_diverged_run_counts_as_accuracy_zero(tmp_path: Path) -> None:
    accs = {(eta, name): flat(0.9 + eta) for eta in ETAS for name in CELLS}

    choice = select_lr.common_eta(e0a(tmp_path, accs, status={(0.1, "ring-c2"): "diverged"}))

    assert choice.worst[0.1] == 0.0 and choice.eta == 0.05


def test_tau_1_or_50_keeps_the_common_eta_when_it_is_within_0_2_points(tmp_path: Path) -> None:
    accs = {(eta, name): flat(0.90) for eta in ETAS for name in ("fc-iid", "ring-a01")}
    accs[(0.1, "ring-a01")] = flat(0.88)
    accs[(0.01, "ring-a01")] = flat(0.9015)  # tốt nhất, nhưng η chung 0.05 chỉ kém 0.15 điểm %

    choice = select_lr.eta_for_tau(e0a(tmp_path, accs, tau=1), common=0.05)

    assert choice.eta == 0.05 and not choice.extra_runs


def test_tau_1_or_50_triggers_the_extra_runs_when_another_eta_is_clearly_better(tmp_path: Path) -> None:
    accs = {(eta, name): flat(0.80) for eta in ETAS for name in ("fc-iid", "ring-a01")}
    accs[(0.1, "fc-iid")], accs[(0.1, "ring-a01")] = flat(0.90), flat(0.85)

    choice = select_lr.eta_for_tau(e0a(tmp_path, accs, tau=50), common=0.05)

    assert choice.eta == 0.1 and choice.extra_runs


def test_b1_gets_its_own_eta_from_the_best_validation_run_with_auc_breaking_ties(tmp_path: Path) -> None:
    # 0.1 có accuracy validation 0.981, 0.05 có 0.980 (hòa trong 0.2 điểm %); AUC của 0.05 (0.9825) lớn hơn.
    accs = {(0.01, "b1"): flat(0.97), (0.05, "b1"): [0.95, 0.98, 0.99, 0.99, 0.99], (0.1, "b1"): flat(0.981)}

    assert select_lr.b1_eta(e0a(tmp_path, accs)) == 0.05


def test_mu_keeps_the_qs_2_strengths_at_tau_20() -> None:
    # η·μ·τ ∈ {0.1, 1} ở τ = 20 (QS-2): với η = 0.05 thì μ ∈ {0.1, 1}, như ADR-006.
    assert select_lr.mu_for(0.05) == pytest.approx({"weak": 0.1, "strong": 1.0})
    assert select_lr.mu_for(0.1) == pytest.approx({"weak": 0.05, "strong": 0.5})


def test_runs_evaluated_on_test_are_refused(tmp_path: Path) -> None:
    # G8, G25: η không bao giờ được chọn bằng kết quả trên tập test.
    accs = {(eta, name): flat(0.9) for eta in ETAS for name in CELLS}

    with pytest.raises(AnalysisError, match="validation"):
        select_lr.common_eta(e0a(tmp_path, accs, split="test"))


def test_a_missing_cell_is_an_error_not_a_skipped_eta(tmp_path: Path) -> None:
    accs = {(eta, name): flat(0.9) for eta in ETAS for name in CELLS}
    del accs[(0.05, "ring-c2")]

    with pytest.raises(AnalysisError, match="thiếu"):
        select_lr.common_eta(e0a(tmp_path, accs))

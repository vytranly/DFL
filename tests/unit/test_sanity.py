"""Test cho analysis/sanity.py: SAN-1…5 (T14a; SPEC §9.3; CONSTRAINTS R7). Mỗi SAN có ba ca: đạt, không đạt,
thiếu lượt; SAN-5 chạy trên lượt của bất kỳ nhóm nào. Ngưỡng lấy từ CONSTRAINTS.md R7, không đổi ở đây.
"""

from pathlib import Path

import pytest

from analysis import aggregate, sanity
from tests.analysis_fakes import CODE_VERSION, config, curve, round_record, write_index, write_run

B1 = {"num_nodes": 1, "local_steps": 17_200, "topology__kind": "central", "algorithm__lr": 0.1}
B1_4K = {"num_nodes": 1, "local_steps": 4_000, "topology__kind": "central"}
FC_IID = {"topology__kind": "fully_connected"}
B2 = {"topology__kind": "local_only"}
ALPHA = {"partition__scheme": "dirichlet", "partition__alpha": 0.1}
ALPHA100 = {"partition__scheme": "dirichlet", "partition__alpha": 100.0}
# FC + MH: mọi node chung một model sau tổng hợp, nên consensus đúng bằng 0 (SAN-5).
FC_ROUNDS = [round_record(s, 20, [0.8] * 4, consensus_distance=0.0) for s in (100, 200)]


def cell(
    tmp_path: Path, name: str, overrides: dict[str, object], accs: list[float], **kwargs: object
) -> list[str]:
    """Ghi một ô (một lượt mỗi seed 0, 1, …) với accuracy cuối `accs`; trả các run_id."""
    ids = []
    for seed, acc in enumerate(accs):
        run_id = f"{name}-s{seed}"
        n = int(overrides.get("num_nodes", 4))
        rounds = kwargs.get("rounds") or curve([0.5, 0.8], nodes=n, tau=20)
        if overrides.get("topology__kind") in ("central", "local_only"):
            rounds = [dict(r, messages_cum=0, bytes_cum=0, max_node_bytes_cum=0) for r in rounds]
        write_run(tmp_path, run_id, cfg=config(seed=seed, **overrides), final_acc=acc, rounds=rounds)
        ids.append(run_id)
    return ids


def read(tmp_path: Path, ids: list[str]) -> list[aggregate.Run]:
    write_index(tmp_path, "e", {CODE_VERSION: ids})
    return [a.finished for a in aggregate.read_experiment(tmp_path, "e") if a.finished is not None]


# ------------------------------------------------------------------------------------------- SAN-1


@pytest.mark.parametrize(
    ("accs", "status"), [([0.99, 0.985, 0.986], "đạt"), ([0.99, 0.97, 0.98], "không đạt")]
)
def test_san1_needs_b1_at_least_98_5_percent_on_average(
    tmp_path: Path, accs: list[float], status: str
) -> None:
    verdict = sanity.san1(read(tmp_path, cell(tmp_path, "b1", B1, accs)))

    assert verdict.status == status


def test_san1_without_b1_runs_reports_missing_runs(tmp_path: Path) -> None:
    verdict = sanity.san1(read(tmp_path, cell(tmp_path, "b1", B1, [0.99, 0.99])))  # thiếu seed 2

    assert verdict.status == "thiếu lượt" and not verdict.passed


# ------------------------------------------------------------------------------------------- SAN-2


@pytest.mark.parametrize(("fc", "status"), [([0.975] * 3, "đạt"), ([0.97, 0.97, 0.96], "không đạt")])
def test_san2_compares_fc_iid_with_b1_4k_within_one_point(
    tmp_path: Path, fc: list[float], status: str
) -> None:
    ids = cell(tmp_path, "b14k", B1_4K, [0.98] * 3) + cell(tmp_path, "fc", FC_IID, fc)

    assert sanity.san2(read(tmp_path, ids)).status == status


def test_san2_without_b1_4k_reports_missing_runs(tmp_path: Path) -> None:
    ids = cell(tmp_path, "fc", FC_IID, [0.98] * 3) + cell(
        tmp_path, "b1", B1, [0.99] * 3
    )  # B1 không thay B1-4k

    assert sanity.san2(read(tmp_path, ids)).status == "thiếu lượt"


# ------------------------------------------------------------------------------------------- SAN-3


@pytest.mark.parametrize(("alpha", "status"), [([0.6] * 3, "đạt"), ([0.75] * 3, "không đạt")])
def test_san3_needs_b2_at_alpha_0_1_twenty_points_below_iid(
    tmp_path: Path, alpha: list[float], status: str
) -> None:
    ids = cell(tmp_path, "iid", B2, [0.9] * 3) + cell(tmp_path, "a01", {**B2, **ALPHA}, alpha)

    assert sanity.san3(read(tmp_path, ids)).status == status


def test_san3_without_the_alpha_cell_reports_missing_runs(tmp_path: Path) -> None:
    assert sanity.san3(read(tmp_path, cell(tmp_path, "iid", B2, [0.9] * 3))).status == "thiếu lượt"


# ------------------------------------------------------------------------------------------- SAN-4


@pytest.mark.parametrize(("a100", "status"), [([0.975] * 5, "đạt"), ([0.96] * 5, "không đạt")])
def test_san4_keeps_alpha_100_within_one_point_of_iid_per_topology(
    tmp_path: Path, a100: list[float], status: str
) -> None:
    ids = cell(tmp_path, "ring-iid", {}, [0.98] * 5) + cell(tmp_path, "ring-a100", ALPHA100, a100)
    ids += cell(tmp_path, "fc-iid", FC_IID, [0.98] * 5) + cell(
        tmp_path, "fc-a100", {**FC_IID, **ALPHA100}, [0.98] * 5
    )

    verdict = sanity.san4(read(tmp_path, ids))

    assert verdict.status == status and ("ring" in verdict.detail)


def test_san4_without_the_iid_cell_of_a_topology_reports_missing_runs(tmp_path: Path) -> None:
    ids = cell(tmp_path, "ring-a100", ALPHA100, [0.98] * 5)

    assert sanity.san4(read(tmp_path, ids)).status == "thiếu lượt"


# ------------------------------------------------------------------------------------------- SAN-5


def test_san5_passes_on_runs_whose_combine_step_acts_as_designed(tmp_path: Path) -> None:
    ids = cell(tmp_path, "ring", {}, [0.9]) + cell(tmp_path, "fc", FC_IID, [0.9], rounds=FC_ROUNDS)
    ids += cell(tmp_path, "b1", B1_4K, [0.9], rounds=curve([0.5, 0.8], nodes=1))
    ids += cell(tmp_path, "b2", B2, [0.9])

    assert sanity.san5(read(tmp_path, ids)).status == "đạt"


@pytest.mark.parametrize(
    ("overrides", "rounds", "message"),
    [
        (FC_IID, [round_record(100, 20, [0.8, 0.8, 0.8, 0.7])], "acc_std"),  # FC + MH mà các node khác nhau
        ({}, [round_record(100, 20, [0.8] * 4, consensus_distance=0.6, drift_pre_combine=0.5)], "consensus"),
        ({}, [round_record(100, 20, [0.8] * 4, messages_cum=0)], "message"),  # DFL không gửi message
    ],
    ids=["fc-nodes-differ", "consensus-above-drift", "no-messages"],
)
def test_san5_flags_a_combine_step_that_did_not_act(
    tmp_path: Path, overrides: dict[str, object], rounds: list[dict[str, object]], message: str
) -> None:
    verdict = sanity.san5(read(tmp_path, cell(tmp_path, "x", overrides, [0.9], rounds=rounds)))

    assert verdict.status == "không đạt" and message in verdict.detail


def test_san5_flags_a_baseline_that_sent_messages(tmp_path: Path) -> None:
    write_run(tmp_path, "b2", cfg=config(**B2), rounds=curve([0.5, 0.8]))  # curve có messages_cum > 0

    assert sanity.san5(read(tmp_path, ["b2"])).status == "không đạt"


def test_san5_with_no_runs_reports_missing_runs() -> None:
    assert sanity.san5([]).status == "thiếu lượt"


# --------------------------------------------------------------------------------------- dòng lệnh


def test_command_line_exits_non_zero_unless_every_san_passes(tmp_path: Path) -> None:
    ids = cell(tmp_path, "b1", B1, [0.99] * 3) + cell(tmp_path, "b14k", B1_4K, [0.98] * 3)
    ids += cell(tmp_path, "fc", FC_IID, [0.975] * 3, rounds=FC_ROUNDS) + cell(tmp_path, "iid", B2, [0.9] * 3)
    write_index(tmp_path, "e0b", {CODE_VERSION: ids})

    assert sanity.main(["--results", str(tmp_path), "--e0b", "e0b"]) == 1  # thiếu ô B2 α = 0.1
    ids += cell(tmp_path, "a01", {**B2, **ALPHA}, [0.6] * 3)
    write_index(tmp_path, "e0b", {CODE_VERSION: ids})
    assert sanity.main(["--results", str(tmp_path), "--e0b", "e0b"]) == 0
    assert sanity.main(["--results", str(tmp_path), "--san5", "e0b"]) == 0

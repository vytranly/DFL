"""Test cho analysis/hypotheses.py: mỗi giả thuyết H1–H8 và RQ4 tính đúng phép so sánh chính đã đăng ký trong
thẻ docs/experiments/E<k>.md, theo quy tắc ADR-007 (T17a, T18a, T19a, T27c, T17b, T19b, T20b)."""

import math
from pathlib import Path

import pytest

from analysis import hypotheses, stats
from analysis.aggregate import AnalysisError
from analysis.convergence import Reach
from tests.analysis_fakes import config, curve, round_record, timing_records, write_index, write_run

SEEDS5, SEEDS3 = [0, 1, 2, 3, 4], [0, 1, 2]
LADDER = [
    ("iid", None),
    ("dirichlet", 100.0),
    ("dirichlet", 1.0),
    ("dirichlet", 0.5),
    ("dirichlet", 0.1),
    ("dirichlet", 0.05),
]


def skew(k: int, nodes: int = 4) -> list[list[int]]:
    """Ma trận node × lớp lệch dần theo k: k = 0 là IID (JS = 0)."""
    return [[10 + (20 * k if c == j else 0) for c in range(10)] for j in range(nodes)]


class Results:
    """Thư mục kết quả giả: ghi lượt theo nhóm thí nghiệm rồi ghi chỉ mục của từng nhóm."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.groups: dict[str, list[str]] = {}

    def add(
        self,
        group: str,
        *,
        seed: int,
        kind: str,
        part: tuple[str, float | None],
        acc: float = 0.9,
        tau: int = 20,
        mu: float = 0.0,
        mixing: str = "metropolis_hastings",
        model: str = "smallcnn",
        runtime: str = "inprocess",
        status: str = "ok",
        k: int = 0,
        rounds: list[dict[str, object]] | None = None,
        timing: list[dict[str, object]] | None = None,
    ) -> str:
        scheme, value = part
        key = {"dirichlet": "alpha", "label_k": "k", "quantity": "beta"}.get(scheme)
        fields: dict[str, object] = {"partition__scheme": scheme}
        if key:
            fields[f"partition__{key}"] = value
        cfg = config(
            seed=seed,
            topology__kind=kind,
            mixing=mixing,
            model=model,
            runtime=runtime,
            algorithm={"batch_size": 32, "lr": hypotheses.ETA, "mu": mu, "tau": tau},
            **fields,
        )
        run_id = f"{group}-{kind}-{scheme}{value}-t{tau}-m{mu}-{mixing}-{model}-{runtime}-s{seed}"
        write_run(
            self.root,
            run_id,
            cfg=cfg,
            rounds=rounds,
            status=status,
            final_acc=acc,
            class_counts=skew(k),
            timing=timing,
        )
        self.groups.setdefault(group, []).append(run_id)
        return run_id

    def index(self) -> Path:
        for group, ids in self.groups.items():
            write_index(self.root, group, {"c" * 64: ids})
        return self.root


def ladder(
    results: Results, *, ring_drop: float, fc_drop: float, diverged: tuple[int, str] | None = None
) -> None:
    """IID (E0b) và thang Dirichlet (E1) cho ring và FC; accuracy giảm `drop` mỗi mức, nhiễu theo seed."""
    for seed in SEEDS5:
        for kind, drop in (("ring", ring_drop), ("fully_connected", fc_drop)):
            for k, part in enumerate(LADDER):
                group = "e0b_sanity" if part[0] == "iid" else "e1_noniid"
                status = "diverged" if diverged == (seed, kind) and k == 4 else "ok"
                results.add(
                    group,
                    seed=seed,
                    kind=kind,
                    part=part,
                    acc=0.98 - drop * k + 0.001 * seed,
                    k=k,
                    status=status,
                )


def test_js_mean_is_zero_for_identical_nodes_and_one_for_disjoint_labels() -> None:
    assert hypotheses.js_mean([[5, 5], [5, 5]]) == 0.0
    # Hai node rời nhau, phân bố gộp (½, ½), m = (¾, ¼): JS = ½·log₂(4/3) + ¼·(log₂(2/3) + 1) ≈ 0.31128 bit.
    expected = 0.5 * math.log2(4 / 3) + 0.25 * (math.log2(2 / 3) + 1)
    assert (
        hypotheses.js_mean([[10, 0], [0, 10]]) == pytest.approx(expected) == pytest.approx(0.31128, abs=1e-5)
    )


def test_h1_is_supported_when_accuracy_falls_with_js_on_both_topologies(tmp_path: Path) -> None:
    results = Results(tmp_path)
    ladder(results, ring_drop=0.03, fc_drop=0.01)

    outcome = hypotheses.h1(hypotheses.load(results.index()))

    assert outcome.verdict == stats.SUPPORTED
    assert len(outcome.values) == 5 and all(v < 0 for v in outcome.values)
    assert len(outcome.run_ids) == 60


def test_h1_is_open_when_a_run_of_the_comparison_diverged(tmp_path: Path) -> None:
    results = Results(tmp_path)
    ladder(results, ring_drop=0.03, fc_drop=0.01, diverged=(2, "ring"))

    outcome = hypotheses.h1(hypotheses.load(results.index()))

    assert outcome.verdict == stats.OPEN
    assert "1/60 lượt phân kỳ" in outcome.detail


def test_a_missing_seed_is_an_error_not_a_smaller_comparison(tmp_path: Path) -> None:
    results = Results(tmp_path)
    ladder(results, ring_drop=0.03, fc_drop=0.01)
    results.groups["e1_noniid"] = [i for i in results.groups["e1_noniid"] if not i.endswith("-s4")]

    with pytest.raises(AnalysisError, match="seed"):
        hypotheses.h1(hypotheses.load(results.index()))


def test_h2_is_supported_when_the_fc_minus_ring_gap_grows_with_js(tmp_path: Path) -> None:
    results = Results(tmp_path)
    ladder(results, ring_drop=0.03, fc_drop=0.01)

    outcome = hypotheses.h2(hypotheses.load(results.index()))

    # g(k) = (0.03 − 0.01)·k tăng theo JS, nên mọi độ dốc dương.
    assert outcome.verdict == stats.SUPPORTED and all(v > 0 for v in outcome.values)


def drift_rounds(value: float) -> list[dict[str, object]]:
    return [round_record(step, 20, [0.9] * 4, drift_pre_combine=value) for step in (100, 200, 300, 400)]


def test_h3_needs_drift_to_grow_on_every_topology(tmp_path: Path) -> None:
    results = Results(tmp_path)
    for seed in SEEDS5:
        for kind in ("ring", "erdos_renyi", "gossip", "fully_connected"):
            grow = 1.0 if kind == "fully_connected" else 2.0
            results.add("e0b_sanity", seed=seed, kind=kind, part=("iid", None), rounds=drift_rounds(0.5))
            group = "e1_noniid" if kind in ("ring", "fully_connected") else "e2_topology"
            results.add(
                group, seed=seed, kind=kind, part=("dirichlet", 0.1), rounds=drift_rounds(0.5 * grow), k=4
            )

    outcome = hypotheses.h3(hypotheses.load(results.index()))

    # FC không tăng drift: min theo topology bằng 0 ở mọi seed, không đúng chiều.
    assert outcome.values == (0.0,) * 5
    assert outcome.verdict == stats.OPEN


def reach_rounds(acc: list[float], bytes_per_step: int) -> list[dict[str, object]]:
    return [
        round_record(step, 20, [a] * 4, bytes_cum=bytes_per_step * step)
        for step, a in zip((100, 200, 300, 400), acc, strict=True)
    ]


def test_h4_compares_bytes_at_95_and_counts_the_cheaper_ring(tmp_path: Path) -> None:
    results = Results(tmp_path)
    for seed in SEEDS5:
        # Ring vượt 95% muộn hơn FC theo bước, nhưng mỗi bước rẻ hơn 9.5 lần.
        results.add(
            "e1_noniid",
            seed=seed,
            kind="ring",
            part=("dirichlet", 0.1),
            rounds=reach_rounds([0.5, 0.9, 0.96, 0.97], 40),
        )
        results.add(
            "e1_noniid",
            seed=seed,
            kind="fully_connected",
            part=("dirichlet", 0.1),
            rounds=reach_rounds([0.9, 0.96, 0.97, 0.97], 380),
        )

    outcome = hypotheses.h4(hypotheses.load(results.index()))

    assert outcome.verdict == stats.SUPPORTED
    assert "5/5 cặp đúng chiều" in outcome.detail


def test_bytes_censored_at_the_first_mark_only_beat_a_value_above_their_bound() -> None:
    low = hypotheses.Cost(Reach(100.0, "≤ 100"), 38_000.0)
    never = hypotheses.Cost(Reach(math.inf, "> ngân sách"), math.inf)

    assert hypotheses.compare_cost(low, hypotheses.Cost(Reach(250.0, "250"), 50_000.0)) == -1
    assert hypotheses.compare_cost(low, hypotheses.Cost(Reach(250.0, "250"), 10_000.0)) == 0
    assert hypotheses.compare_cost(hypotheses.Cost(Reach(250.0, "250"), 10_000.0), never) == -1
    assert hypotheses.compare_cost(never, never) == 0
    assert hypotheses.compare_cost(low, low) == 0


def test_h5_h6_h7_on_three_seeds_reach_at_most_suggested(tmp_path: Path) -> None:
    results = Results(tmp_path)
    a01 = ("dirichlet", 0.1)
    for seed in SEEDS3:
        for kind in ("ring", "fully_connected"):
            results.add("e1_noniid", seed=seed, kind=kind, part=a01, acc=0.85 + 0.01 * seed)
            results.add("e3a_algorithms", seed=seed, kind=kind, part=a01, mu=0.5, acc=0.87 + 0.01 * seed)
            results.add("e3b_tau_mu", seed=seed, kind=kind, part=a01, tau=5, acc=0.9 + 0.012 * seed)
            results.add("e3b_tau_mu", seed=seed, kind=kind, part=a01, tau=50, acc=0.8)
        results.add(
            "e1c_objective",
            seed=seed,
            kind="server",
            part=a01,
            mixing="size_weighted",
            acc=0.86 + 0.011 * seed,
        )
    runs = hypotheses.load(results.index())

    assert [hypotheses.h5(runs).verdict, hypotheses.h6(runs).verdict, hypotheses.h7(runs).verdict] == [
        stats.SUGGESTED
    ] * 3
    assert hypotheses.h5(runs).values == pytest.approx((0.02, 0.02, 0.02))


def test_rq4_compares_mh_with_uniform_weights_on_er(tmp_path: Path) -> None:
    results = Results(tmp_path)
    for seed in SEEDS3:
        results.add(
            "e2_topology", seed=seed, kind="erdos_renyi", part=("dirichlet", 0.1), acc=0.86 + 0.01 * seed
        )
        results.add(
            "e4_quantity", seed=seed, kind="erdos_renyi", part=("dirichlet", 0.1), mixing="uniform", acc=0.84
        )

    outcome = hypotheses.rq4(hypotheses.load(results.index()))

    assert outcome.verdict == stats.SUGGESTED and outcome.values == pytest.approx((0.02, 0.03, 0.04))


def comm_timing(rounds: list[dict[str, object]], seconds_per_round: float) -> list[dict[str, object]]:
    records = timing_records(rounds, 4)
    for record in records:
        record["comm_s"] = 5 * seconds_per_round  # 5 round giữa hai lần ghi (100 bước, τ = 20)
    return records


def e8_results(tmp_path: Path, exponent: float, results: Results | None = None) -> Results:
    results = results if results is not None else Results(tmp_path)
    for seed in SEEDS3:
        for model, params in hypotheses.PARAMS.items():
            for kind in ("ring", "gossip", "fully_connected"):
                for part in (("iid", None), ("dirichlet", 0.1)):
                    rounds = curve([0.5, 0.7, 0.8, 0.85])
                    results.add(
                        "e8_model_complexity",
                        seed=seed,
                        kind=kind,
                        part=part,
                        model=model,
                        runtime="processes",
                        rounds=rounds,
                        timing=comm_timing(rounds, 1e-4 * params**exponent * (1 + 0.01 * seed)),
                    )
    return results


def test_h8_fits_comm_time_against_parameter_count_on_a_log_scale(tmp_path: Path) -> None:
    outcome = hypotheses.h8(hypotheses.load(e8_results(tmp_path, 0.5).index()))

    assert outcome.verdict == stats.SUGGESTED
    assert outcome.values == pytest.approx((0.5, 0.5, 0.5))


def test_e8_cells_never_mix_with_core_cells_of_the_same_topology_and_level(tmp_path: Path) -> None:
    # Review T25: smallcnn × {ring, FC} × α = 0.1 của E8 (N = 8, process runtime) trùng mọi trường khác với ô
    # của E1; trước khi Cell so runtime, H4 báo "hai lượt ở seed 0" và H8 thiếu ô.
    results = Results(tmp_path)
    for seed in SEEDS5:
        for kind in ("ring", "fully_connected"):
            results.add(
                "e1_noniid",
                seed=seed,
                kind=kind,
                part=("dirichlet", 0.1),
                rounds=curve([0.8, 0.96, 0.97, 0.98]),
            )
    runs = hypotheses.load(e8_results(tmp_path, 0.5, results).index())

    assert hypotheses.h4(runs).verdict != "chưa đủ dữ liệu"
    assert hypotheses.h8(runs).values == pytest.approx((0.5, 0.5, 0.5))


def test_e8_checks_flag_bytes_that_are_not_messages_times_parameter_bytes(tmp_path: Path) -> None:
    runs = hypotheses.load(e8_results(tmp_path, 0.5).index())

    problems = hypotheses.e8_checks(runs)

    # Fakes ghi 1,000 byte mỗi message, khác 4 × số tham số của mọi model.
    assert problems and "4 × số tham số" in problems[0]


def test_cli_prints_each_verdict_and_reports_comparisons_without_data(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results = Results(tmp_path)
    ladder(results, ring_drop=0.03, fc_drop=0.01)
    results.index()

    assert hypotheses.main(["--results", str(tmp_path), "--only", "H1", "H5"]) == 0

    out = capsys.readouterr().out
    assert "H1" in out and stats.SUPPORTED in out
    assert "H5: chưa đủ dữ liệu" in out


# ------------------------------------------------------------------------- H9 (E7) và H10 (E9), ADR-014/015

STAR = [[0, i] for i in range(1, 8)]  # node 0 là hub (bậc 7), 7 leaf (bậc 1)


def graph_run(
    root: Path, group: str, run_id: str, *, seed: int, partition: dict, acc: list[float], **kw
) -> str:
    cfg = config(
        seed=seed,
        num_nodes=8,
        topology={"kind": "barabasi_albert", "m": 1, "p": None},
        algorithm={"batch_size": 32, "lr": hypotheses.ETA, "mu": 0.0, "tau": 20},
        partition={
            "alpha": None,
            "beta": None,
            "holders": None,
            "k": None,
            "min_size": 64,
            "placement": "random",
            **partition,
        },
    )
    write_run(
        root,
        run_id,
        cfg=cfg,
        graph_edges=STAR,
        final_acc=acc,
        rounds=curve([0.5, 0.7, 0.8, 0.85], nodes=8),
        **kw,
    )
    return run_id


def e7_results(root: Path, gaps: tuple[float, ...], *, diverged_seed: int | None = None) -> Path:
    held = [[10] * 10] + [[10] * 5 + [0] * 5] * 7  # node 0 giữ lớp 5–9
    ids = []
    for seed, gap in zip(SEEDS3, gaps, strict=True):
        for placement, acc in (("hub", 0.8 + gap), ("edge", 0.8)):
            status = "diverged" if (seed == diverged_seed and placement == "hub") else "ok"
            ids.append(
                graph_run(
                    root,
                    "e7_centrality",
                    f"e7-{placement}-{seed}",
                    seed=seed,
                    partition={"scheme": "rare_classes", "holders": 1, "placement": placement},
                    acc=[0.99] + [acc] * 7,
                    class_counts=held,
                    status=status,
                )
            )
    write_index(root, "e7_centrality", {"c" * 64: ids})
    return root


def test_h9_compares_non_holders_of_hub_against_edge_and_is_at_most_suggested(tmp_path: Path) -> None:
    runs = hypotheses.load(e7_results(tmp_path, (0.10, 0.20, 0.15)), ["e7_centrality"])

    outcome = hypotheses.h9(runs)

    assert outcome.verdict == stats.SUGGESTED and outcome.values == pytest.approx((0.10, 0.20, 0.15))


def test_h9_is_open_and_reports_the_diverged_seed(tmp_path: Path) -> None:
    runs = hypotheses.load(e7_results(tmp_path, (0.10, 0.20, 0.15), diverged_seed=1), ["e7_centrality"])

    outcome = hypotheses.h9(runs)

    assert outcome.verdict == stats.OPEN and "1/6 lượt phân kỳ" in outcome.detail


def test_h10_compares_hubs_with_leaves_on_ba_at_alpha_0_1(tmp_path: Path) -> None:
    ids = [
        graph_run(
            tmp_path,
            "e9",
            f"e9-ba-{seed}",
            seed=seed,
            partition={"scheme": "dirichlet", "alpha": 0.1},
            acc=[0.95] + [0.9 - 0.01 * seed] * 7,
        )
        for seed in SEEDS3
    ]
    write_index(tmp_path, "e9_topology_shapes", {"c" * 64: ids})

    outcome = hypotheses.h10(hypotheses.load(tmp_path, ["e9_topology_shapes"]))

    assert outcome.verdict == stats.SUGGESTED and outcome.values == pytest.approx((0.05, 0.06, 0.07))


def test_each_comparison_reads_only_its_own_groups_so_e9_never_doubles_core_cells(tmp_path: Path) -> None:
    # E9 chạy lại ô ring, FC ở α = 0.1 với cùng cấu hình như E1 (khác code_version); H4 chỉ đọc nhóm lõi.
    results = Results(tmp_path)
    for seed in SEEDS5:
        for kind in ("ring", "fully_connected"):
            results.add(
                "e1_noniid",
                seed=seed,
                kind=kind,
                part=("dirichlet", 0.1),
                rounds=curve([0.8, 0.96, 0.97, 0.98]),
            )
    for seed in SEEDS3:
        results.add(
            "e9_topology_shapes",
            seed=seed,
            kind="ring",
            part=("dirichlet", 0.1),
            rounds=curve([0.8, 0.96, 0.97, 0.98]),
        )

    assert hypotheses.groups_of("H4") == hypotheses.GROUPS
    assert hypotheses.groups_of("H10") == ("e9_topology_shapes",)
    assert hypotheses.h4(hypotheses.load(results.index(), hypotheses.groups_of("H4"))).values

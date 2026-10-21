"""Test cho analysis/plots.py và analysis/provenance.py: quy tắc INT-1 (--draft, results/, analysis/ sạch),
PROVENANCE, dữ liệu của P1 và P7 (T16a; SPEC INT-1, INT-2, INT-6, REQ-10). Không phụ thuộc trạng thái git của
cây: kiểm "analysis/ bẩn" bằng cách thay hàm kiểm, và thư mục results/ của repo được thay bằng thư mục tạm.
"""

import dataclasses
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from analysis import aggregate, plots, provenance
from analysis.aggregate import AnalysisError
from tests.analysis_fakes import CODE_VERSION, config, curve, round_record, write_index, write_run

ANALYSIS = Path(__file__).resolve().parents[2] / "analysis"


def two_topologies(root: Path) -> list[aggregate.Run]:
    for seed in (0, 1):
        write_run(root, f"ring-iid-{seed}", cfg=config(seed=seed), rounds=curve([0.5 + 0.1 * seed, 0.8]))
        write_run(
            root,
            f"fc-a01-{seed}",
            cfg=config(
                seed=seed,
                topology__kind="fully_connected",
                partition__scheme="dirichlet",
                partition__alpha=0.1,
            ),
            rounds=curve([0.4, 0.7]),
        )
    return [a.finished for a in aggregate.scan(root) if a.finished is not None]


def test_analysis_version_hashes_every_analysis_file_like_code_version(tmp_path: Path) -> None:
    copy = tmp_path / "analysis"
    shutil.copytree(ANALYSIS, copy, ignore=shutil.ignore_patterns("__pycache__"))
    before = provenance.analysis_version(copy)

    (copy / "plots.py").write_text((copy / "plots.py").read_text() + "\n# đổi\n")

    assert len(before) == 64 and before == provenance.analysis_version(ANALYSIS)
    assert provenance.analysis_version(copy) != before
    (copy / "plots.py").write_bytes((ANALYSIS / "plots.py").read_bytes().replace(b"\n", b"\r\n"))
    assert provenance.analysis_version(copy) == before  # \r\n và \n như nhau, như code_version


def test_learning_curves_average_over_seeds_per_topology_and_level(tmp_path: Path) -> None:
    curves = plots.learning_curves(two_topologies(tmp_path))

    steps, mean, std, seeds = curves[("ring", "IID")]
    assert list(steps) == [100, 200] and seeds == 2
    assert mean == pytest.approx([0.55, 0.8]) and std == pytest.approx([0.05, 0.0])
    assert set(curves) == {("ring", "IID"), ("fully_connected", "α = 0.1")}


def test_class_matrices_come_from_the_manifest_once_per_partition(tmp_path: Path) -> None:
    matrices = plots.class_matrices(two_topologies(tmp_path))

    assert set(matrices) == {"IID", "α = 0.1"}
    counts = matrices["IID"]
    assert counts.shape == (4, 10) and np.array_equal(counts.sum(axis=1), [100] * 4)


def test_draft_figures_carry_the_draft_label_and_each_figure_one_code_version(tmp_path: Path) -> None:
    runs = two_topologies(tmp_path)

    figure = plots.draw_p1(runs, draft=True)

    assert any("DRAFT" in text.get_text() for text in figure.texts)
    assert CODE_VERSION[:12] in " ".join(text.get_text() for text in figure.texts)


def test_without_draft_results_outside_the_repo_results_are_refused(tmp_path: Path) -> None:
    with pytest.raises(AnalysisError, match="results/"):
        plots.main(["--results", str(tmp_path), "--out", str(tmp_path / "out")])


def test_a_draft_never_writes_into_figures(tmp_path: Path) -> None:
    with pytest.raises(AnalysisError, match="figures/"):
        plots.main(["--results", str(tmp_path), "--out", str(plots.FIGURES_ROOT), "--draft"])


def test_without_draft_a_dirty_analysis_directory_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(plots, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(provenance, "analysis_dirty", lambda: True)

    with pytest.raises(AnalysisError, match="analysis/"):
        plots.main(["--results", str(tmp_path), "--out", str(tmp_path / "out")])


def test_without_draft_a_code_version_missing_from_the_ledger_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    two_topologies(tmp_path)
    write_index(tmp_path, "e1_noniid", {CODE_VERSION: [p.name for p in (tmp_path / "runs").iterdir()]})
    monkeypatch.setattr(plots, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(plots, "LEDGER", tmp_path / "CODE_VERSIONS.md")
    monkeypatch.setattr(provenance, "analysis_dirty", lambda: False)

    with pytest.raises(AnalysisError, match="INT-6"):
        plots.main(["--results", str(tmp_path), "--out", str(tmp_path / "out")])

    (tmp_path / "CODE_VERSIONS.md").write_text(f"| 2026-10-16 | {CODE_VERSION} | E1 |\n")
    assert plots.main(["--results", str(tmp_path), "--out", str(tmp_path / "out")]) == 0
    record = json.loads((tmp_path / "out" / "PROVENANCE.json").read_text())
    assert record["figures"]["P1"]["code_version"] == CODE_VERSION
    assert len(record["figures"]["P1"]["run_ids"]) == 4 and len(record["analysis_version"]) == 64


def test_a_figure_without_data_is_skipped_and_the_old_file_is_kept(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "P1.png").write_bytes(b"old")
    (out / "PROVENANCE.json").write_text(json.dumps({"figures": {"P1": {"run_ids": ["old"]}}}))
    (tmp_path / "runs").mkdir()

    assert plots.main(["--results", str(tmp_path), "--out", str(out), "--draft"]) == 0

    assert (out / "P1.png").read_bytes() == b"old"
    assert "bỏ qua P1" in capsys.readouterr().out
    assert json.loads((out / "PROVENANCE.json").read_text())["figures"]["P1"]["run_ids"] == ["old"]


def test_a_draft_still_refuses_runs_that_differ_in_a_fixed_variable(tmp_path: Path) -> None:
    # INT-1 chỉ nới lượt bẩn, lịch đánh giá riêng và INT-6; biến cố định (INT-2) vẫn phải trùng.
    write_run(tmp_path, "n4", rounds=curve([0.5, 0.8]))
    write_run(tmp_path, "n8", cfg=config(num_nodes=8), rounds=curve([0.5, 0.8, 0.9], nodes=8))

    with pytest.raises(AnalysisError, match="num_nodes"):
        plots.main(["--results", str(tmp_path), "--out", str(tmp_path / "out"), "--draft"])


def comm_runs(root: Path) -> list[aggregate.Run]:
    """Ring và FC (2 seed), đường server, cùng B1 và B2 là hai baseline không có message."""
    for seed in (0, 1):
        write_run(root, f"ring-{seed}", cfg=config(seed=seed))
        write_run(root, f"fc-{seed}", cfg=config(seed=seed, topology__kind="fully_connected"))
    server = [
        round_record(
            step,
            20,
            [0.6] * 4,
            node_messages_cum=[k] * 4 + [4 * k],
            node_bytes_cum=[1000 * k] * 4 + [4000 * k],
        )
        for step, k in ((100, 10), (200, 20), (300, 30), (400, 40))
    ]
    write_run(root, "server-0", cfg=config(topology__kind="server"), rounds=server)
    write_run(root, "b1-0", cfg=config(num_nodes=1, topology__kind="central"), rounds=curve([0.9], nodes=1))
    write_run(root, "b2-0", cfg=config(topology__kind="local_only"))
    return [a.finished for a in aggregate.scan(root) if a.finished is not None]


def test_comm_figures_keep_the_server_path_and_drop_baselines_without_messages(tmp_path: Path) -> None:
    runs = plots.select_comm(comm_runs(tmp_path), draft=False)

    assert sorted(r.run_id for r in runs) == ["fc-0", "fc-1", "ring-0", "ring-1", "server-0"]


def test_comm_figures_keep_only_tau_20_and_mu_0_outside_draft(tmp_path: Path) -> None:
    write_run(tmp_path, "tau20", cfg=config())
    write_run(tmp_path, "tau10", cfg=config(algorithm={"batch_size": 32, "lr": 0.05, "mu": 0.0, "tau": 10}))
    write_run(tmp_path, "prox", cfg=config(algorithm={"batch_size": 32, "lr": 0.05, "mu": 0.05, "tau": 20}))
    runs = [a.finished for a in aggregate.scan(tmp_path) if a.finished is not None]

    assert [r.run_id for r in plots.select_comm(runs, draft=False)] == ["tau20"]
    assert sorted(r.run_id for r in plots.select_comm(runs, draft=True)) == ["tau10", "tau20"]


def test_p12_has_one_panel_per_cost_type_with_units(tmp_path: Path) -> None:
    runs = plots.select_comm(comm_runs(tmp_path), draft=False)

    figure = plots.draw_p12(runs, draft=False)

    labels = [ax.get_ylabel() for ax in figure.axes]
    assert labels == ["message tích lũy", "byte tích lũy (MB)", "thời gian mỗi round (s)"]
    assert figure.axes[1].get_yscale() == "log"
    assert {t.get_text() for t in figure.axes[0].get_legend().get_texts()} == {
        "fully_connected (2 seed)",
        "ring (2 seed)",
        "server (1 seed)",
    }
    # Runtime inprocess không chờ giao tiếp: hình nói rõ comm = 0 là vì runtime (ADR-010).
    assert "comm = 0" in figure.axes[2].get_xlabel()


def test_p13_draws_the_virtual_server_as_a_labelled_point(tmp_path: Path) -> None:
    runs = plots.select_comm(comm_runs(tmp_path), draft=False)

    figure = plots.draw_p13(runs, draft=False)

    labels = [ax.get_ylabel() for ax in figure.axes]
    assert labels == ["message mỗi round", "byte mỗi round (MB)", "thời gian mỗi round (s)"]
    legend = {t.get_text() for t in figure.axes[0].get_legend().get_texts()}
    assert "server ảo" in legend


def test_p13_refuses_runs_written_before_adr_013(tmp_path: Path) -> None:
    rounds = curve([0.5, 0.8])
    for r in rounds:
        del r["node_messages_cum"], r["node_bytes_cum"]
    write_run(tmp_path, "old", rounds=rounds)
    runs = [a.finished for a in aggregate.scan(tmp_path) if a.finished is not None]

    with pytest.raises(AnalysisError, match="ADR-013"):
        plots.draw_p13(runs, draft=False)


def test_a_draft_draws_p12_and_p13_with_their_provenance(tmp_path: Path) -> None:
    for seed in (0, 1):
        write_run(tmp_path, f"ring-{seed}", cfg=config(seed=seed))

    assert plots.main(["--results", str(tmp_path), "--out", str(tmp_path / "out"), "--draft"]) == 0

    record = json.loads((tmp_path / "out" / "PROVENANCE.json").read_text())
    for name in ("P12", "P13"):
        assert (tmp_path / "out" / f"{name}.png").stat().st_size > 0
        assert record["figures"][name]["run_ids"] == ["ring-0", "ring-1"]


# ------------------------------------------------------------------------------------------- E7: P14, P15

E7_EDGES = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3]]


def e7_runs(root: Path, *, with_ring: bool = False) -> list[aggregate.Run]:
    """Ba vị trí × 2 seed trên cùng đồ thị 5 node; node giữ lớp hiếm là hub (0), lá (4) hay node 2."""
    holder = {"hub": 0, "edge": 4, "random": 2}
    for placement, node in holder.items():
        for seed in (0, 1):
            cfg = config(
                seed=seed,
                num_nodes=5,
                topology={"kind": "barabasi_albert", "m": 2, "p": None},
                partition={
                    "alpha": None,
                    "beta": None,
                    "holders": 1,
                    "k": None,
                    "min_size": 64,
                    "placement": placement,
                    "scheme": "rare_classes",
                },
            )
            counts = [[10] * 10 if i == node else [10] * 5 + [0] * 5 for i in range(5)]
            write_run(
                root,
                f"{placement}-{seed}",
                cfg=cfg,
                rounds=curve([0.5, 0.8], nodes=5),
                graph_edges=E7_EDGES,
                class_counts=counts,
                final_acc=[0.6 + 0.05 * i for i in range(5)],
            )
    if with_ring:
        write_run(root, "ring-0", cfg=config())
    return [a.finished for a in aggregate.scan(root) if a.finished is not None]


def test_e7_figures_take_only_rare_class_runs(tmp_path: Path) -> None:
    runs = plots.select_e7(e7_runs(tmp_path, with_ring=True), draft=False)

    assert len(runs) == 6 and {r.config["partition"]["scheme"] for r in runs} == {"rare_classes"}


def test_p14_has_one_panel_per_placement_with_degree_and_accuracy_axes(tmp_path: Path) -> None:
    runs = plots.select_e7(e7_runs(tmp_path), draft=False)

    figure = plots.draw_p14(runs, draft=False)

    panels = [ax for ax in figure.axes if ax.get_xlabel()]
    assert [ax.get_title().split(" · ")[0] for ax in panels] == ["hub", "edge", "random"]
    assert {ax.get_xlabel() for ax in panels} == {"bậc của node"}
    assert panels[0].get_ylabel() == "accuracy cuối trên đủ tập test (%)"
    legend = {t.get_text() for t in panels[0].get_legend().get_texts()}
    assert legend == {"giữ lớp 5–9", "không giữ lớp 5–9"}


def test_p15_draws_seed_0_of_each_placement_with_an_accuracy_colorbar(tmp_path: Path) -> None:
    runs = plots.select_e7(e7_runs(tmp_path), draft=False)

    figure = plots.draw_p15(runs, draft=False)

    titles = [ax.get_title() for ax in figure.axes if ax.get_title()]
    assert [t.split(" · ")[0] for t in titles] == ["hub", "edge", "random"]
    assert any(ax.get_ylabel() == "accuracy cuối (%)" for ax in figure.axes)
    assert "seed 0" in figure.get_suptitle()


def test_e7_figures_refuse_runs_without_the_graph(tmp_path: Path) -> None:
    runs = plots.select_e7(e7_runs(tmp_path), draft=False)
    broken = [dataclasses.replace(runs[0], manifest={**runs[0].manifest, "graph_edges": None}), *runs[1:]]

    with pytest.raises(AnalysisError, match="graph_edges"):
        plots.draw_p14(broken, draft=False)


def test_a_draft_draws_p14_and_p15_with_their_provenance(tmp_path: Path) -> None:
    e7_runs(tmp_path)

    assert plots.main(["--results", str(tmp_path), "--out", str(tmp_path / "out"), "--draft"]) == 0

    record = json.loads((tmp_path / "out" / "PROVENANCE.json").read_text())
    for name in ("P14", "P15"):
        assert (tmp_path / "out" / f"{name}.png").stat().st_size > 0
        assert len(record["figures"][name]["run_ids"]) == 6

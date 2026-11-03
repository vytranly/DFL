"""Test cho các hình P2–P6, P8–P11 và bảng T1 của analysis/plots.py (T16b, T16c, T27c; KH §3.8; REQ-10):
mỗi hình sinh được từ kết quả giả, có đơn vị trên trục, không vẽ khi thiếu dữ liệu, có mục trong PROVENANCE.
"""

import json
from pathlib import Path

import pytest

from analysis import aggregate, plots
from tests.analysis_fakes import config, round_record, timing_records, write_run

LEVELS = {"iid": {}, "a0.1": {"partition__scheme": "dirichlet", "partition__alpha": 0.1}}


def rounds(acc: list[float], *, drift: float = 0.5, bytes_per_step: int = 1_000) -> list[dict[str, object]]:
    return [
        round_record(
            step, 20, [a, a - 0.02, a + 0.01, a], drift_pre_combine=drift, bytes_cum=bytes_per_step * step
        )
        for step, a in zip((100, 200, 300, 400), acc, strict=True)
    ]


def core(root: Path) -> list[aggregate.Run]:
    """IID và α = 0.1 trên ring, ER, gossip, FC; 2 seed."""
    gaps = {"ring": 0.03, "erdos_renyi": 0.2, "gossip": None, "fully_connected": 1.0}
    for seed in (0, 1):
        for kind, gap in gaps.items():
            for name, fields in LEVELS.items():
                skewed = name != "iid"
                path = write_run(
                    root,
                    f"{kind}-{name}-{seed}",
                    cfg=config(seed=seed, topology__kind=kind, **fields),
                    rounds=rounds(
                        [0.8, 0.9, 0.96, 0.97] if not skewed else [0.5, 0.7, 0.8, 0.85],
                        drift=1.0 if skewed else 0.5,
                    ),
                    final_acc=0.85 if skewed else 0.97,
                    class_counts=[
                        [10 + (30 if c == j and skewed else 0) for c in range(10)] for j in range(4)
                    ],
                )
                manifest = json.loads((path / "manifest.json").read_text())
                manifest["spectral_gap"] = gap
                (path / "manifest.json").write_text(json.dumps(manifest))
    return [a.finished for a in aggregate.scan(root) if a.finished is not None]


def tau_scan(root: Path) -> list[aggregate.Run]:
    for seed in (0, 1):
        for kind in ("ring", "fully_connected"):
            for tau, mu in ((1, 0.0), (5, 0.0), (20, 0.0), (50, 0.0), (20, 0.5)):
                write_run(
                    root,
                    f"{kind}-t{tau}-m{mu}-{seed}",
                    cfg=config(
                        seed=seed,
                        topology__kind=kind,
                        algorithm={"batch_size": 32, "lr": 0.05, "mu": mu, "tau": tau},
                        **LEVELS["a0.1"],
                    ),
                    rounds=rounds([0.5, 0.8, 0.9, 0.96], bytes_per_step=4_000 // tau),
                )
    return [a.finished for a in aggregate.scan(root) if a.finished is not None]


def e8(root: Path) -> list[aggregate.Run]:
    for seed in (0, 1):
        for model, params in plots.hypotheses.PARAMS.items():
            for kind in ("ring", "gossip", "fully_connected"):
                for name, fields in LEVELS.items():
                    curve_rounds = rounds([0.5, 0.7, 0.8, 0.9], bytes_per_step=params // 100)
                    timing = timing_records(curve_rounds, 4)
                    for record in timing:
                        record["comm_s"] = 1e-6 * params
                    write_run(
                        root,
                        f"{model}-{kind}-{name}-{seed}",
                        cfg=config(
                            seed=seed, topology__kind=kind, model=model, runtime="processes", **fields
                        ),
                        rounds=curve_rounds,
                        timing=timing,
                    )
    return [a.finished for a in aggregate.scan(root) if a.finished is not None]


def labels(figure: plots.Figure) -> list[str]:
    return [ax.get_xlabel() + " | " + ax.get_ylabel() for ax in figure.axes]


@pytest.mark.parametrize("name", ["p2", "p3", "p4", "p5", "p8"])
def test_core_figures_draw_from_the_core_grid_and_skip_without_runs(tmp_path: Path, name: str) -> None:
    draw = getattr(plots, f"draw_{name}")

    figure = draw(plots.select_core(core(tmp_path), draft=False), draft=False)

    assert figure is not None and figure.axes
    assert draw([], draft=False) is None


def test_p2_puts_measured_js_on_x_and_final_accuracy_in_percent_on_y(tmp_path: Path) -> None:
    figure = plots.draw_p2(plots.select_core(core(tmp_path), draft=False), draft=False)

    assert labels(figure)[0] == "JS đo được của partition (bit) | accuracy cuối trên đủ tập test (%)"
    legend = {t.get_text() for t in figure.axes[0].get_legend().get_texts()}
    assert {"ring · Dirichlet", "fully_connected · Dirichlet"} <= legend


def test_p8_keeps_gossip_off_the_spectral_gap_axis(tmp_path: Path) -> None:
    figure = plots.draw_p8(plots.select_core(core(tmp_path), draft=False), draft=False)

    static, dynamic = figure.axes[:2]
    assert static.get_xscale() == "log"
    assert {t.get_text() for t in static.get_legend().get_texts()} == {
        "ring",
        "erdos_renyi",
        "fully_connected",
    }
    assert any("tốc độ trộn kỳ vọng 0.4" in t.get_text() for t in dynamic.get_xticklabels())  # N = 4


def test_p6_and_p9_have_a_byte_axis_and_a_step_axis(tmp_path: Path) -> None:
    runs = plots.select_p6(tau_scan(tmp_path), draft=False)

    p6 = plots.draw_p6(runs, draft=False)
    p9 = plots.draw_p9(runs, draft=False)

    assert all(ax.get_xscale() == "log" for ax in p6.axes[:2])
    assert "byte tích lũy của cả mạng (MB, trục log)" in labels(p6)[0]
    assert labels(p9)[0].startswith("bước cục bộ") and "MB" in labels(p9)[2]
    p9_legend = {t.get_text().split(" (")[0] for t in p9.axes[0].get_legend().get_texts()}
    assert p9_legend == {"τ = 1, μ = 0", "τ = 5, μ = 0", "τ = 20, μ = 0", "τ = 20, μ = 0.5", "τ = 50, μ = 0"}


def test_p10_and_p11_use_the_parameter_count_and_the_model_ladder(tmp_path: Path) -> None:
    runs = plots.select_e8(e8(tmp_path), draft=False)

    p10 = plots.draw_p10(runs, draft=False)
    p11 = plots.draw_p11(runs, draft=False)

    assert all(ax.get_xscale() == "log" for ax in p10.axes)
    assert labels(p10)[0].startswith("số tham số của model (trục log)")
    assert len(p11.axes) == 3
    legend = [t.get_text() for t in p11.axes[0].get_legend().get_texts()]
    assert legend[0].startswith("logreg (7,850 tham số)") and legend[-1].startswith("cnn_mcmahan")


def test_t1_has_one_row_per_cell_with_every_registered_column(tmp_path: Path) -> None:
    table = plots.table_t1(plots.select_t1(core(tmp_path), draft=False))

    assert len(table) == 8
    assert list(table.columns) == [
        "topology",
        "mức",
        "τ",
        "μ",
        "trộn",
        "seed",
        "phân kỳ",
        "acc (%)",
        "acc^loc (%)",
        "macro-F1 (%)",
        "R@95 (bước)",
        "bytes@95 (MB)",
        "độ dao động",
        "std giữa node (%)",
    ]
    ring_iid = table[(table["topology"] == "ring") & (table["mức"] == "IID")].iloc[0]
    assert ring_iid["acc (%)"] == "97.00 ± 0.00" and ring_iid["R@95 (bước)"].endswith("(2/2 seed đạt)")


def test_a_draft_writes_every_figure_and_the_t1_table_with_provenance(tmp_path: Path) -> None:
    core(tmp_path)

    assert plots.main(["--results", str(tmp_path), "--out", str(tmp_path / "out"), "--draft"]) == 0

    record = json.loads((tmp_path / "out" / "PROVENANCE.json").read_text())
    for name in ("P2", "P3", "P4", "P5", "P8", "T1"):
        assert name in record["figures"]
    assert (tmp_path / "out" / "T1.md").read_text().startswith("| topology | mức |")


def test_a_diverged_seed_counts_as_not_reaching_95_in_t1_and_p3(tmp_path: Path) -> None:
    # Review T25 (R4): lượt phân kỳ là +∞ trong trung vị R@95 trên đủ seed, không bị bỏ khỏi mẫu.
    runs = core(tmp_path)
    write_run(
        tmp_path,
        "ring-iid-2",
        cfg=config(seed=2, topology__kind="ring"),
        rounds=rounds([0.8, 0.9, 0.95, 0.96])[:2],
        status="diverged",
    )
    runs = [a.finished for a in aggregate.scan(tmp_path) if a.finished is not None]

    table = plots.table_t1(plots.select_t1(runs, draft=False))
    figure = plots.draw_p3(plots.select_core(runs, draft=False), draft=False)

    ring_iid = table[(table["topology"] == "ring") & (table["mức"] == "IID")].iloc[0]
    assert ring_iid["phân kỳ"] == 1 and ring_iid["R@95 (bước)"].endswith("(2/3 seed đạt)")
    assert figure is not None


def test_provenance_is_written_after_every_figure_so_a_later_failure_leaves_it_in_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Review T25 (R1): hình sau lỗi thì PROVENANCE.json vẫn ghi các hình đã lưu.
    core(tmp_path)

    def broken(runs: list[aggregate.Run], *, draft: bool) -> None:
        raise plots.AnalysisError("hỏng giả")

    monkeypatch.setitem(plots.FIGURES, "P17", (plots.FIGURES["P17"][0], plots.select_comm, broken))
    with pytest.raises(plots.AnalysisError, match="hỏng giả"):
        plots.main(["--results", str(tmp_path), "--out", str(tmp_path / "out"), "--draft"])

    record = json.loads((tmp_path / "out" / "PROVENANCE.json").read_text())
    assert "P1" in record["figures"] and (tmp_path / "out" / "P1.png").exists()

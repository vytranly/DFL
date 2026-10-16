"""Test cho P7 từ describe.json (T05b; KH §3.8; SPEC INT-1): heatmap node × lớp kèm kích thước node cho mọi
cách chia, trục có nhãn; hình từ describe chỉ ghi ra ngoài figures/.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from analysis import plots
from analysis.aggregate import AnalysisError


def describe_report(n: int = 4) -> dict[str, object]:
    """describe.json mẫu theo định dạng của dfl.sim.describe.write: 3 cách chia, 2 seed."""

    def entry(label: str, js: float, skew: int) -> dict[str, object]:
        per_seed = []
        for seed in (0, 1):
            counts = [[10 + skew * ((i + c + seed) % 2) for c in range(10)] for i in range(n)]
            per_seed.append({"seed": seed, "js_mean": js, "node_class_counts": counts})
        return {"label": label, "mean": {"js_mean": js}, "per_seed": per_seed}

    return {
        "config": {"dataset": "mnist", "num_nodes": n},
        "seeds": [0, 1],
        "skipped": {},
        "train_size": 55_000,
        "schemes": {
            "dirichlet_0.1": entry("Dirichlet α = 0.1", 0.55, 30),
            "iid": entry("IID", 0.001, 0),
            "label_2": entry("#C = 2", 0.61, 50),
        },
    }


def test_describe_matrices_take_one_seed_per_scheme_ordered_by_js() -> None:
    matrices = plots.describe_matrices(describe_report(), seed=1)

    assert list(matrices) == ["IID", "Dirichlet α = 0.1", "#C = 2"]  # JS tăng dần
    assert matrices["#C = 2"].shape == (4, 10)
    assert np.array_equal(matrices["#C = 2"][0], [10 + 50 * ((c + 1) % 2) for c in range(10)])


def test_p7_from_describe_has_a_labelled_panel_and_size_bar_per_scheme() -> None:
    figure = plots.draw_p7_describe(describe_report(), seed=0)

    heatmaps = [ax for ax in figure.axes if ax.get_xlabel() == "lớp"]
    assert len(heatmaps) == 3 and all(ax.get_ylabel() == "node" for ax in heatmaps)
    assert [ax.get_title() for ax in heatmaps] == ["IID", "Dirichlet α = 0.1", "#C = 2"]
    assert sum("kích thước" in ax.get_xlabel() for ax in figure.axes) == 3


def test_an_unknown_seed_is_an_error() -> None:
    with pytest.raises(AnalysisError, match="seed 7"):
        plots.describe_matrices(describe_report(), seed=7)


def test_command_line_writes_p7_from_describe_outside_figures_only(tmp_path: Path) -> None:
    source = tmp_path / "describe.json"
    source.write_text(json.dumps(describe_report()), encoding="utf-8")

    assert plots.main(["--describe", str(source), "--out", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "P7-describe.png").stat().st_size > 0
    with pytest.raises(AnalysisError, match="figures/"):
        plots.main(["--describe", str(source), "--out", str(plots.FIGURES_ROOT)])

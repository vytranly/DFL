"""Tập đánh giá và đánh giá cuối (T11e; ADR-007; SPEC REQ-03, §5.2): lượt đánh giá trên validation (E0a) không
nạp hay đánh giá ảnh test nào; đánh giá cuối dùng đủ split, learning curve dùng tập con cố định theo split.
"""

import dataclasses
import json
from pathlib import Path

import pytest
import torch

from dfl.config import EvalConfig, load_config
from dfl.data import datasets
from dfl.sim import run

ROOT = Path(__file__).resolve().parents[2]
SMOKE = dataclasses.replace(
    load_config(ROOT / "configs" / "profiles" / "smoke.yaml"),
    local_steps=4,
    eval=EvalConfig(subset_size=100, every_steps=None),
)


def on(split: str) -> object:
    return dataclasses.replace(SMOKE, eval=EvalConfig(split=split, subset_size=100, every_steps=None))


@pytest.mark.parametrize(("split", "test_loads", "final_size"), [("validation", 0, 200), ("test", 1, 500)])
def test_a_validation_run_never_loads_test_images_and_evaluates_the_whole_split_at_the_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, split: str, test_loads: int, final_size: int
) -> None:
    # Arrange: đếm số lần gọi bộ nạp test; ca test cho thấy bộ đếm thật sự nằm trên đường nạp.
    calls: list[str] = []
    original = datasets.load_test_raw

    def counting(name: str, root: object) -> datasets.RawSplit:
        calls.append(name)
        return original(name, root)

    monkeypatch.setattr(datasets, "load_test_raw", counting)

    # Act
    attempt = run.run(on(split), out=tmp_path)

    # Assert
    summary = json.loads((attempt / "summary.json").read_text(encoding="utf-8"))
    assert len(calls) == test_loads
    assert (summary["status"], summary["final_eval_split"], summary["final_eval_size"]) == (
        "ok",
        split,
        final_size,
    )
    for key in ("final_acc_per_node", "final_acc_local_per_node", "final_f1_macro_per_node"):
        assert len(summary[key]) == SMOKE.num_nodes


@pytest.mark.parametrize("split", ["test", "validation"])
def test_the_curve_uses_the_fixed_subset_and_the_end_uses_the_whole_split(split: str) -> None:
    data = datasets.load("synthetic")
    assembly = run.assemble(on(split), data)

    assert len(assembly.curve_set().y) == 100
    assert torch.equal(assembly.final_set().y, data.split(split).y)


def test_the_subset_is_the_same_for_every_seed_and_differs_between_splits() -> None:
    data = datasets.load("synthetic")
    seed0 = run.assemble(on("test"), data).curve_set()
    seed3 = run.assemble(dataclasses.replace(on("test"), seed=3), data).curve_set()
    validation = run.assemble(on("validation"), data).curve_set()

    assert torch.equal(seed0.x, seed3.x) and torch.equal(seed0.y, seed3.y)
    assert not torch.equal(seed0.x, validation.x)

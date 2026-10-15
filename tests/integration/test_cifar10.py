"""Đường CIFAR-10 qua `sim.run` trên dữ liệu tổng hợp 3×32×32: CNN của NIID-Bench, byte theo số tham số
của nó, augmentation crop_flip ở cả hai runtime (T28d, T28e; ADR-012; SPEC §5.2, REQ-05; T-INV-13). Fixture
autouse của tests/conftest.py kiểm RNG toàn cục không đổi (T-INV-5).
"""

import dataclasses
import json
from pathlib import Path

import pytest

from dfl.config import EvalConfig, load_config
from dfl.data import datasets
from dfl.io import manifest
from dfl.learning.augment import CropFlip
from dfl.sim import run

ROOT = Path(__file__).resolve().parents[2]
# Profile smoke_cifar10 rút gọn còn 2 round và tập con đánh giá 100 ảnh (đủ 10 lớp), để test nhanh.
SMOKE_CIFAR10 = dataclasses.replace(
    load_config(ROOT / "configs" / "profiles" / "smoke_cifar10.yaml"),
    local_steps=4,
    eval=EvalConfig(subset_size=100, every_steps=None),
)
# CNN của NIID-Bench, float32: 4 × 62,006 byte mỗi message; ring N = 4 có 8 message mỗi round.
MESSAGE_BYTES = 4 * 62_006


def test_smoke_cifar10_trains_the_niidbench_cnn_and_counts_its_bytes(tmp_path: Path) -> None:
    attempt = run.run(SMOKE_CIFAR10, out=tmp_path)

    rounds = [
        json.loads(line) for line in (attempt / "rounds.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    summary = json.loads((attempt / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((attempt / "manifest.json").read_text(encoding="utf-8"))
    assert summary["status"] == "ok" and (summary["final_eval_split"], summary["final_eval_size"]) == (
        "test",
        500,
    )
    assert [r["messages_cum"] for r in rounds] == [8, 16]
    assert all(r["bytes_cum"] == r["messages_cum"] * MESSAGE_BYTES for r in rounds)
    assert (manifest["config"]["dataset"], manifest["config"]["model"]) == (
        "synthetic_cifar10",
        "cnn_niidbench",
    )
    assert [sum(row) for row in manifest["node_class_counts"]] == [450] * 4  # 1,800 ảnh chia đều


@pytest.fixture(scope="module")
def here() -> manifest.Provenance:
    return dataclasses.replace(manifest.current_provenance(), dirty=False)


def test_crop_flip_reaches_the_trainer_with_the_black_of_the_run_data() -> None:
    cifar = datasets.load("synthetic_cifar10")

    trainer = run.assemble(SMOKE_CIFAR10, cifar).trainer()
    plain = run.assemble(
        load_config(ROOT / "configs" / "profiles" / "smoke.yaml"), datasets.load("synthetic")
    )

    assert trainer.augment == CropFlip(fill=cifar.black())
    assert plain.trainer().augment is None  # đường MNIST giữ nguyên


def test_t_inv_13_processes_reproduce_the_cifar10_path_with_crop_flip_byte_for_byte(
    tmp_path: Path, here: manifest.Provenance
) -> None:
    inprocess = run.run(SMOKE_CIFAR10, out=tmp_path / "inprocess", provenance=here)
    processes = run.run(
        dataclasses.replace(SMOKE_CIFAR10, runtime="processes"), out=tmp_path / "processes", provenance=here
    )

    assert (processes / "rounds.jsonl").read_bytes() == (inprocess / "rounds.jsonl").read_bytes()

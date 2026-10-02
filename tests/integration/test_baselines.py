"""Baseline B1, B1-4k, B2, B3 và biến thể trộn của E4 trên dữ liệu tổng hợp: số bước, số message,
objective_weights, spectral gap (T11c; KH §3.4, §3.5; SPEC §5.2, REQ-12; ADR-005).
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dfl import seeding
from dfl.config import AlgorithmConfig, EvalConfig, PartitionConfig, RunConfig, TopologyConfig, load_config
from dfl.data import datasets
from dfl.seeding import Stream
from dfl.sim import run

SMOKE = load_config(Path(__file__).resolve().parents[2] / "configs" / "profiles" / "smoke.yaml")
MESSAGE_BYTES = 4 * 46_730


def central(local_steps: int) -> RunConfig:
    """B1 (hay B1-4k): một node, τ = 20 nên 100 chia hết cho τ, đánh giá ở các bội của 100."""
    return dataclasses.replace(
        SMOKE,
        num_nodes=1,
        topology=TopologyConfig(kind="central"),
        local_steps=local_steps,
        algorithm=AlgorithmConfig(tau=20),
        eval=EvalConfig(subset_size=500, every_steps=100),
    )


def execute(
    config: RunConfig, out: Path
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    attempt = run.run(config, out=out)
    manifest = json.loads((attempt / "manifest.json").read_text(encoding="utf-8"))
    rounds = [
        json.loads(line) for line in (attempt / "rounds.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    summary = json.loads((attempt / "summary.json").read_text(encoding="utf-8"))
    return manifest, rounds, summary


@pytest.fixture(scope="module")
def data() -> datasets.Datasets:
    return datasets.load("synthetic")


@pytest.mark.parametrize("local_steps", [200, 300], ids=["b1", "b1-4k"])
def test_central_baseline_trains_one_node_on_all_data_without_messages(
    tmp_path: Path, local_steps: int
) -> None:
    manifest, rounds, summary = execute(central(local_steps), tmp_path)

    assert [r["local_steps"] for r in rounds] == list(range(100, local_steps + 1, 100))
    assert all(r["messages_cum"] == 0 and r["bytes_cum"] == 0 for r in rounds)
    assert (manifest["objective_weights"], manifest["spectral_gap"]) == ([1.0], None)
    assert sum(manifest["node_class_counts"][0]) == 1800
    assert (summary["messages_total"], len(summary["final_acc_per_node"])) == (0, 1)


def test_central_baseline_reads_the_train_set_in_order_with_its_own_batch_stream(
    data: datasets.Datasets,
) -> None:
    setup = run.build(central(200), data)

    (node,) = setup.nodes
    assert torch.equal(node.data.y, data.train.y) and torch.equal(node.data.x, data.train.x)
    expected = seeding.torch_generator(SMOKE.seed, Stream.B1_BATCH)
    assert torch.equal(node.generator.get_state(), expected.get_state())


def test_local_only_baseline_never_mixes(tmp_path: Path) -> None:
    config = dataclasses.replace(SMOKE, topology=TopologyConfig(kind="local_only"))

    manifest, rounds, _ = execute(config, tmp_path)

    assert all(r["messages_cum"] == 0 for r in rounds)
    assert all(r["consensus_distance"] > 0 for r in rounds)  # W = I: các model tách nhau
    assert (manifest["objective_weights"], manifest["spectral_gap"]) == ([0.25] * 4, None)


def test_server_baseline_sends_two_messages_per_node_per_round(tmp_path: Path) -> None:
    config = dataclasses.replace(SMOKE, topology=TopologyConfig(kind="server"), mixing="size_weighted")

    manifest, rounds, _ = execute(config, tmp_path)

    assert [r["messages_cum"] for r in rounds] == [8 * k for k in range(1, 7)]  # 2N mỗi round
    assert all(r["bytes_cum"] == r["messages_cum"] * MESSAGE_BYTES for r in rounds)
    assert all(r["consensus_distance"] == 0 for r in rounds)  # mọi node nhận cùng model của server
    sizes = np.array([sum(row) for row in manifest["node_class_counts"]], dtype=float)
    assert manifest["objective_weights"] == pytest.approx(list(sizes / sizes.sum()))
    assert manifest["spectral_gap"] == pytest.approx(1.0)


def test_uniform_mixing_optimizes_weights_proportional_to_degree_plus_one(tmp_path: Path) -> None:
    # Star N = 4: hub bậc 3, lá bậc 1, nên π = (4, 2, 2, 2)/10 (ADR-005).
    config = dataclasses.replace(SMOKE, topology=TopologyConfig(kind="star"), mixing="uniform")

    manifest, rounds, _ = execute(config, tmp_path)

    assert manifest["objective_weights"] == pytest.approx([0.4, 0.2, 0.2, 0.2])
    assert [r["messages_cum"] for r in rounds] == [6 * k for k in range(1, 7)]  # 3 cạnh, 2 chiều


def test_size_corrected_scales_each_loss_by_node_size_over_the_mean(
    tmp_path: Path, data: datasets.Datasets
) -> None:
    config = dataclasses.replace(
        SMOKE, mixing="size_corrected", partition=PartitionConfig(scheme="quantity", beta=2.0)
    )
    setup = run.build(config, data)
    sizes = np.array([len(part) for part in setup.partition.parts], dtype=float)

    manifest, _, _ = execute(config, tmp_path)

    assert len(set(sizes)) > 1  # kích thước node khác nhau, nên c_i không tầm thường
    assert [node.loss_scale for node in setup.nodes] == pytest.approx(list(sizes / sizes.mean()))
    assert manifest["objective_weights"] == pytest.approx(list(sizes / sizes.sum()))

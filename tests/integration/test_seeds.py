"""T-INV-5 qua một lượt chạy thật trên profile smoke (T11b; SPEC §6, §9.2; G9; INV-6).

Cùng seed và `code_version` thì `rounds.jsonl` trùng từng byte; khác seed thì khác cách chia và khác w⁰ ở mọi
tensor; cùng seed thì cùng cách chia, cùng vị trí node, cùng w⁰ và cùng luồng batch ở mọi topology và thuật
toán; RNG toàn cục của torch không đổi sau một lượt.
"""

import dataclasses
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch

from dfl import seeding
from dfl.config import AlgorithmConfig, EvalConfig, PartitionConfig, RunConfig, TopologyConfig, load_config
from dfl.data import datasets
from dfl.data.partition import IIDPartitioner, assign_positions
from dfl.seeding import Stream
from dfl.sim import run

SMOKE = load_config(Path(__file__).resolve().parents[2] / "configs" / "profiles" / "smoke.yaml")


@pytest.fixture(scope="module")
def data() -> datasets.Datasets:
    return datasets.load("synthetic")


def w0(setup: run.RunSetup) -> list[dict[str, torch.Tensor]]:
    return [node.model.state_dict() for node in setup.nodes]


# Lượt chạy thật trong test: 3 round, tập con đánh giá 100 ảnh (đủ 10 lớp), để test nhanh.
SHORT = dataclasses.replace(SMOKE, local_steps=6, eval=EvalConfig(subset_size=100, every_steps=None))


def test_t_inv_5_same_seed_and_code_give_byte_identical_rounds_files(tmp_path: Path) -> None:
    first = run.run(SHORT, out=tmp_path / "a")
    second = run.run(SHORT, out=tmp_path / "b")

    assert (first / "rounds.jsonl").read_bytes() == (second / "rounds.jsonl").read_bytes()


def test_t_inv_5_a_run_leaves_the_global_torch_rng_untouched(tmp_path: Path) -> None:
    before = torch.get_rng_state()

    run.run(SHORT, out=tmp_path)

    assert torch.equal(torch.get_rng_state(), before)


def test_t_inv_5_another_seed_changes_the_partition_and_every_w0_tensor(data: datasets.Datasets) -> None:
    a = run.build(SMOKE, data)
    b = run.build(dataclasses.replace(SMOKE, seed=1), data)

    assert not all(np.array_equal(x, y) for x, y in zip(a.partition.parts, b.partition.parts, strict=True))
    first, second = w0(a)[0], w0(b)[0]
    assert all(not torch.equal(first[name], second[name]) for name in first)


def test_t_inv_5_nodes_get_the_partition_permuted_by_the_placement_stream(data: datasets.Datasets) -> None:
    labels = data.train.y.numpy()
    raw = IIDPartitioner(SMOKE.partition.min_size)(labels, 4, seeding.numpy_rng(SMOKE.seed, Stream.PARTITION))
    expected = assign_positions(raw, seeding.numpy_rng(SMOKE.seed, Stream.PLACEMENT))

    setup = run.build(SMOKE, data)

    assert all(np.array_equal(x, y) for x, y in zip(setup.partition.parts, expected.parts, strict=True))
    assert not all(np.array_equal(x, y) for x, y in zip(raw.parts, expected.parts, strict=True))


def test_every_node_starts_from_the_same_w0_in_its_own_memory(data: datasets.Datasets) -> None:
    # Giả định A3 (KH §3.1): w⁰ chung theo seed; mỗi node vẫn có bộ nhớ tham số riêng (engine kiểm), và rút
    # minibatch từ luồng batch của chính nó, spawn_key (ID_BATCH, i) (SPEC §6).
    setup = run.build(SMOKE, data)
    states = w0(setup)

    for node in setup.nodes:
        expected = seeding.torch_generator(SMOKE.seed, Stream.BATCH, node.node_id)
        assert torch.equal(node.generator.get_state(), expected.get_state())

    for state in states[1:]:
        assert all(torch.equal(state[name], states[0][name]) for name in state)
        assert all(state[name].data_ptr() != states[0][name].data_ptr() for name in state)


VARIANTS: dict[str, Callable[[RunConfig], RunConfig]] = {
    "fully-connected": lambda c: dataclasses.replace(c, topology=TopologyConfig(kind="fully_connected")),
    "erdos-renyi": lambda c: dataclasses.replace(c, topology=TopologyConfig(kind="erdos_renyi", p=0.6)),
    "gossip": lambda c: dataclasses.replace(c, topology=TopologyConfig(kind="gossip")),
    "er-gossip": lambda c: dataclasses.replace(c, topology=TopologyConfig(kind="er_gossip", p=0.6)),
    "star": lambda c: dataclasses.replace(c, topology=TopologyConfig(kind="star")),
    "local-only": lambda c: dataclasses.replace(c, topology=TopologyConfig(kind="local_only")),
    "server": lambda c: dataclasses.replace(
        c, topology=TopologyConfig(kind="server"), mixing="size_weighted"
    ),
    "uniform": lambda c: dataclasses.replace(c, mixing="uniform"),
    "size-corrected": lambda c: dataclasses.replace(c, mixing="size_corrected"),
    "prox": lambda c: dataclasses.replace(c, algorithm=AlgorithmConfig(tau=2, mu=0.1)),
    "tau-1": lambda c: dataclasses.replace(c, algorithm=AlgorithmConfig(tau=1)),
}


@pytest.mark.parametrize("variant", list(VARIANTS), ids=list(VARIANTS))
def test_t_inv_5_same_seed_shares_partition_placement_w0_and_batch_streams_across_variants(
    data: datasets.Datasets, variant: str
) -> None:
    base = run.build(SMOKE, data)
    other = run.build(VARIANTS[variant](SMOKE), data)

    assert all(np.array_equal(x, y) for x, y in zip(base.partition.parts, other.partition.parts, strict=True))
    for mine, theirs in zip(w0(base), w0(other), strict=True):
        assert all(torch.equal(mine[name], theirs[name]) for name in mine)
    for mine, theirs in zip(base.nodes, other.nodes, strict=True):
        assert torch.equal(mine.generator.get_state(), theirs.generator.get_state())
        assert torch.equal(mine.data.x, theirs.data.x) and torch.equal(mine.data.y, theirs.data.y)


SCHEMES = {
    "iid": PartitionConfig(scheme="iid"),
    "dirichlet": PartitionConfig(scheme="dirichlet", alpha=0.5),
    "quantity": PartitionConfig(scheme="quantity", beta=2.0),
    "label-k": PartitionConfig(scheme="label_k", k=2),
}


@pytest.mark.parametrize("scheme", list(SCHEMES), ids=list(SCHEMES))
def test_t_inv_5_every_partition_scheme_is_fixed_by_the_seed(data: datasets.Datasets, scheme: str) -> None:
    # #C = k cần N ≥ 10 (G23); 1,800 ảnh chia cho 10 node vẫn trên min_size = 64.
    config = dataclasses.replace(SMOKE, partition=SCHEMES[scheme], num_nodes=10 if scheme == "label-k" else 4)

    first = run.build(config, data).partition
    again = run.build(config, data).partition

    assert all(np.array_equal(x, y) for x, y in zip(first.parts, again.parts, strict=True))
    assert sorted(np.concatenate(first.parts).tolist()) == list(range(len(data.train.y)))


def test_shards_are_rejected_outside_describe(data: datasets.Datasets) -> None:
    config = dataclasses.replace(SMOKE, partition=PartitionConfig(scheme="shards"))

    with pytest.raises(ValueError, match="QS-11"):
        run.build(config, data)

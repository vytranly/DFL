"""`dfl describe` trên dữ liệu tổng hợp: mọi cách chia × seed, JSON đọc lại được, không NaN, CSV, lệnh
`--check-req02` (T05a; KH §3.2; SPEC REQ-02, QS-11), và bảng topology (T08b; KH §3.3; ADR-009; INV-2).
"""

import csv
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from dfl import cli, seeding
from dfl.config import PartitionConfig, RunConfig, TopologyConfig, load_config
from dfl.data import datasets
from dfl.data.heterogeneity import summarize
from dfl.data.partition import DirichletPartitioner, assign_positions
from dfl.network import mixing, topology
from dfl.network.transport import InMemoryTransport
from dfl.seeding import Stream
from dfl.sim import describe
from dfl.sim.node import make_snapshot

ROOT = Path(__file__).resolve().parents[2]
# N = 10 để có #C = k (G23); min_size nhỏ để α = 0.05 chia được 1,800 ảnh tổng hợp.
CONFIG = dataclasses.replace(
    load_config(ROOT / "configs" / "profiles" / "smoke.yaml"),
    num_nodes=10,
    partition=PartitionConfig(min_size=8),
)
SEEDS = 3


@pytest.fixture(scope="module")
def described(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, object]]:
    out = tmp_path_factory.mktemp("describe")
    labels = datasets.load("synthetic").train.y.numpy()
    describe.write(describe.describe(CONFIG, labels, seeds=SEEDS), out)
    return out, json.loads((out / "describe.json").read_text(encoding="utf-8"))


def walk_numbers(value: object) -> list[float]:
    if isinstance(value, dict):
        return [x for v in value.values() for x in walk_numbers(v)]
    if isinstance(value, list):
        return [x for v in value for x in walk_numbers(v)]
    return [float(value)] if isinstance(value, int | float) and not isinstance(value, bool) else []


def test_every_scheme_is_described_for_every_seed(described: tuple[Path, dict[str, object]]) -> None:
    _, report = described
    schemes = report["schemes"]

    assert set(schemes) == {scheme.key for scheme in describe.schemes(min_size=8)}
    for entry in schemes.values():
        assert [row["seed"] for row in entry["per_seed"]] == list(range(SEEDS))
        assert all(np.array(row["node_class_counts"]).shape == (10, 10) for row in entry["per_seed"])


def test_the_report_reads_back_without_nan(described: tuple[Path, dict[str, object]]) -> None:
    numbers = walk_numbers(described[1])

    assert numbers and all(math.isfinite(x) for x in numbers)


def test_each_seed_uses_the_partition_and_placement_streams_of_a_run(
    described: tuple[Path, dict[str, object]],
) -> None:
    # Cùng đường với sim.run: ma trận node × lớp của describe trùng với lượt chạy cùng seed (P7).
    labels = datasets.load("synthetic").train.y.numpy()
    raw = DirichletPartitioner(0.1, 8)(labels, 10, seeding.numpy_rng(2, Stream.PARTITION))
    placed = assign_positions(raw, seeding.numpy_rng(2, Stream.PLACEMENT))
    expected = summarize(labels, placed.parts)

    row = described[1]["schemes"]["dirichlet_0.1"]["per_seed"][2]

    assert row["node_class_counts"] == [list(r) for r in expected.node_class_counts]
    assert (row["js_mean"], row["retries"]) == (expected.js_mean, raw.retries)


def test_the_csv_has_one_row_of_seed_means_per_scheme(described: tuple[Path, dict[str, object]]) -> None:
    out, report = described

    with (out / "describe.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    # CSV theo thứ tự hàng của bảng KH §3.2; JSON sắp khóa theo chữ cái (sort_keys).
    assert [row["scheme"] for row in rows] == [scheme.key for scheme in describe.schemes(min_size=8)]
    for row in rows:
        mean = report["schemes"][row["scheme"]]["mean"]
        assert float(row["js_mean"]) == pytest.approx(mean["js_mean"])


def test_label_k_schemes_are_skipped_below_ten_nodes(tmp_path: Path) -> None:
    labels = datasets.load("synthetic").train.y.numpy()
    small = dataclasses.replace(CONFIG, num_nodes=4)

    report = describe.describe(small, labels, seeds=1)

    assert {"label_2", "label_1"} <= set(report["skipped"])
    assert not {"label_2", "label_1"} & set(report["schemes"])


def test_check_req02_on_synthetic_data_exits_non_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Dữ liệu tổng hợp không phải MNIST, nên JS không khớp bảng KH §3.2: lệnh phải báo lỗi, không im lặng.
    config_path = tmp_path / "describe.yaml"
    config_path.write_text(
        f"extends: {ROOT / 'configs' / 'profiles' / 'smoke.yaml'}\nnum_nodes: 10\npartition:\n  min_size: 8\n"
    )

    status = cli.main(
        [
            "describe",
            "--config",
            str(config_path),
            "--seeds",
            "2",
            "--out",
            str(tmp_path / "out"),
            "--check-req02",
        ]
    )

    assert status == 1
    assert "REQ-02" in capsys.readouterr().out
    assert (tmp_path / "out" / "describe.json").exists()


# Bảng topology (T08b): khe phổ, tốc độ trộn kỳ vọng và số thông điệp kỳ vọng mỗi round.


@pytest.fixture(scope="module")
def topologies_n20() -> dict[str, dict[str, object]]:
    return describe.describe_topologies(RunConfig(), seeds=20, matchings=2_000)


def test_spectral_gaps_match_the_kh_3_3_table_at_n_20(topologies_n20: dict[str, dict[str, object]]) -> None:
    ring, fc, er = topologies_n20["T1 ring"], topologies_n20["T4 FC"], topologies_n20["T2 ER"]

    assert ring["spectral_gap"] == pytest.approx(0.0326, abs=1e-4)
    assert fc["spectral_gap"] == pytest.approx(1.0)
    assert 0.15 <= er["spectral_gap"] <= 0.25 and len(er["spectral_gap_per_seed"]) == 20
    assert min(er["spectral_gap_per_seed"]) <= er["spectral_gap"] <= max(er["spectral_gap_per_seed"])


def test_gossip_reports_an_expected_mixing_rate_not_a_spectral_gap(
    topologies_n20: dict[str, dict[str, object]],
) -> None:
    # ADR-009: 1 − √λ₂(E[W]) = 0.312 ở N = 20; so với E[W] ước lượng từ chính lịch gossip của lượt chạy.
    t3 = topologies_n20["T3 gossip"]
    schedule = topology.gossip(20, lambda r: seeding.numpy_rng(0, Stream.GOSSIP, r))
    expected = np.mean([mixing.metropolis_hastings(schedule.graph_at(r)) for r in range(4_000)], axis=0)
    sampled = 1 - math.sqrt(np.sort(np.linalg.eigvalsh(expected))[-2])

    assert "spectral_gap" not in t3 and t3["mixing_rate"] == pytest.approx(0.312, abs=1e-3)
    assert sampled == pytest.approx(t3["mixing_rate"], abs=0.01)
    assert t3["messages_per_round"] == 20
    assert "spectral_gap" not in topologies_n20["T3b ER gossip"]
    assert 0 < topologies_n20["T3b ER gossip"]["mixing_rate"] < 1


def test_expected_messages_are_twice_the_edges_and_match_one_transport_round() -> None:
    # Chi phí thật chỉ do Transport đếm (INV-2); 2·|E| chỉ là thuộc tính của đồ thị, nên phải khớp số đếm.
    config = RunConfig(num_nodes=6)
    rows = describe.describe_topologies(config, seeds=2, matchings=200)
    graphs = {
        "star": topology.star(6).graph_at(0),
        "T1 ring": topology.ring(6).graph_at(0),
        "T4 FC": topology.fully_connected(6).graph_at(0),
        "T2 ER": topology.erdos_renyi(6, 0.3, seeding.numpy_rng(0, Stream.GRAPH)).graph_at(0),
    }
    snapshot = make_snapshot(0, 0, {"w": torch.zeros(3)})
    for name, graph in graphs.items():
        transport = InMemoryTransport(6)
        for u, v in graph.edges():
            transport.send(v, dataclasses.replace(snapshot, sender=u))
            transport.send(u, dataclasses.replace(snapshot, sender=v))
        expected = rows[name]["messages_per_round" if name != "T2 ER" else "messages_per_round_seed0"]
        assert transport.counters().messages_cum == expected == 2 * graph.number_of_edges(), name


def test_t0_counts_the_server_path_of_b3_like_the_engine() -> None:
    # KH §3.3: T0 là star + server ảo id N (B3, E1c); engine gửi N snapshot lên và N snapshot về.
    rows = describe.describe_topologies(RunConfig(num_nodes=6), seeds=1, matchings=10)
    transport = InMemoryTransport(6, with_server=True)
    snapshot = make_snapshot(0, 0, {"w": torch.zeros(3)})
    for k in range(6):
        transport.send(6, dataclasses.replace(snapshot, sender=k))
        transport.send(k, dataclasses.replace(snapshot, sender=6))

    assert rows["T0 server"]["messages_per_round"] == transport.counters().messages_cum == 12
    assert "spectral_gap" not in rows["T0 server"]


def test_gossip_mixing_rate_has_an_odd_n_branch() -> None:
    # N lẻ: mỗi cặp được ghép với xác suất 1/N, nên λ₂(E[W]) = 1/2.
    assert describe.gossip_mixing_rate(5) == pytest.approx(1 - math.sqrt(0.5))
    assert describe.gossip_mixing_rate(20) == pytest.approx(1 - math.sqrt(0.5 - 1 / 38))


def test_er_gossip_on_a_complete_base_graph_is_the_gossip_of_t3() -> None:
    # p = 1 cho ER là K_N, nên T3b phải về đúng T3; đây là phép kiểm có giá trị biết trước cho ước lượng T3b.
    config = dataclasses.replace(RunConfig(num_nodes=8), topology=TopologyConfig("erdos_renyi", 1.0))

    t3b = describe.describe_topologies(config, seeds=1, matchings=4_000)["T3b ER gossip"]

    assert t3b["mixing_rate"] == pytest.approx(describe.gossip_mixing_rate(8), abs=0.01)
    assert t3b["messages_per_round"] == 8  # K_8 luôn có perfect matching được nhận tham lam


def test_rows_that_need_more_nodes_are_skipped_not_fatal() -> None:
    rows = describe.describe_topologies(RunConfig(num_nodes=2), seeds=2, matchings=10)

    assert rows["T1 ring"] == {"skipped": "ring cần N ≥ 3, nhận 2"}
    assert rows["T4 FC"]["spectral_gap"] == pytest.approx(1.0)


def test_the_kh_column_is_printed_only_for_the_kh_setting() -> None:
    rows = describe.describe_topologies(RunConfig(num_nodes=6), seeds=2, matchings=10)

    assert "0.033" not in "\n".join(describe.topology_table(rows, RunConfig(num_nodes=6)))


def test_the_command_prints_the_topology_table_next_to_kh_3_3(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(
        [
            "describe",
            "--config",
            str(ROOT / "configs" / "profiles" / "smoke.yaml"),
            "--seeds",
            "2",
            "--out",
            str(tmp_path),
        ]
    )

    printed = capsys.readouterr().out
    assert code == 0 and "T1 ring" in printed and "KH §3.3" in printed
    assert "T3 gossip" in json.loads((tmp_path / "describe.json").read_text(encoding="utf-8"))["topologies"]

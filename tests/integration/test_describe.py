"""`dfl describe` trên dữ liệu tổng hợp: mọi cách chia × seed, JSON đọc lại được, không NaN, CSV, và lệnh
`--check-req02` (T05a; KH §3.2; SPEC REQ-02, QS-11).
"""

import csv
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from dfl import cli, seeding
from dfl.config import PartitionConfig, load_config
from dfl.data import datasets
from dfl.data.heterogeneity import summarize
from dfl.data.partition import DirichletPartitioner, assign_positions
from dfl.seeding import Stream
from dfl.sim import describe

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

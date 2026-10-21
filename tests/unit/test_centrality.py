"""Test cho analysis/centrality.py: bậc từ manifest, node giữ lớp hiếm, độ đo và hiệu số ghép cặp của H9
(E7; ADR-014)."""

from pathlib import Path

import numpy as np
import pytest

from analysis import aggregate, centrality
from analysis.aggregate import AnalysisError
from tests.analysis_fakes import config, write_run

# Đồ thị 5 node: node 0 là hub (bậc 4), node 4 là lá (bậc 1).
EDGES = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3]]
COMMON = [10] * 5 + [0] * 5
HELD = [10] * 10


def e7_config(placement: str, seed: int = 0) -> dict[str, object]:
    return config(
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


def write_e7(root: Path, placement: str, seed: int, holder: int, acc: list[float]) -> None:
    counts = [HELD if i == holder else COMMON for i in range(5)]
    write_run(
        root,
        f"{placement}-{seed}",
        cfg=e7_config(placement, seed),
        graph_edges=EDGES,
        class_counts=counts,
        final_acc=acc,
    )


def runs(root: Path) -> list[aggregate.Run]:
    return sorted(
        (a.finished for a in aggregate.scan(root) if a.finished is not None), key=lambda r: r.run_id
    )


def test_degrees_come_from_the_manifest_edges(tmp_path: Path) -> None:
    write_e7(tmp_path, "hub", 0, holder=0, acc=[0.9] * 5)

    (run,) = runs(tmp_path)

    assert centrality.degrees(run).tolist() == [4, 2, 3, 2, 1]


def test_a_run_without_a_static_graph_is_refused(tmp_path: Path) -> None:
    write_run(tmp_path, "gossip", cfg=config(topology__kind="gossip"))

    (run,) = runs(tmp_path)

    with pytest.raises(AnalysisError, match="graph_edges"):
        centrality.degrees(run)


def test_holders_are_the_nodes_with_any_sample_of_classes_5_to_9(tmp_path: Path) -> None:
    write_e7(tmp_path, "edge", 0, holder=4, acc=[0.9] * 5)

    (run,) = runs(tmp_path)

    assert centrality.holders(run).tolist() == [False, False, False, False, True]


def test_the_h9_measure_averages_the_final_accuracy_of_non_holders_only(tmp_path: Path) -> None:
    write_e7(tmp_path, "hub", 0, holder=0, acc=[0.99, 0.8, 0.7, 0.6, 0.5])

    (run,) = runs(tmp_path)

    assert centrality.non_holder_accuracy(run) == pytest.approx(0.65)


def test_hub_minus_edge_is_paired_by_seed_and_skips_unpaired_seeds(tmp_path: Path) -> None:
    for seed, (hub, edge) in enumerate([(0.8, 0.6), (0.7, 0.65)]):
        write_e7(tmp_path, "hub", seed, holder=0, acc=[0.99] + [hub] * 4)
        write_e7(tmp_path, "edge", seed, holder=4, acc=[edge] * 4 + [0.99])
    write_e7(tmp_path, "hub", 7, holder=0, acc=[0.9] * 5)

    differences = centrality.hub_minus_edge(runs(tmp_path))

    assert set(differences) == {0, 1}
    assert differences[0] == pytest.approx(0.2) and differences[1] == pytest.approx(0.05)


def test_final_byte_load_is_the_last_node_bytes_without_the_virtual_server(tmp_path: Path) -> None:
    write_e7(tmp_path, "hub", 0, holder=0, acc=[0.9] * 5)

    (run,) = runs(tmp_path)

    load = centrality.final_byte_load(run)
    assert load.shape == (5,) and np.all(load == run.rounds[-1]["node_bytes_cum"][:5])

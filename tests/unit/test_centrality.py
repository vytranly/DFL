"""Test cho analysis/centrality.py: bậc từ manifest, node giữ lớp hiếm, độ đo và hiệu số ghép cặp của H9
(E7; ADR-014)."""

from pathlib import Path

import numpy as np
import pytest

from analysis import aggregate, centrality
from analysis.aggregate import AnalysisError
from tests.analysis_fakes import config, round_record, write_run

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


def test_js_mean_is_zero_for_identical_nodes_and_grows_with_rare_class_holders(tmp_path: Path) -> None:
    write_run(tmp_path, "same", cfg=config(num_nodes=5), class_counts=[[10] * 10] * 5)
    write_e7(tmp_path, "hub", 0, holder=0, acc=[0.9] * 5)
    by_id = {r.run_id: r for r in runs(tmp_path)}

    assert centrality.js_mean(by_id["same"]) == pytest.approx(0.0)
    assert 0 < centrality.js_mean(by_id["hub-0"]) < 1


# ------------------------------------------------------------------ bậc suy từ tải message (ADR-015)


def test_static_runs_without_graph_edges_get_their_degree_from_the_message_load(tmp_path: Path) -> None:
    # Ring N = 4: mỗi round mỗi node gửi 2 và nhận 2 message; fake ghi 160 message sau 20 round, 80 mỗi node.
    write_run(tmp_path, "ring", cfg=config())

    (run,) = runs(tmp_path)

    assert run.manifest["graph_edges"] is None
    assert centrality.degrees(run).tolist() == [2, 2, 2, 2]


def test_a_message_load_that_is_not_a_whole_degree_is_refused(tmp_path: Path) -> None:
    rounds = [round_record(100, 20, [0.5] * 4, node_messages_cum=[20, 20, 21, 20])]
    write_run(tmp_path, "odd", cfg=config(), rounds=rounds)

    (run,) = runs(tmp_path)

    with pytest.raises(AnalysisError, match="bậc"):
        centrality.degrees(run)


@pytest.mark.parametrize(
    ("degree", "kind"),
    [(1, "leaf"), (2, "leaf"), (3, "giữa"), (5, "giữa"), (6, "hub"), (19, "hub")],
)
def test_degree_classes_follow_the_hub_and_leaf_definitions_of_the_review(degree: int, kind: str) -> None:
    assert centrality.degree_class(degree) == kind


def test_hub_minus_leaf_compares_the_final_accuracy_of_hubs_and_leaves_in_one_run(tmp_path: Path) -> None:
    # H10 (E9, ADR-015): hình sao 8 node, hub (bậc 7) và 7 leaf (bậc 1).
    star = [[0, i] for i in range(1, 8)]
    cfg = config(num_nodes=8, topology={"kind": "barabasi_albert", "m": 1, "p": None})
    acc = [0.9, 0.5, 0.6, 0.7, 0.5, 0.6, 0.7, 0.6]
    write_run(tmp_path, "star", cfg=cfg, graph_edges=star, final_acc=acc)

    (run,) = runs(tmp_path)

    assert centrality.hub_minus_leaf(run) == pytest.approx(0.9 - 0.6)


def test_hub_minus_leaf_is_undefined_without_both_kinds_of_node(tmp_path: Path) -> None:
    write_run(tmp_path, "ring", cfg=config())

    (run,) = runs(tmp_path)

    assert centrality.hub_minus_leaf(run) is None


def test_a_run_written_before_adr_014_has_the_random_placement(tmp_path: Path) -> None:
    cfg = config()
    del cfg["partition"]["placement"]
    write_run(tmp_path, "old", cfg=cfg)

    (run,) = runs(tmp_path)

    assert centrality.placement(run) == "random"

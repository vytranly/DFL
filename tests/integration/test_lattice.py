"""Lattice (T7, ADR-015) qua sim.run: đồ thị trong manifest và bậc mà analysis đọc lại.

N = 9 (xuyến 3 × 3) trên dữ liệu tổng hợp của profile smoke; 1,800 ảnh chia cho 9 node vẫn trên min_size = 64.
"""

import dataclasses
import json
from pathlib import Path

import networkx as nx

from analysis import aggregate, centrality
from dfl.config import EvalConfig, TopologyConfig, load_config
from dfl.io import manifest
from dfl.sim import run

ROOT = Path(__file__).resolve().parents[2]


def test_a_lattice_run_writes_a_degree_four_torus_and_analysis_reads_it(tmp_path: Path) -> None:
    config = dataclasses.replace(
        load_config(ROOT / "configs" / "profiles" / "smoke.yaml"),
        num_nodes=9,
        local_steps=4,
        topology=TopologyConfig(kind="lattice"),
        eval=EvalConfig(subset_size=100, every_steps=None),
    )
    provenance = dataclasses.replace(manifest.current_provenance(), dirty=False)

    attempt = run.run(config, out=tmp_path, provenance=provenance)

    record = json.loads((attempt / "manifest.json").read_text(encoding="utf-8"))
    graph = nx.Graph(record["graph_edges"])
    assert graph.number_of_edges() == 18 and {d for _, d in graph.degree()} == {4}
    assert 0 < record["spectral_gap"] < 1
    (entry,) = aggregate.scan(tmp_path)
    assert entry.finished is not None
    assert set(centrality.degrees(entry.finished).tolist()) == {4}
    old = dataclasses.replace(entry.finished, manifest={**entry.finished.manifest, "graph_edges": None})
    assert set(centrality.degrees(old).tolist()) == {4}

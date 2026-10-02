"""Test tích hợp cho TorchDistTransport: 4 tiến trình thật, gloo trên 127.0.0.1 (T22a; ADR-010; INV-2, G7).

Mỗi tiến trình con là một node, gửi đúng một snapshot cho mỗi láng giềng mỗi round. Nó tự kiểm snapshot nhận
được (trùng từng bit với cái láng giềng đã gửi), rồi ghi số đếm ra file. Tiến trình cha so số đếm đó với
InMemoryTransport trên cùng đồ thị. Rendezvous dùng `file://`, nên không tranh cổng với tiến trình khác:
https://docs.pytorch.org/docs/2.14/distributed.html#torch.distributed.init_process_group
Tiến trình con khởi động bằng `spawn` và được đợi có hạn giờ (ProcessContext.join(timeout)), nên test không
bao giờ treo:
https://docs.pytorch.org/docs/2.14/multiprocessing.html#spawning-subprocesses
"""

import datetime
import json
import time
from pathlib import Path

import networkx as nx
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from dfl.network import topology as topo
from dfl.network.transport import InMemoryTransport
from dfl.sim.node import make_snapshot
from dfl.sim.torchdist import StateLayout, TorchDistTransport

WORLD = 4
ROUNDS = 2
TIMEOUT_S = 90
GRAPHS = {"ring": topo.ring(WORLD).graph_at(0), "fc": topo.fully_connected(WORLD).graph_at(0)}


def node_state(node_id: int, round_idx: int) -> dict[str, torch.Tensor]:
    """State tất định của một node ở một round, để người nhận tự tính lại được và so từng bit."""
    generator = torch.Generator().manual_seed(1000 * round_idx + node_id)
    return {
        "w": torch.randn((8, 3), generator=generator),
        "w64": torch.randn((5,), generator=generator, dtype=torch.float64),
        "steps": torch.tensor(round_idx * 10 + node_id, dtype=torch.int64),
    }


def worker(rank: int, init_method: str, out_dir: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=init_method, rank=rank, world_size=WORLD, timeout=datetime.timedelta(seconds=60)
    )
    report = {}
    layout = StateLayout.of(node_state(rank, 0))
    for offset, (name, graph) in enumerate(GRAPHS.items()):
        transport = TorchDistTransport(node_id=rank, num_nodes=WORLD, layout=layout)
        neighbors = sorted(graph.neighbors(rank))
        for local_round in range(ROUNDS):
            round_idx = offset * ROUNDS + local_round  # tag khác nhau giữa hai đồ thị
            transport.expect(round_idx, neighbors)
            message = make_snapshot(rank, round_idx, node_state(rank, round_idx))
            for dst in neighbors:
                transport.send(dst, message)
            inbox = transport.inbox(rank, round_idx)
            assert sorted(inbox) == neighbors
            for sender, received in inbox.items():
                assert (received.sender, received.round_idx) == (sender, round_idx)
                expected = node_state(sender, round_idx)
                assert all(torch.equal(received.state[k], expected[k]) for k in expected)
                assert all(received.state[k].dtype == expected[k].dtype for k in expected)
            transport.end_round(round_idx)
        counters = transport.counters()
        report[name] = {
            "messages": counters.messages_cum,
            "bytes": counters.bytes_cum,
            "load": counters.max_node_bytes_cum,
        }
    Path(out_dir, f"rank{rank}.json").write_text(json.dumps(report))
    dist.barrier()
    dist.destroy_process_group()


def in_memory_reference(graph: nx.Graph) -> InMemoryTransport:
    transport = InMemoryTransport(WORLD)
    for round_idx in range(ROUNDS):
        for node in range(WORLD):
            message = make_snapshot(node, round_idx, node_state(node, round_idx))
            for dst in sorted(graph.neighbors(node)):
                transport.send(dst, message)
        transport.end_round(round_idx)
    return transport


def test_four_processes_exchange_bitwise_exact_snapshots_and_count_like_the_in_memory_transport(
    tmp_path,
) -> None:
    # Arrange
    context = mp.spawn(
        worker, args=(f"file://{tmp_path}/rendezvous", str(tmp_path)), nprocs=WORLD, join=False
    )

    # Act: đợi có hạn giờ; quá hạn là treo (deadlock) và test fail.
    deadline = time.monotonic() + TIMEOUT_S
    while not context.join(timeout=1):
        if time.monotonic() > deadline:
            for process in context.processes:
                process.terminate()
            raise AssertionError(f"các tiến trình gloo không xong sau {TIMEOUT_S} s (treo?)")

    # Assert: mỗi node đếm đúng phần của nó; cộng lại bằng số đếm của InMemoryTransport.
    reports = [json.loads(Path(tmp_path, f"rank{rank}.json").read_text()) for rank in range(WORLD)]
    nbytes = make_snapshot(0, 0, node_state(0, 0)).nbytes
    for name, graph in GRAPHS.items():
        reference = in_memory_reference(graph)
        for rank, report in enumerate(reports):
            degree = graph.degree(rank)
            assert report[name]["messages"] == ROUNDS * degree
            assert report[name]["bytes"] == ROUNDS * degree * nbytes
            assert report[name]["load"] == reference.node_bytes()[rank]
        assert sum(report[name]["messages"] for report in reports) == reference.counters().messages_cum
        assert sum(report[name]["bytes"] for report in reports) == reference.counters().bytes_cum

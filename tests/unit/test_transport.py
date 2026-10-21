"""Test thành phần cho src/dfl/network/transport.py: Transport in-memory + CommLedger (ADR-002; INV-1, INV-2).

Bất biến về số message qua engine thuộc T10b; ở đây chỉ kiểm từng thao tác của Transport.
"""

import gc
import weakref

import networkx as nx
import pytest

from dfl import seeding
from dfl.contracts import CommCounters, Snapshot, Transport
from dfl.network import topology as topo
from dfl.network.transport import InMemoryTransport
from dfl.seeding import Stream


class _Payload:
    """Payload mờ: Transport không được đọc `state`, nên một đối tượng bất kỳ là đủ (và hỗ trợ weakref)."""


def _snap(sender: int, round_idx: int = 0, nbytes: int = 100) -> Snapshot:
    return Snapshot(sender=sender, round_idx=round_idx, state=_Payload(), nbytes=nbytes)


def test_in_memory_transport_satisfies_the_contract() -> None:
    assert isinstance(InMemoryTransport(3), Transport)


def _counters(
    messages: int, nbytes: int, node_messages: tuple[int, ...], node_bytes: tuple[int, ...]
) -> CommCounters:
    return CommCounters(
        messages_cum=messages,
        bytes_cum=nbytes,
        max_node_bytes_cum=max(node_bytes),
        node_messages_cum=node_messages,
        node_bytes_cum=node_bytes,
    )


def test_ledger_counts_messages_bytes_and_load_as_sent_plus_received() -> None:
    # Arrange
    transport = InMemoryTransport(3)

    # Act
    transport.send(1, _snap(0, nbytes=100))
    transport.send(0, _snap(1, nbytes=50))
    transport.send(1, _snap(2, nbytes=10))

    # Assert: node 1 nhận hai và gửi một message (ADR-013).
    assert transport.counters() == CommCounters(
        messages_cum=3,
        bytes_cum=160,
        max_node_bytes_cum=160,
        node_messages_cum=(2, 3, 1),
        node_bytes_cum=(150, 160, 10),
    )
    assert transport.node_bytes() == (150, 160, 10)


def test_counters_accumulate_across_rounds_and_survive_end_round() -> None:
    transport = InMemoryTransport(2)
    transport.send(1, _snap(0, round_idx=0))
    transport.end_round(0)

    transport.send(0, _snap(1, round_idx=1))

    assert transport.counters() == _counters(2, 200, (2, 2), (200, 200))


def test_self_send_is_rejected_and_not_counted() -> None:
    transport = InMemoryTransport(3)

    with pytest.raises(ValueError, match="tự gửi"):
        transport.send(1, _snap(1))

    assert transport.counters() == _counters(0, 0, (0, 0, 0), (0, 0, 0))


@pytest.mark.parametrize(("dst", "sender"), [(3, 0), (-1, 0), (0, 3), (0, -1)])
def test_ids_outside_the_nodes_are_rejected_without_a_server(dst: int, sender: int) -> None:
    with pytest.raises(ValueError, match="0..N"):
        InMemoryTransport(3).send(dst, _snap(sender))


def test_virtual_server_has_id_n_and_its_load_is_counted() -> None:
    # Đường server của B3/E1c: N lên, N xuống mỗi round, tải lớn nhất là của server (SPEC REQ-05).
    n = 4
    transport = InMemoryTransport(n, with_server=True)

    for i in range(n):
        transport.send(n, _snap(i))
    for i in range(n):
        transport.send(i, _snap(n))

    assert transport.counters() == _counters(2 * n, 800, (2, 2, 2, 2, 2 * n), (200, 200, 200, 200, 800))
    assert transport.node_bytes() == (200, 200, 200, 200, 800)


def test_payload_is_not_copied() -> None:
    transport = InMemoryTransport(3)
    message = _snap(0)

    transport.send(1, message)
    transport.send(2, message)

    assert transport.inbox(1, 0)[0] is message
    assert transport.inbox(2, 0)[0] is message


def test_inbox_holds_only_messages_for_that_node_and_round() -> None:
    transport = InMemoryTransport(3)
    transport.send(1, _snap(0, round_idx=0))
    transport.send(2, _snap(0, round_idx=0))
    transport.send(1, _snap(2, round_idx=1))

    assert set(transport.inbox(1, 0)) == {0}
    assert set(transport.inbox(1, 1)) == {2}
    assert set(transport.inbox(0, 0)) == set()


def test_inbox_is_read_only_and_reading_it_does_not_change_counters() -> None:
    transport = InMemoryTransport(2)
    transport.send(1, _snap(0))
    before = transport.counters()

    inbox = transport.inbox(1, 0)

    with pytest.raises(TypeError):
        inbox[5] = _snap(0)
    assert transport.counters() == before


def test_sending_twice_from_the_same_sender_in_a_round_is_rejected() -> None:
    # Inbox lưu một snapshot cho mỗi người gửi; gửi lần hai sẽ ghi đè mà vẫn bị tính phí.
    transport = InMemoryTransport(2)
    transport.send(1, _snap(0))

    with pytest.raises(ValueError, match="đã gửi"):
        transport.send(1, _snap(0))
    assert transport.counters().messages_cum == 1


def test_end_round_releases_snapshot_references() -> None:
    transport = InMemoryTransport(2)
    message = _snap(0)
    payload_ref = weakref.ref(message.state)
    transport.send(1, message)
    del message

    transport.end_round(0)
    gc.collect()

    assert payload_ref() is None


def test_using_a_round_after_end_round_raises_runtime_error() -> None:
    transport = InMemoryTransport(2)
    transport.end_round(0)

    with pytest.raises(RuntimeError):
        transport.send(1, _snap(0, round_idx=0))
    with pytest.raises(RuntimeError):
        transport.inbox(1, 0)


def test_negative_nbytes_is_rejected() -> None:
    with pytest.raises(ValueError, match="nbytes"):
        InMemoryTransport(2).send(1, _snap(0, nbytes=-1))


@pytest.mark.parametrize(
    ("graph", "expected"),
    [
        (topo.ring(20).graph_at(0), 40),
        (topo.fully_connected(20).graph_at(0), 380),
        (topo.gossip(20, lambda r: seeding.numpy_rng(0, Stream.GOSSIP, r)).graph_at(0), 20),
    ],
    ids=["ring", "fc", "gossip"],
)
def test_one_message_per_directed_edge_gives_the_expected_count(graph: nx.Graph, expected: int) -> None:
    transport = InMemoryTransport(20)

    for i, j in graph.edges():
        transport.send(j, _snap(i))
        transport.send(i, _snap(j))

    assert transport.counters().messages_cum == expected


@pytest.mark.parametrize(
    "graph",
    [
        topo.ring(20).graph_at(0),
        topo.fully_connected(20).graph_at(0),
        topo.erdos_renyi(20, 0.3, seeding.numpy_rng(0, Stream.GRAPH)).graph_at(0),
        topo.gossip(20, lambda r: seeding.numpy_rng(0, Stream.GOSSIP, r)).graph_at(0),
    ],
    ids=["ring", "fc", "er", "gossip"],
)
def test_node_load_is_twice_the_degree_and_matches_the_totals(graph: nx.Graph) -> None:
    # ADR-013: mỗi message được tính cho cả người gửi và người nhận, nên tổng tải message gấp đôi số message,
    # và tải byte lớn nhất là max_node_bytes_cum.
    transport = InMemoryTransport(20)

    for i, j in graph.edges():
        transport.send(j, _snap(i, nbytes=7))
        transport.send(i, _snap(j, nbytes=7))

    counters = transport.counters()
    assert counters.node_messages_cum == tuple(2 * graph.degree(i) for i in range(20))
    assert counters.node_bytes_cum == tuple(7 * m for m in counters.node_messages_cum)
    assert sum(counters.node_messages_cum) == 2 * counters.messages_cum
    assert max(counters.node_bytes_cum) == counters.max_node_bytes_cum


def test_transport_needs_at_least_one_node() -> None:
    with pytest.raises(ValueError, match="num_nodes"):
        InMemoryTransport(0)

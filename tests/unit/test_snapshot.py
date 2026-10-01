"""Test cho make_snapshot trong src/dfl/sim/node.py (KH §3.4 bước 2; INV-3; SPEC REQ-05)."""

import dataclasses

import pytest
import torch

from dfl.contracts import Snapshot
from dfl.network.transport import InMemoryTransport
from dfl.sim.node import make_snapshot

# Shape tham số của SmallCNN theo ADR-004; tổng 46,730 phần tử.
SMALLCNN_SHAPES = {
    "conv1.weight": (16, 1, 5, 5),
    "conv1.bias": (16,),
    "conv2.weight": (32, 16, 5, 5),
    "conv2.bias": (32,),
    "fc1.weight": (64, 512),
    "fc1.bias": (64,),
    "fc2.weight": (10, 64),
    "fc2.bias": (10,),
}


def _smallcnn_state() -> dict[str, torch.Tensor]:
    return {name: torch.zeros(shape) for name, shape in SMALLCNN_SHAPES.items()}


def test_smallcnn_state_snapshot_has_4_times_46730_bytes() -> None:
    snapshot = make_snapshot(sender=3, round_idx=7, state=_smallcnn_state())

    assert isinstance(snapshot, Snapshot)
    assert (snapshot.sender, snapshot.round_idx) == (3, 7)
    assert snapshot.nbytes == 4 * 46_730


def test_nbytes_is_computed_from_the_actual_tensors() -> None:
    state = {"a": torch.zeros(10, dtype=torch.float64), "b": torch.zeros(3, dtype=torch.int8)}

    assert make_snapshot(0, 0, state).nbytes == 10 * 8 + 3 * 1


def test_bytes_through_transport_equal_messages_times_snapshot_nbytes() -> None:
    snapshot = make_snapshot(0, 0, _smallcnn_state())
    transport = InMemoryTransport(5)

    for dst in range(1, 5):
        transport.send(dst, snapshot)

    assert transport.counters().bytes_cum == 4 * (4 * 46_730)


def test_snapshot_does_not_share_storage_with_the_model() -> None:
    # state_dict() trả về tham chiếu tới tham số; snapshot phải là bản sao riêng (G22).
    model = torch.nn.Linear(3, 2)
    snapshot = make_snapshot(0, 0, model.state_dict())
    before = snapshot.state["weight"].clone()

    with torch.no_grad():
        model.weight.add_(1.0)

    torch.testing.assert_close(snapshot.state["weight"], before)
    assert snapshot.state["weight"].data_ptr() != model.weight.data_ptr()


def test_snapshot_tensors_are_detached_from_autograd() -> None:
    weight = torch.ones(2, requires_grad=True)

    snapshot = make_snapshot(0, 0, {"w": weight * 2})

    assert not snapshot.state["w"].requires_grad
    assert snapshot.state["w"].grad_fn is None


def test_snapshot_is_frozen() -> None:
    snapshot = make_snapshot(0, 0, {"w": torch.zeros(1)})

    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.nbytes = 0

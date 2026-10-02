"""Test cho phần không cần tiến trình của src/dfl/sim/torchdist.py: mã hóa state và phép kiểm của Transport
(T22a; ADR-010; INV-1, INV-2, G7).
"""

import pytest
import torch

from dfl.contracts import Snapshot, Transport
from dfl.sim.node import make_snapshot
from dfl.sim.torchdist import StateLayout, TorchDistTransport, decode_state, encode_state


def mixed_state() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(3)
    return {
        "w": torch.randn((4, 5), generator=generator),
        "w64": torch.randn((3,), generator=generator, dtype=torch.float64),
        "steps": torch.tensor(7, dtype=torch.int64),
        "mask": torch.tensor([True, False, True]),
        "empty": torch.zeros(0),
        "scalar": torch.tensor(2.5),
        "transposed": torch.randn((3, 2), generator=generator).t(),  # không liền bộ nhớ
    }


def test_encode_then_decode_is_bitwise_identical_for_every_dtype() -> None:
    # Arrange
    state = mixed_state()
    layout = StateLayout.of(state)

    # Act
    decoded = decode_state(encode_state(state, layout), layout)

    # Assert
    assert list(decoded) == list(state)
    for name, tensor in state.items():
        assert decoded[name].dtype == tensor.dtype
        assert decoded[name].shape == tensor.shape
        assert torch.equal(decoded[name], tensor)


def test_encoded_length_equals_the_snapshot_nbytes() -> None:
    state = mixed_state()
    layout = StateLayout.of(state)

    buffer = encode_state(state, layout)

    assert buffer.dtype == torch.uint8
    assert buffer.numel() == layout.nbytes == make_snapshot(0, 0, state).nbytes


def test_decoded_tensors_own_their_memory() -> None:
    state = {"w": torch.ones(4)}
    layout = StateLayout.of(state)
    buffer = encode_state(state, layout)

    decoded = decode_state(buffer, layout)
    buffer.zero_()

    assert torch.equal(decoded["w"], torch.ones(4))


@pytest.mark.parametrize(
    "other",
    [
        {"v": torch.ones(4)},
        {"w": torch.ones(5)},
        {"w": torch.ones(4, dtype=torch.float64)},
        {"w": torch.ones(4), "extra": torch.ones(1)},
    ],
    ids=["name", "shape", "dtype", "extra-key"],
)
def test_encode_rejects_a_state_that_does_not_match_the_layout(other: dict[str, torch.Tensor]) -> None:
    layout = StateLayout.of({"w": torch.ones(4)})

    with pytest.raises(ValueError, match="layout"):
        encode_state(other, layout)


def test_decode_rejects_a_buffer_of_the_wrong_size() -> None:
    layout = StateLayout.of({"w": torch.ones(4)})

    with pytest.raises(ValueError, match="byte"):
        decode_state(torch.zeros(15, dtype=torch.uint8), layout)


def make_transport() -> TorchDistTransport:
    return TorchDistTransport(node_id=1, num_nodes=4, layout=StateLayout.of({"w": torch.ones(4)}))


def snapshot(sender: int, round_idx: int = 0) -> Snapshot:
    return make_snapshot(sender, round_idx, {"w": torch.ones(4)})


def test_transport_satisfies_the_contract_without_a_process_group() -> None:
    assert isinstance(make_transport(), Transport)


@pytest.mark.parametrize(
    ("dst", "message", "match"),
    [
        (1, snapshot(1), "tự gửi"),
        (2, snapshot(3), "người gửi"),
        (4, snapshot(1), "0..N-1"),
        (-1, snapshot(1), "0..N-1"),
    ],
    ids=["self-send", "foreign-sender", "dst-too-large", "dst-negative"],
)
def test_send_rejects_invalid_messages_before_touching_the_network(
    dst: int, message: Snapshot, match: str
) -> None:
    transport = make_transport()

    with pytest.raises(ValueError, match=match):
        transport.send(dst, message)

    assert transport.counters().messages_cum == 0


def test_a_process_can_only_read_its_own_inbox() -> None:
    with pytest.raises(ValueError, match="INV-1"):
        make_transport().inbox(2, 0)


def test_expect_rejects_self_and_out_of_range_senders() -> None:
    transport = make_transport()

    with pytest.raises(ValueError):
        transport.expect(0, [1])
    with pytest.raises(ValueError):
        transport.expect(0, [7])

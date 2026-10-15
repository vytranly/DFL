"""Hạ tầng cho test tích hợp của engine (T10a–c; KH §5.1): MLP tí hon, dữ liệu tổng hợp 4 node, trainer theo
Protocol LocalTrainer, và Transport ghi lại mọi lần gửi và đọc inbox.

Không gì ở đây dùng RNG toàn cục của torch (fixture autouse ở tests/conftest.py kiểm): module được tạo bằng
`skip_init` rồi điền giá trị từ `torch.Generator` riêng; minibatch lấy bằng `torch.randint(generator=…)`.
torch.nn.utils.skip_init: https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.skip_init.html
torch.randn: https://docs.pytorch.org/docs/2.14/generated/torch.randn.html
torch.randint: https://docs.pytorch.org/docs/2.14/generated/torch.randint.html
torch.autograd.grad: https://docs.pytorch.org/docs/2.14/generated/torch.autograd.grad.html
Fixture trả về hàm dựng ("factories as fixtures"):
https://docs.pytest.org/en/stable/how-to/fixtures.html#factories-as-fixtures
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
import pytest
import torch

from dfl import seeding
from dfl.contracts import CommCounters, NodeDataset, Snapshot
from dfl.metrics.classification import ClassificationEvaluator
from dfl.network.transport import InMemoryTransport
from dfl.seeding import Stream
from dfl.sim.node import Node

NUM_CLASSES = 10
IMAGE_SHAPE = (1, 28, 28)
HIDDEN = 4
DATA_SEED = 1234
# id riêng cho tập đánh giá, để nhiễu của nó không trùng nhiễu của node nào.
EVAL_SET_ID = 1000


class TinyMLP(torch.nn.Module):
    """784 → 4 → 10, không BatchNorm; tùy chọn thêm một buffer float và một buffer int64 để kiểm G3."""

    def __init__(
        self,
        generator: torch.Generator,
        *,
        dtype: torch.dtype,
        float_buffer: float | None = None,
        int_buffer: int | None = None,
    ) -> None:
        super().__init__()
        size = IMAGE_SHAPE[1] * IMAGE_SHAPE[2]
        self.fc1 = torch.nn.utils.skip_init(torch.nn.Linear, size, HIDDEN, dtype=dtype)
        self.fc2 = torch.nn.utils.skip_init(torch.nn.Linear, HIDDEN, NUM_CLASSES, dtype=dtype)
        with torch.no_grad():
            for param in self.parameters():
                param.copy_(0.1 * torch.randn(param.shape, generator=generator, dtype=dtype))
        if float_buffer is not None:
            self.register_buffer("float_buf", torch.full((3,), float_buffer, dtype=dtype))
        if int_buffer is not None:
            self.register_buffer("int_buf", torch.tensor(int_buffer, dtype=torch.int64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x.flatten(1))))


def synthetic_dataset(node_id: int, size: int, classes: Sequence[int]) -> NodeDataset:
    """Dữ liệu cùng shape với MNIST; mỗi lớp lệch khỏi nhiễu theo một mẫu riêng, chung cho mọi node."""
    prototypes = torch.randn((NUM_CLASSES, *IMAGE_SHAPE), generator=torch.Generator().manual_seed(DATA_SEED))
    noise = torch.Generator().manual_seed(DATA_SEED + 1 + node_id)
    y = torch.tensor([classes[k % len(classes)] for k in range(size)], dtype=torch.int64)
    x = prototypes[y] + torch.randn((size, *IMAGE_SHAPE), generator=noise)
    return NodeDataset(node_id=node_id, x=x, y=y)


@dataclass(frozen=True)
class TrainerCall:
    node_id: int
    loss_scale: float
    # Bản clone + detach của tham số lúc bắt đầu gọi: đúng anchor của số hạng prox.
    anchor: dict[str, torch.Tensor]


@dataclass
class SgdTrainer:
    """LocalTrainer cho test: τ bước SGD thuần, minibatch có hoàn lại, prox μ, hệ số c_i (theo contracts.py).

    Ghi lại mọi lần gọi, để test kiểm thứ tự duyệt node, anchor và c_i mà engine truyền vào. Nếu model có
    buffer `int_buf` thì cộng τ vào nó, như một bộ đếm bước.
    """

    tau: int
    lr: float
    mu: float = 0.0
    batch_size: int = 8
    calls: list[TrainerCall] = field(default_factory=list)

    def __call__(
        self, model: torch.nn.Module, data: NodeDataset, *, loss_scale: float, generator: torch.Generator
    ) -> float:
        params = dict(model.named_parameters())
        anchor = {name: param.detach().clone() for name, param in params.items()}
        self.calls.append(TrainerCall(data.node_id, loss_scale, anchor))
        dtype = next(iter(params.values())).dtype
        total = 0.0
        for _ in range(self.tau):
            idx = torch.randint(len(data.y), (self.batch_size,), generator=generator)
            logits = model(data.x[idx].to(dtype))
            loss = loss_scale * torch.nn.functional.cross_entropy(logits, data.y[idx])
            if self.mu:
                loss = loss + self.mu / 2 * sum(((p - anchor[name]) ** 2).sum() for name, p in params.items())
            grads = torch.autograd.grad(loss, list(params.values()))
            with torch.no_grad():
                for param, grad in zip(params.values(), grads, strict=True):
                    param.sub_(self.lr * grad)
            total += float(loss.detach())
        if hasattr(model, "int_buf"):
            model.int_buf += self.tau
        return total / self.tau


@dataclass(frozen=True)
class Sent:
    dst: int
    message: Snapshot
    # Bản clone payload lúc gửi, để kiểm snapshot không bị sửa sau đó (T-INV-4).
    state_at_send: dict[str, torch.Tensor]


class RecordingTransport:
    """Bọc InMemoryTransport: ghi mọi lần gửi (kèm bản clone payload lúc gửi) và mọi lần đọc inbox.

    `tamper=True`: inbox giao bản sao trong đó mọi tensor float được thay bằng hằng số sender + 1, nên kết quả
    tổng hợp lộ ra engine có dùng đúng nội dung inbox hay đọc thẳng model của node khác (INV-1).
    `reverse_inbox=True`: inbox trả về theo thứ tự chèn ngược lại.
    `with_server=True`: thêm server ảo id N cho đường server, như InMemoryTransport.
    """

    def __init__(
        self, num_nodes: int, *, tamper: bool = False, reverse_inbox: bool = False, with_server: bool = False
    ) -> None:
        self.inner = InMemoryTransport(num_nodes, with_server=with_server)
        self.tamper = tamper
        self.reverse_inbox = reverse_inbox
        self.sent: list[Sent] = []
        self.inbox_reads: list[tuple[int, int]] = []

    def send(self, dst: int, message: Snapshot) -> None:
        self.sent.append(Sent(dst, message, {name: t.clone() for name, t in message.state.items()}))
        self.inner.send(dst, message)

    def inbox(self, dst: int, round_idx: int) -> Mapping[int, Snapshot]:
        self.inbox_reads.append((dst, round_idx))
        items = list(self.inner.inbox(dst, round_idx).items())
        if self.reverse_inbox:
            items.reverse()
        if self.tamper:
            items = [(sender, _constant_payload(message)) for sender, message in items]
        return MappingProxyType(dict(items))

    def end_round(self, round_idx: int) -> None:
        self.inner.end_round(round_idx)

    def counters(self) -> CommCounters:
        return self.inner.counters()


def _constant_payload(message: Snapshot) -> Snapshot:
    state = {
        name: torch.full_like(t, message.sender + 1) if t.is_floating_point() else t
        for name, t in message.state.items()
    }
    return Snapshot(sender=message.sender, round_idx=message.round_idx, state=state, nbytes=message.nbytes)


@pytest.fixture
def make_nodes() -> Callable[..., list[Node]]:
    """Dựng N node theo id; mặc định 4 node, mỗi node 64 mẫu của 3 lớp (Non-IID), cùng w⁰ (KH §3.5)."""

    def make(
        sizes: Sequence[int] = (64, 64, 64, 64),
        *,
        seed: int = 0,
        dtype: torch.dtype = torch.float32,
        shared_init: bool = True,
        loss_scales: Sequence[float] | None = None,
        buffers: bool = False,
    ) -> list[Node]:
        nodes = []
        for i, size in enumerate(sizes):
            init_key = () if shared_init else (i,)
            model = TinyMLP(
                seeding.torch_generator(seed, Stream.INIT, *init_key),
                dtype=dtype,
                float_buffer=1.0 + i if buffers else None,
                int_buffer=100 * i if buffers else None,
            )
            classes = [(3 * i + k) % NUM_CLASSES for k in range(3)]
            nodes.append(
                Node(
                    node_id=i,
                    model=model,
                    data=synthetic_dataset(i, size, classes),
                    generator=seeding.torch_generator(seed, Stream.BATCH, i),
                    loss_scale=1.0 if loss_scales is None else loss_scales[i],
                )
            )
        return nodes

    return make


@pytest.fixture
def sgd_trainer() -> Callable[..., SgdTrainer]:
    return SgdTrainer


@pytest.fixture
def recording_transport() -> Callable[..., RecordingTransport]:
    return RecordingTransport


def _eval_set() -> NodeDataset:
    return synthetic_dataset(EVAL_SET_ID, 200, range(NUM_CLASSES))


@pytest.fixture
def eval_set() -> NodeDataset:
    """Tập đánh giá tổng hợp 200 mẫu, đủ 10 lớp; cùng tập mà make_evaluator dùng."""
    return _eval_set()


@pytest.fixture
def make_evaluator() -> Callable[..., ClassificationEvaluator]:
    """Evaluator thật (T09b) trên tập đánh giá tổng hợp, P_i lấy từ dữ liệu của từng node."""

    def make(
        nodes: Sequence[Node], *, objective_weights: np.ndarray | None = None
    ) -> ClassificationEvaluator:
        eval_set = _eval_set()
        counts = np.stack([np.bincount(node.data.y.numpy(), minlength=NUM_CLASSES) for node in nodes])
        pi = np.full(len(nodes), 1 / len(nodes)) if objective_weights is None else objective_weights
        dtype = next(nodes[0].model.parameters()).dtype
        return ClassificationEvaluator(
            TinyMLP(torch.Generator().manual_seed(0), dtype=dtype),
            eval_set.x,
            eval_set.y,
            label_distributions=counts / counts.sum(axis=1, keepdims=True),
            objective_weights=pi,
        )

    return make

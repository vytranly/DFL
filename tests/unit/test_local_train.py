"""Test cho src/dfl/learning/local_train.py: τ bước SGD thuần, minibatch có hoàn lại, prox μ, hệ số c_i
(T09a; KH §3.4; ADR-004, ADR-006; T-INV-5, T-INV-7; G6).

Oracle của "SGD thường" là torch.optim.SGD (momentum 0, weight decay 0, vòng for, không foreach), chạy trên
cùng minibatch nhờ một Generator sinh đôi:
https://docs.pytorch.org/docs/2.14/generated/torch.optim.SGD.html
"""

import math
import statistics
from collections.abc import Callable

import pytest
import torch

from dfl import seeding
from dfl.contracts import NodeDataset
from dfl.learning.local_train import SgdTrainer, make_anchor, prox_term
from dfl.learning.models import make_smallcnn
from dfl.seeding import Stream

LR = 0.1


def dataset(size: int, *, label: int | None = None) -> NodeDataset:
    generator = torch.Generator().manual_seed(5)
    x = torch.randn((size, 1, 28, 28), generator=generator)
    y = torch.full((size,), label) if label is not None else torch.arange(size) % 10
    return NodeDataset(node_id=0, x=x, y=y)


def smallcnn() -> torch.nn.Module:
    return make_smallcnn(seeding.torch_generator(0, Stream.INIT))


def batches() -> torch.Generator:
    return seeding.torch_generator(0, Stream.BATCH, 0)


def reference_sgd(
    model: torch.nn.Module,
    data: NodeDataset,
    *,
    tau: int,
    batch_size: int,
    loss_scale: float,
    mu: float,
    generator: torch.Generator,
) -> list[float]:
    """SGD của PyTorch trên c·CE + μ/2 · Σ‖w − w_đầu‖² (anchor: tham số lúc bắt đầu); trả về mất mát
    từng bước."""
    params = list(model.parameters())
    start = [p.detach().clone() for p in params]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=0, weight_decay=0, foreach=False)
    losses = []
    for _ in range(tau):
        idx = torch.randint(len(data.y), (batch_size,), generator=generator)
        loss = loss_scale * torch.nn.functional.cross_entropy(model(data.x[idx]), data.y[idx])
        if mu:
            loss = loss + 0.5 * mu * sum(torch.sum((p - s) ** 2) for p, s in zip(params, start, strict=True))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


@pytest.mark.parametrize(("size", "loss_scale"), [(64, 1.0), (8, 2.5)], ids=["c-1", "c-2.5-batch-gt-data"])
def test_t_inv_7_mu_zero_is_plain_sgd_step_for_step(size: int, loss_scale: float) -> None:
    # Arrange: hai model và hai Generator sinh đôi. size = 8 < B = 32 chỉ chạy được khi lấy có hoàn lại.
    data = dataset(size)
    expected_model, expected_batches = smallcnn(), batches()
    expected_losses = reference_sgd(
        expected_model, data, tau=5, batch_size=32, loss_scale=loss_scale, mu=0.0, generator=expected_batches
    )
    model, generator = smallcnn(), batches()

    # Act
    loss = SgdTrainer(tau=5, lr=LR)(model, data, loss_scale=loss_scale, generator=generator)

    # Assert: trùng từng bit, cùng số minibatch đã rút, và trả về mất mát trung bình của τ bước.
    expected = expected_model.state_dict()
    assert all(torch.equal(tensor, expected[name]) for name, tensor in model.state_dict().items())
    assert torch.equal(generator.get_state(), expected_batches.get_state())
    assert loss == pytest.approx(statistics.fmean(expected_losses), rel=1e-12)


def test_t_inv_7_prox_gradient_is_mu_times_the_distance_to_the_anchor() -> None:
    # Arrange: anchor lấy lúc đầu, rồi dời tham số đi một đoạn ngẫu nhiên.
    model = smallcnn()
    params = list(model.parameters())
    original = [p.detach().clone() for p in params]
    anchor = make_anchor(model)
    noise = torch.Generator().manual_seed(9)
    with torch.no_grad():
        for p in params:
            p.add_(torch.randn(p.shape, generator=noise))

    # Act
    term = prox_term(params, anchor, 0.7)
    grads = torch.autograd.grad(term, params)

    # Assert: gradient μ(w − anchor), giá trị μ/2 · Σ‖w − anchor‖², anchor giữ nguyên giá trị lúc lấy.
    for p, start, grad in zip(params, original, grads, strict=True):
        torch.testing.assert_close(grad, 0.7 * (p.detach() - start))
    distance = sum(
        float(((p.detach() - start) ** 2).sum()) for p, start in zip(params, original, strict=True)
    )
    assert term.item() == pytest.approx(0.35 * distance, rel=1e-5)
    assert all(torch.equal(a, start) for a, start in zip(anchor, original, strict=True))


def test_t_inv_7_anchor_is_a_detached_clone_without_gradient() -> None:
    model = smallcnn()

    anchor = make_anchor(model)

    for a, p in zip(anchor, model.parameters(), strict=True):
        assert torch.equal(a, p.detach())
        assert not a.requires_grad and a.grad_fn is None
        assert a.untyped_storage().data_ptr() != p.untyped_storage().data_ptr()


def test_t_inv_7_prox_training_pulls_towards_the_parameters_at_the_start_of_the_call() -> None:
    # Arrange: μ lớn để số hạng prox đổi kết quả rõ rệt so với μ = 0 sau vài bước.
    data = dataset(64)
    expected_model = smallcnn()
    expected_losses = reference_sgd(
        expected_model, data, tau=5, batch_size=32, loss_scale=1.0, mu=5.0, generator=batches()
    )
    plain_model = smallcnn()
    reference_sgd(plain_model, data, tau=5, batch_size=32, loss_scale=1.0, mu=0.0, generator=batches())
    model = smallcnn()

    # Act
    loss = SgdTrainer(tau=5, lr=LR, mu=5.0)(model, data, loss_scale=1.0, generator=batches())

    # Assert: khớp oracle có anchor cố định từ đầu lời gọi; khác hẳn SGD không prox.
    expected, plain = expected_model.state_dict(), plain_model.state_dict()
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, expected[name])
    assert max(float((model.state_dict()[n] - plain[n]).abs().max()) for n in plain) > 1e-3
    assert loss == pytest.approx(statistics.fmean(expected_losses), rel=1e-6)


def test_local_training_does_not_touch_the_global_rng() -> None:
    model, data = smallcnn(), dataset(64)
    before = torch.get_rng_state()

    SgdTrainer(tau=3, lr=LR, mu=0.1)(model, data, loss_scale=1.0, generator=batches())

    assert torch.equal(torch.get_rng_state(), before)


class Fuse(torch.nn.Module):
    """SmallCNN; từ lần forward thứ `at + 1`, logit lớp 0 thành `bad`, nên với nhãn 0 mất mát thành
    inf hay NaN."""

    def __init__(self, at: int, bad: float) -> None:
        super().__init__()
        self.inner = smallcnn()
        self.at = at
        self.bad = bad
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.inner(x)
        self.calls += 1
        if self.calls > self.at:
            offset = torch.zeros(logits.shape[1])
            offset[0] = self.bad
            logits = logits + offset
        return logits


@pytest.mark.parametrize(
    ("bad", "is_returned"),
    [(-math.inf, lambda loss: loss == math.inf), (math.nan, math.isnan)],
    ids=["inf", "nan"],
)
def test_a_non_finite_loss_is_returned_at_once_without_updating_or_running_more_steps(
    bad: float, is_returned: Callable[[float], bool]
) -> None:
    # Arrange: mất mát hỏng ở bước thứ 3 (đếm từ 1); model sinh đôi chỉ chạy 2 bước lành.
    data = dataset(64, label=0)
    model = Fuse(at=2, bad=bad)
    twin = smallcnn()
    SgdTrainer(tau=2, lr=LR)(twin, data, loss_scale=1.0, generator=batches())

    # Act
    loss = SgdTrainer(tau=10, lr=LR)(model, data, loss_scale=1.0, generator=batches())

    # Assert: trả về đúng giá trị hỏng, dừng ở bước 3, và bước 3 không cập nhật tham số.
    assert is_returned(loss)
    assert model.calls == 3
    expected = twin.state_dict()
    assert all(torch.equal(tensor, expected[name]) for name, tensor in model.inner.state_dict().items())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"tau": 0}, "tau"),
        ({"tau": True}, "tau"),
        ({"tau": 2.0}, "tau"),
        ({"lr": 0.0}, "lr"),
        ({"lr": -0.1}, "lr"),
        ({"lr": math.nan}, "lr"),
        ({"mu": -0.1}, "mu"),
        ({"mu": math.inf}, "mu"),
        ({"batch_size": 0}, "batch_size"),
        ({"batch_size": 1.5}, "batch_size"),
    ],
    ids=["tau-0", "tau-bool", "tau-float", "lr-0", "lr-neg", "lr-nan", "mu-neg", "mu-inf", "b-0", "b-float"],
)
def test_invalid_hyperparameters_raise_value_error(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SgdTrainer(**{"tau": 1, "lr": LR, **kwargs})


def test_the_trainer_trains_on_the_augmented_batch_drawn_after_the_indices() -> None:
    # Augmentation giả (ADR-012): ghi lại batch và Generator nhận được, rồi trả ảnh 0, để thấy model học trên
    # ảnh đã biến đổi, với cùng Generator batch của node.
    model, data, generator = smallcnn(), dataset(64), batches()
    seen: list[tuple[torch.Tensor, torch.Generator]] = []

    def zero_out(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
        seen.append((x.clone(), g))
        return torch.zeros_like(x)

    loss = SgdTrainer(tau=1, lr=LR, augment=zero_out)(model, data, loss_scale=1.0, generator=generator)

    idx = torch.randint(64, (32,), generator=batches())
    expected = torch.nn.functional.cross_entropy(smallcnn()(torch.zeros((32, 1, 28, 28))), data.y[idx])
    assert len(seen) == 1 and torch.equal(seen[0][0], data.x[idx]) and seen[0][1] is generator
    assert loss == expected.item()

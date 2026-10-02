"""Test cho src/dfl/learning/models.py: SmallCNN 46,730 tham số và thang M0–M3 của E8, không BatchNorm hay
Dropout, w⁰ từ luồng init (T09a, T09c; ADR-004, ADR-011; SPEC §6; T-INV-5).
"""

import math

import pytest
import torch

from dfl import config, seeding
from dfl.learning.models import MODEL_CLASSES, SmallCNN, build_model, init_from_stream, make_smallcnn
from dfl.seeding import Stream
from dfl.sim.engine import BATCH_NORMS, DROPOUTS


def test_smallcnn_has_46730_parameters_and_no_batchnorm_or_dropout() -> None:
    model = make_smallcnn(seeding.torch_generator(0, Stream.INIT))

    assert sum(p.numel() for p in model.parameters()) == 46_730
    assert not any(isinstance(module, BATCH_NORMS + DROPOUTS) for module in model.modules())
    assert list(model.buffers()) == []  # không có thống kê chạy nào để bị trộn giữa các node (G3)


def test_forward_maps_mnist_images_to_ten_logits() -> None:
    model = make_smallcnn(seeding.torch_generator(0, Stream.INIT))

    logits = model(torch.zeros((7, 1, 28, 28)))

    assert logits.shape == (7, 10)


def test_building_the_model_does_not_touch_the_global_rng() -> None:
    before = torch.get_rng_state()

    make_smallcnn(seeding.torch_generator(0, Stream.INIT))

    assert torch.equal(torch.get_rng_state(), before)


def test_same_seed_gives_the_same_w0_and_other_seeds_differ_in_every_tensor() -> None:
    a = make_smallcnn(seeding.torch_generator(0, Stream.INIT)).state_dict()
    again = make_smallcnn(seeding.torch_generator(0, Stream.INIT)).state_dict()
    b = make_smallcnn(seeding.torch_generator(1, Stream.INIT)).state_dict()

    assert all(torch.equal(a[name], again[name]) for name in a)
    assert all(not torch.equal(a[name], b[name]) for name in a)


def test_w0_follows_pytorch_default_bounds_of_one_over_sqrt_fan_in() -> None:
    # Linear và Conv2d mặc định lấy mẫu từ U(−√k, √k), k = 1/fan_in; ở đây cũng vậy, qua Generator riêng.
    model = make_smallcnn(seeding.torch_generator(0, Stream.INIT))

    for layer in (model.conv1, model.conv2, model.fc1, model.fc2):
        bound = 1 / math.sqrt(layer.weight[0].numel())
        for tensor in (layer.weight, layer.bias):
            assert tensor.abs().max().item() <= bound * (1 + 1e-6)  # float32 làm tròn sát biên
            assert tensor.std().item() > bound / 4  # không phải tensor hằng hay toàn 0


def test_smallcnn_class_can_be_built_on_the_meta_device_for_skip_init() -> None:
    # skip_init cần module nhận tham số `device` và chuyển nó cho mọi tham số con.
    model = SmallCNN(device="meta")

    assert all(p.is_meta for p in model.parameters())


class Scaled(torch.nn.Module):
    """Linear cộng một tham số hay buffer không thuộc lớp nào có quy tắc khởi tạo."""

    def __init__(self, *, extra: str) -> None:
        super().__init__()
        self.fc = torch.nn.utils.skip_init(torch.nn.Linear, 2, 2)
        if extra == "parameter":
            self.scale = torch.nn.Parameter(torch.empty(1))
        else:
            self.register_buffer("scale", torch.empty(1))


@pytest.mark.parametrize("extra", ["parameter", "buffer"])
def test_init_rejects_tensors_left_uninitialized_by_skip_init(extra: str) -> None:
    # Sau skip_init, tensor không có quy tắc khởi tạo chứa bộ nhớ rác, nên w⁰ sẽ không tất định.
    with pytest.raises(ValueError, match="scale"):
        init_from_stream(Scaled(extra=extra), seeding.torch_generator(0, Stream.INIT))


# ---------------------------------------------------------------------------------------- thang model (T09c)

# ADR-011: số tham số của M0–M3; byte mỗi message = 4 × số tham số (float32).
LADDER = [("logreg", 7_850), ("smallcnn", 46_730), ("mlp2nn", 199_210), ("cnn_mcmahan", 1_663_370)]


def test_every_configured_model_name_has_a_model_class() -> None:
    assert set(MODEL_CLASSES) == set(config.MODELS)


@pytest.mark.parametrize(("name", "parameters"), LADDER, ids=[name for name, _ in LADDER])
def test_ladder_models_have_the_adr_011_size_and_no_batchnorm_or_dropout(name: str, parameters: int) -> None:
    model = build_model(name, seeding.torch_generator(0, Stream.INIT))

    assert sum(p.numel() for p in model.parameters()) == parameters
    assert sum(t.numel() * t.element_size() for t in model.state_dict().values()) == 4 * parameters
    assert not any(isinstance(module, BATCH_NORMS + DROPOUTS) for module in model.modules())
    assert list(model.buffers()) == []
    assert model(torch.zeros((3, 1, 28, 28))).shape == (3, 10)


@pytest.mark.parametrize("name", [name for name, _ in LADDER])
def test_ladder_models_draw_w0_only_from_the_init_stream(name: str) -> None:
    before = torch.get_rng_state()

    a = build_model(name, seeding.torch_generator(0, Stream.INIT)).state_dict()
    again = build_model(name, seeding.torch_generator(0, Stream.INIT)).state_dict()
    b = build_model(name, seeding.torch_generator(1, Stream.INIT)).state_dict()

    assert torch.equal(torch.get_rng_state(), before)
    assert all(torch.equal(a[key], again[key]) for key in a)
    assert all(not torch.equal(a[key], b[key]) for key in a)


def test_build_model_keeps_the_core_smallcnn_w0() -> None:
    built = build_model("smallcnn", seeding.torch_generator(3, Stream.INIT)).state_dict()
    core = make_smallcnn(seeding.torch_generator(3, Stream.INIT)).state_dict()

    assert all(torch.equal(built[key], core[key]) for key in core)


def test_unknown_model_name_raises_value_error() -> None:
    with pytest.raises(ValueError, match="resnet"):
        build_model("resnet", seeding.torch_generator(0, Stream.INIT))


def reference_forward(name: str, w: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """Forward viết lại từ mô tả kiến trúc ở ADR-004 và ADR-011, chỉ dùng state của model."""
    f = torch.nn.functional
    if name == "logreg":
        return f.linear(x.flatten(1), w["fc.weight"], w["fc.bias"])
    if name == "mlp2nn":
        h = f.relu(f.linear(x.flatten(1), w["fc1.weight"], w["fc1.bias"]))
        h = f.relu(f.linear(h, w["fc2.weight"], w["fc2.bias"]))
        return f.linear(h, w["fc3.weight"], w["fc3.bias"])
    padding = 2 if name == "cnn_mcmahan" else 0
    h = f.max_pool2d(f.relu(f.conv2d(x, w["conv1.weight"], w["conv1.bias"], padding=padding)), 2)
    h = f.max_pool2d(f.relu(f.conv2d(h, w["conv2.weight"], w["conv2.bias"], padding=padding)), 2)
    h = f.relu(f.linear(h.flatten(1), w["fc1.weight"], w["fc1.bias"]))
    return f.linear(h, w["fc2.weight"], w["fc2.bias"])


@pytest.mark.parametrize("name", [name for name, _ in LADDER])
def test_ladder_models_compute_the_architecture_written_in_the_adrs(name: str) -> None:
    model = build_model(name, seeding.torch_generator(0, Stream.INIT))
    x = torch.randn((4, 1, 28, 28), generator=torch.Generator().manual_seed(2))

    torch.testing.assert_close(model(x), reference_forward(name, model.state_dict(), x))

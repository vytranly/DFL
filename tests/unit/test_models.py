"""Test cho src/dfl/learning/models.py: SmallCNN 46,730 tham số, không BatchNorm/Dropout, w⁰ từ luồng init
(T09a; ADR-004; SPEC §6; T-INV-5).
"""

import math

import torch

from dfl import seeding
from dfl.learning.models import SmallCNN, make_smallcnn
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

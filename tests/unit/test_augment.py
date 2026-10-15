"""Test cho src/dfl/learning/augment.py: crop + flip trên ảnh CIFAR-10 đã chuẩn hóa, ngẫu nhiên chỉ lấy từ
Generator được truyền vào (T28e; ADR-012; T-INV-5).

Oracle là torchvision.transforms.v2.functional áp trên ảnh uint8, đúng thứ tự của RandomCrop(32, padding=4)
(fill = 0, padding_mode = 'constant') rồi RandomHorizontalFlip(p=0.5), sau đó chuẩn hóa bằng bảng tra float32
như data/datasets.py:
https://docs.pytorch.org/vision/0.29/generated/torchvision.transforms.v2.functional.pad.html
https://docs.pytorch.org/vision/0.29/generated/torchvision.transforms.v2.functional.crop.html
https://docs.pytorch.org/vision/0.29/generated/torchvision.transforms.v2.functional.horizontal_flip.html
"""

import numpy as np
import pytest
import torch
from torchvision.transforms.v2 import functional as tvf

from dfl.learning.augment import CropFlip

MEAN = (0.49, 0.48, 0.45)
STD = (0.25, 0.24, 0.26)
TABLES = np.stack(
    [((np.arange(256) / 255 - m) / s).astype(np.float32) for m, s in zip(MEAN, STD, strict=True)]
)
# Giá trị đã chuẩn hóa của điểm ảnh đen (mức 0) ở mỗi kênh: phần đệm của RandomCrop(fill=0).
BLACK = tuple(float(TABLES[c, 0]) for c in range(3))


def normalize(images: np.ndarray) -> torch.Tensor:
    """Ảnh uint8 (n, 3, H, W) → float32 chuẩn hóa theo kênh bằng bảng tra, như data/datasets.py."""
    return torch.from_numpy(np.stack([TABLES[c][images[:, c]] for c in range(3)], axis=1))


def uint8_images(n: int, seed: int) -> np.ndarray:
    # Không có mức 0, để phần đệm (đen) không trùng điểm ảnh nào của ảnh gốc.
    return np.random.default_rng(seed).integers(1, 256, (n, 3, 32, 32), dtype=np.uint8)


def test_crop_flip_matches_torchvision_pad_crop_and_flip_on_the_uint8_image() -> None:
    images = uint8_images(64, seed=1)

    out = CropFlip(fill=BLACK)(normalize(images), torch.Generator().manual_seed(7))

    # Rút lại đúng các số CropFlip phải rút: độ dời (hàng, cột) trong 0..8 của từng ảnh, rồi cờ lật.
    twin = torch.Generator().manual_seed(7)
    shifts = torch.randint(0, 9, (64, 2), generator=twin)
    flips = torch.randint(0, 2, (64,), generator=twin)
    for i in range(64):
        image = tvf.pad(torch.from_numpy(images[i]), [4], fill=0)
        image = tvf.crop(image, int(shifts[i, 0]), int(shifts[i, 1]), 32, 32)
        if flips[i]:
            image = tvf.horizontal_flip(image)
        assert torch.equal(out[i], normalize(image.numpy()[None])[0]), i
    assert {int(s) for s in shifts.flatten()} == set(range(9)) and 0 < int(flips.sum()) < 64  # ca đủ loại


def test_crop_flip_draws_three_numbers_per_image_from_the_given_generator_only() -> None:
    x = normalize(uint8_images(8, seed=2))
    before = torch.get_rng_state()
    generator = torch.Generator().manual_seed(3)

    first = CropFlip(fill=BLACK)(x, generator)
    next_draw = torch.randint(0, 1000, (4,), generator=generator)

    twin = torch.Generator().manual_seed(3)
    torch.randint(0, 9, (8, 2), generator=twin)
    torch.randint(0, 2, (8,), generator=twin)
    assert torch.equal(torch.randint(0, 1000, (4,), generator=twin), next_draw)
    assert torch.equal(torch.get_rng_state(), before)
    assert torch.equal(CropFlip(fill=BLACK)(x, torch.Generator().manual_seed(3)), first)


def test_crop_flip_returns_a_new_batch_of_the_same_shape_and_leaves_its_input_alone() -> None:
    x = normalize(uint8_images(5, seed=4))
    original = x.clone()

    out = CropFlip(fill=BLACK)(x, torch.Generator().manual_seed(0))

    assert out.shape == x.shape and out.dtype == torch.float32
    assert torch.equal(x, original)


@pytest.mark.parametrize(
    ("fill", "padding", "shape", "message"),
    [
        (BLACK, 4, (2, 1, 28, 28), "kênh"),
        (BLACK, 4, (3, 32, 32), "4 chiều"),
        (BLACK, -1, (2, 3, 32, 32), "padding"),
    ],
    ids=["mnist-channels", "no-batch-axis", "negative-padding"],
)
def test_crop_flip_rejects_inputs_it_cannot_pad(
    fill: tuple[float, ...], padding: int, shape: tuple[int, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        CropFlip(fill=fill, padding=padding)(torch.zeros(shape), torch.Generator().manual_seed(0))

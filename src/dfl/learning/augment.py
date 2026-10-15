"""Augmentation crop + flip cho ảnh CIFAR-10 khi huấn luyện cục bộ (module learning; ADR-012; T-INV-5).

Mỗi ảnh của minibatch được đệm `padding` điểm ảnh mỗi cạnh bằng điểm ảnh đen, cắt lại đúng cỡ cũ ở một độ
dời ngẫu nhiên, rồi lật ngang với xác suất 1/2. Đây là RandomCrop(32, padding=4) (fill = 0, padding_mode =
'constant') rồi RandomHorizontalFlip(p=0.5) của torchvision, áp trên ảnh uint8 trước khi chuẩn hóa. Ảnh ở đây
đã chuẩn hóa, nên phần đệm là giá trị đã chuẩn hóa của mức 0 ở từng kênh (`Datasets.black()`).
https://docs.pytorch.org/vision/0.29/generated/torchvision.transforms.RandomCrop.html
https://docs.pytorch.org/vision/0.29/generated/torchvision.transforms.RandomHorizontalFlip.html

Không dùng transform của torchvision khi huấn luyện, vì chúng lấy số từ RNG toàn cục của torch (T-INV-5).
Mọi số ngẫu nhiên lấy từ Generator được truyền vào (luồng batch của node), theo thứ tự cố định: độ dời (hàng,
cột) của từng ảnh trong 0..2·padding, rồi cờ lật của từng ảnh.
torch.randint (low, high, size, *, generator): https://docs.pytorch.org/docs/2.14/generated/torch.randint.html
torch.Tensor.repeat: https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.repeat.html
torch.stack: https://docs.pytorch.org/docs/2.14/generated/torch.stack.html
torch.flip: https://docs.pytorch.org/docs/2.14/generated/torch.flip.html
torch.where: https://docs.pytorch.org/docs/2.14/generated/torch.where.html
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class CropFlip:
    """Crop + flip của ADR-012; `fill` là giá trị đã chuẩn hóa của điểm ảnh đen ở từng kênh."""

    fill: tuple[float, ...]
    padding: int = 4

    def __post_init__(self) -> None:
        if isinstance(self.padding, bool) or not isinstance(self.padding, int) or self.padding < 0:
            raise ValueError(f"padding phải là số nguyên ≥ 0, nhận {self.padding!r}")

    def __call__(self, x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """Batch mới (n, C, H, W) đã cắt dời và lật ngang; `x` không bị sửa."""
        if x.ndim != 4:
            raise ValueError(f"crop_flip nhận batch 4 chiều (n, C, H, W), nhận shape {tuple(x.shape)}")
        n, channels, height, width = x.shape
        if channels != len(self.fill):
            raise ValueError(f"crop_flip có giá trị đệm cho {len(self.fill)} kênh, batch có {channels} kênh")
        p = self.padding
        shifts = torch.randint(0, 2 * p + 1, (n, 2), generator=generator)
        flips = torch.randint(0, 2, (n,), generator=generator).bool()
        padded = torch.tensor(self.fill, dtype=x.dtype).view(1, channels, 1, 1)
        padded = padded.repeat(n, 1, height + 2 * p, width + 2 * p)
        padded[:, :, p : p + height, p : p + width] = x
        crops = torch.stack(
            [
                padded[i, :, top : top + height, left : left + width]
                for i, (top, left) in enumerate(shifts.tolist())
            ]
        )
        return torch.where(flips.view(n, 1, 1, 1), crops.flip(-1), crops)

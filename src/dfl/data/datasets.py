"""Dữ liệu trong RAM: MNIST qua torchvision, hoặc dữ liệu tổng hợp cùng shape cho test và profile smoke
(module data; KH §3.1, §3.5; SPEC §5.2, §6, REQ-14).

- Validation được tách phân tầng khỏi tập train gốc (MNIST: 5k của 60k, còn 55k chia cho node) bằng luồng dữ
  liệu cố định VAL_SPLIT. Tập con đánh giá được chọn phân tầng từ test hay validation bằng luồng EVAL_SUBSET,
  khóa theo split. Hai luồng có entropy DATA_SEED, nên mọi lượt, ở mọi seed, dùng cùng ảnh (SPEC §6).
- Phân tầng: định mức mỗi lớp tỉ lệ với số ảnh của lớp, làm tròn bằng phương pháp số dư lớn nhất (phần dư
  bằng nhau thì lớp có chỉ số nhỏ hơn được trước), rồi chọn không lặp trong từng lớp, lớp 0 trước.
- Chuẩn hóa: (ảnh/255 − mean)/std, với mean và std của mọi điểm ảnh thuộc phần chia cho node (không gồm
  validation), áp cho cả ba tập, để không thông tin nào của tập đánh giá lọt vào huấn luyện. Thống kê tính từ
  histogram 256 mức xám, nên không cần mảng tạm float64 cỡ cả tập.

torchvision.datasets.MNIST (root, train, download; khi tải có kiểm md5):
https://docs.pytorch.org/vision/0.29/generated/torchvision.datasets.MNIST.html
torchvision chỉ được import trong hai hàm cần nó: import gói này nạp mọi model (khoảng 2.4 s trên máy chủ, đo
ngày 2026-10-02 bằng `python -X importtime`), mà lượt chạy trên dữ liệu tổng hợp không cần nó.
UNVERIFIED: thuộc tính .data (uint8, (n, 28, 28)) và .targets (int64, (n,)) không có trong trang tài liệu
0.29, chỉ có trong mã nguồn đã cài (xem contracts.NodeDataset). Dự án vẫn dùng chúng vì đọc 60k ảnh qua
__getitem__ có tài liệu mất 7.1 s, còn qua .data mất 0.06 s (đo ngày 2026-10-02). Test so hai đường đọc trên
file MNIST giả (tests/unit/test_datasets.py).
numpy: bincount, lexsort, flatnonzero, Generator.choice (`a` rỗng thì raise, nên lớp định mức 0 không gọi),
Generator.permutation:
https://numpy.org/doc/2.5/reference/generated/numpy.bincount.html
https://numpy.org/doc/2.5/reference/generated/numpy.lexsort.html
https://numpy.org/doc/2.5/reference/generated/numpy.flatnonzero.html
https://numpy.org/doc/2.5/reference/random/generated/numpy.random.Generator.choice.html
https://numpy.org/doc/2.5/reference/random/generated/numpy.random.Generator.permutation.html
torch.from_numpy (tensor dùng chung bộ nhớ với mảng): https://docs.pytorch.org/docs/2.14/generated/torch.from_numpy.html
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from dfl import seeding
from dfl.seeding import Stream

NUM_CLASSES = 10
IMAGE_SIZE = 28
MNIST_VALIDATION_SIZE = 5000
# Profile smoke (SPEC §5.2): 2,000 ảnh train, 200 trong đó làm validation, và 500 ảnh test.
SYNTHETIC_TRAIN_SIZE = 2000
SYNTHETIC_VALIDATION_SIZE = 200
SYNTHETIC_TEST_SIZE = 500
# Khóa của luồng EVAL_SUBSET theo split; đổi số ở đây là đổi tập con đánh giá của mọi lượt.
EVAL_SUBSET_KEYS = {"test": 0, "validation": 1}
# Số ảnh mỗi khối khi đếm histogram điểm ảnh.
HISTOGRAM_BLOCK = 4096


def _check_labels(labels: np.ndarray) -> None:
    if labels.ndim != 1:
        raise ValueError(f"nhãn phải là mảng một chiều, nhận shape {labels.shape}")
    if not np.issubdtype(labels.dtype, np.integer) or (
        len(labels) > 0 and (labels.min() < 0 or labels.max() >= NUM_CLASSES)
    ):
        raise ValueError("nhãn phải là số nguyên 0..9")


@dataclass(frozen=True, slots=True)
class RawSplit:
    """Ảnh uint8 (n, 28, 28) và nhãn nguyên 0..9, như trong file MNIST, chưa chuẩn hóa."""

    images: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        if self.images.dtype != np.uint8:
            raise ValueError(f"ảnh phải là uint8, nhận {self.images.dtype}")
        if self.images.ndim != 3 or self.images.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"ảnh phải có shape (n, 28, 28), nhận {self.images.shape}")
        if self.labels.shape != (len(self.images),):
            raise ValueError(f"số nhãn {self.labels.shape} không khớp số ảnh {len(self.images)}")
        _check_labels(self.labels)


@dataclass(frozen=True, slots=True)
class ImageSet:
    """Ảnh float32 (n, 1, 28, 28) đã chuẩn hóa và nhãn int64 (n,), nằm sẵn trong RAM."""

    x: torch.Tensor
    y: torch.Tensor


@dataclass(frozen=True, slots=True)
class Datasets:
    """Ba tập của một lượt; `train` là phần chia cho node (55k ảnh với MNIST)."""

    train: ImageSet
    validation: ImageSet
    test: ImageSet
    # Chỉ số của ảnh validation trong tập train gốc, tăng dần.
    validation_index: np.ndarray
    # mean và std (ddof = 0) điểm ảnh của `train`, thang 0..1; dùng chuẩn hóa cả ba tập.
    mean: float
    std: float

    def split(self, name: str) -> ImageSet:
        """Đủ tập đánh giá theo `eval.split` (ADR-007): test, hoặc validation với E0a."""
        if name not in EVAL_SUBSET_KEYS:
            raise ValueError(f"split đánh giá phải là một trong {sorted(EVAL_SUBSET_KEYS)}, nhận {name!r}")
        return self.test if name == "test" else self.validation


def stratified_sample(labels: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    """`size` chỉ số không lặp, tăng dần, chia cho các lớp theo tỉ lệ số ảnh của lớp."""
    _check_labels(labels)
    n = len(labels)
    if not 1 <= size <= n:
        raise ValueError(f"size phải nằm trong 1..{n}, nhận {size}")
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    quotas = counts * size // n
    remainders = counts * size - quotas * n
    by_remainder = np.lexsort((np.arange(NUM_CLASSES), -remainders))
    quotas[by_remainder[: size - int(quotas.sum())]] += 1
    chosen = [
        rng.choice(np.flatnonzero(labels == c), size=int(q), replace=False) for c, q in enumerate(quotas) if q
    ]
    return np.sort(np.concatenate(chosen))


def _pixel_stats(images: np.ndarray) -> tuple[float, float]:
    """mean và std (ddof = 0) của mọi điểm ảnh, thang 0..1, tính từ histogram 256 mức xám."""
    # bincount đổi đầu vào sang int64; đếm từng khối thì mảng tạm nhỏ, và tổng các histogram nguyên vẫn
    # chính xác.
    histogram = np.zeros(256, dtype=np.int64)
    for start in range(0, len(images), HISTOGRAM_BLOCK):
        histogram += np.bincount(images[start : start + HISTOGRAM_BLOCK].reshape(-1), minlength=256)
    levels = np.arange(256) / 255
    total = histogram.sum()
    mean = float(np.sum(histogram * levels) / total)
    std = float(np.sqrt(np.sum(histogram * (levels - mean) ** 2) / total))
    if std == 0:
        raise ValueError("mọi điểm ảnh của phần train bằng nhau (std = 0), không chuẩn hóa được")
    return mean, std


def prepare(train: RawSplit, test: RawSplit, *, validation_size: int) -> Datasets:
    """Tách validation, chuẩn hóa ba tập bằng thống kê của phần chia cho node, rồi chuyển sang tensor."""
    if validation_size >= len(train.labels):
        raise ValueError(
            f"validation_size = {validation_size} không chừa ảnh nào cho node "
            f"(train có {len(train.labels)} ảnh)"
        )
    validation_index = stratified_sample(train.labels, validation_size, seeding.data_rng(Stream.VAL_SPLIT))
    keep = np.ones(len(train.labels), dtype=bool)
    keep[validation_index] = False
    node_images = train.images[keep]
    mean, std = _pixel_stats(node_images)
    # Mỗi mức xám ứng với đúng một giá trị float32, nên chuẩn hóa là một phép tra bảng.
    table = ((np.arange(256) / 255 - mean) / std).astype(np.float32)

    def to_set(images: np.ndarray, labels: np.ndarray) -> ImageSet:
        x = table[images].reshape(len(images), 1, IMAGE_SIZE, IMAGE_SIZE)
        return ImageSet(x=torch.from_numpy(x), y=torch.from_numpy(labels.astype(np.int64)))

    return Datasets(
        train=to_set(node_images, train.labels[keep]),
        validation=to_set(train.images[validation_index], train.labels[validation_index]),
        test=to_set(test.images, test.labels),
        validation_index=validation_index,
        mean=mean,
        std=std,
    )


def eval_subset(datasets: Datasets, split: str, size: int) -> ImageSet:
    """Tập con phân tầng cố định `size` ảnh của `split`, cho learning curve (ADR-007); đánh giá cuối dùng
    đủ tập."""
    full = datasets.split(split)
    rng = seeding.data_rng(Stream.EVAL_SUBSET, EVAL_SUBSET_KEYS[split])
    index = torch.from_numpy(stratified_sample(full.y.numpy(), size, rng))
    return ImageSet(x=full.x[index], y=full.y[index])


def synthetic_raw() -> tuple[RawSplit, RawSplit]:
    """Train và test tổng hợp cùng shape MNIST cho test và profile smoke, cố định theo DATA_SEED.

    Mỗi lớp có một mẫu cố định (khoảng 1/4 số điểm ảnh sáng); mỗi ảnh là mẫu của lớp cộng nhiễu Gauss, nên các
    lớp tách được và smoke thấy accuracy tăng. Lớp cân bằng, thứ tự ngẫu nhiên.
    """
    rng = seeding.data_rng(Stream.SMOKE_DATA)
    prototypes = rng.random((NUM_CLASSES, IMAGE_SIZE, IMAGE_SIZE)) < 0.25

    def draw(n: int) -> RawSplit:
        labels = rng.permutation(np.arange(n) % NUM_CLASSES)
        pixels = 32 + 160 * prototypes[labels] + rng.normal(0.0, 48.0, (n, IMAGE_SIZE, IMAGE_SIZE))
        return RawSplit(images=np.clip(pixels, 0, 255).astype(np.uint8), labels=labels.astype(np.int64))

    return draw(SYNTHETIC_TRAIN_SIZE), draw(SYNTHETIC_TEST_SIZE)


def load_mnist_raw(root: Path | str) -> tuple[RawSplit, RawSplit]:
    """Train (60k) và test (10k) của MNIST đã tải vào `root`; không tải gì (ENG-8)."""
    import torchvision

    splits = []
    for is_train in (True, False):
        try:
            mnist = torchvision.datasets.MNIST(root, train=is_train, download=False)
        except RuntimeError as err:
            raise FileNotFoundError(
                f"không thấy MNIST trong {root}; tải bằng `python -m dfl download mnist --root {root}`"
            ) from err
        splits.append(RawSplit(images=mnist.data.numpy(), labels=mnist.targets.numpy()))
    return splits[0], splits[1]


def download_mnist(root: Path | str) -> None:
    """Tải train và test của MNIST vào `root`/MNIST/raw (torchvision kiểm md5); đã có thì không tải lại."""
    import torchvision

    for is_train in (True, False):
        torchvision.datasets.MNIST(root, train=is_train, download=True)


def load(dataset: str, *, root: Path | str = Path("data")) -> Datasets:
    """Ba tập của `dataset` (config.DATASETS). MNIST đọc từ `root`, mặc định data/ như lệnh download."""
    if dataset == "mnist":
        train, test = load_mnist_raw(root)
        return prepare(train, test, validation_size=MNIST_VALIDATION_SIZE)
    if dataset == "synthetic":
        train, test = synthetic_raw()
        return prepare(train, test, validation_size=SYNTHETIC_VALIDATION_SIZE)
    raise ValueError(f"dataset phải là mnist hoặc synthetic, nhận {dataset!r}")

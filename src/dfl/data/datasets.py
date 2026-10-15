"""Dữ liệu trong RAM: MNIST qua torchvision, CIFAR-10 từ bản binary, hoặc dữ liệu tổng hợp cùng shape cho test
và profile smoke (module data; KH §3.1, §3.5; SPEC §5.2, §6, REQ-14; ADR-012).

- Validation được tách phân tầng khỏi tập train gốc (MNIST: 5k của 60k, còn 55k chia cho node; CIFAR-10: 5k
  của 50k, còn 45k) bằng luồng dữ liệu cố định VAL_SPLIT. Tập con đánh giá được chọn phân tầng từ test hay
  validation bằng luồng EVAL_SUBSET, khóa theo split. Hai luồng có entropy DATA_SEED, nên mọi lượt, ở mọi
  seed, dùng cùng ảnh (SPEC §6).
- Phân tầng: định mức mỗi lớp tỉ lệ với số ảnh của lớp, làm tròn bằng phương pháp số dư lớn nhất (phần dư
  bằng nhau thì lớp có chỉ số nhỏ hơn được trước), rồi chọn không lặp trong từng lớp, lớp 0 trước.
- Chuẩn hóa theo kênh: (ảnh/255 − mean_c)/std_c, với mean và std của mọi điểm ảnh kênh c thuộc phần chia cho
  node (không gồm validation), áp cho cả ba tập, để không thông tin nào của tập đánh giá lọt vào huấn luyện.
  Thống kê tính từ histogram 256 mức, nên không cần mảng tạm float64 cỡ cả tập. MNIST có một kênh, nên ra
  đúng các giá trị như trước ADR-012.
- CIFAR-10 đọc từ bản binary bằng numpy, vì torchvision.datasets.CIFAR10 unpickle bản python (F6, ADR-012).
  Theo https://www.cs.toronto.edu/~kriz/cifar.html, mỗi file có 10,000 bản ghi 3,073 byte: byte nhãn 0–9, rồi
  1,024 byte kênh đỏ, 1,024 byte xanh lá, 1,024 byte xanh dương, mỗi kênh theo thứ tự hàng.

torchvision.datasets.MNIST (root, train, download; khi tải có kiểm md5):
https://docs.pytorch.org/vision/0.29/generated/torchvision.datasets.MNIST.html
torchvision chỉ được import trong các hàm cần nó: import gói này nạp mọi model (khoảng 2.4 s trên máy chủ, đo
ngày 2026-10-02 bằng `python -X importtime`), mà lượt chạy trên dữ liệu tổng hợp không cần nó.
UNVERIFIED: thuộc tính .data (uint8, (n, 28, 28)) và .targets (int64, (n,)) không có trong trang tài liệu
0.29, chỉ có trong mã nguồn đã cài (xem contracts.NodeDataset). Dự án vẫn dùng chúng vì đọc 60k ảnh qua
__getitem__ có tài liệu mất 7.1 s, còn qua .data mất 0.06 s (đo ngày 2026-10-02). Test so hai đường đọc trên
file MNIST giả (tests/unit/test_datasets.py).
UNVERIFIED: torchvision.datasets.utils.download_and_extract_archive(url, download_root, md5=…) không có trong
trang tài liệu 0.29. Theo mã nguồn đã cài, nó kiểm md5 rồi mới giải nén, và không tải lại file đã đúng md5;
torchvision.datasets.CIFAR10 tải bằng chính hàm này.
numpy: bincount, lexsort, flatnonzero, fromfile, Generator.choice (`a` rỗng thì raise, nên lớp định mức 0
không gọi), Generator.permutation:
https://numpy.org/doc/2.5/reference/generated/numpy.bincount.html
https://numpy.org/doc/2.5/reference/generated/numpy.lexsort.html
https://numpy.org/doc/2.5/reference/generated/numpy.flatnonzero.html
https://numpy.org/doc/2.5/reference/generated/numpy.fromfile.html
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
CIFAR10_VALIDATION_SIZE = 5000
# Shape một ảnh trong file → (C, H, W) trong RAM: MNIST lưu ảnh xám (28, 28), CIFAR-10 lưu (3, 32, 32).
MNIST_SHAPE = (IMAGE_SIZE, IMAGE_SIZE)
CIFAR10_SHAPE = (3, 32, 32)
CHANNEL_SHAPES = {MNIST_SHAPE: (1, IMAGE_SIZE, IMAGE_SIZE), CIFAR10_SHAPE: CIFAR10_SHAPE}
# Bản binary của CIFAR-10, theo https://www.cs.toronto.edu/~kriz/cifar.html (đọc ngày 2026-10-15).
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
CIFAR10_MD5 = "c32a1d4ab5d03f1284b67883e8d87530"
CIFAR10_DIR = "cifar-10-batches-bin"
CIFAR10_TRAIN_FILES = tuple(f"data_batch_{k}.bin" for k in range(1, 6))
CIFAR10_TEST_FILE = "test_batch.bin"
CIFAR10_RECORDS_PER_FILE = 10_000
CIFAR10_RECORD_BYTES = 1 + 3 * 32 * 32
# Profile smoke (SPEC §5.2): 2,000 ảnh train, 200 trong đó làm validation, và 500 ảnh test.
SYNTHETIC_TRAIN_SIZE = 2000
SYNTHETIC_VALIDATION_SIZE = 200
SYNTHETIC_TEST_SIZE = 500
# Khóa của luồng SMOKE_DATA cho dữ liệu tổng hợp 3×32×32; dữ liệu cùng shape MNIST không có khóa, để dữ liệu
# smoke có từ trước ADR-012 giữ nguyên.
SYNTHETIC_CIFAR10_KEY = 1
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


def _channels(images: np.ndarray) -> np.ndarray:
    """View (n, C, H, W) của ảnh có shape như trong file."""
    return images.reshape(len(images), *CHANNEL_SHAPES[images.shape[1:]])


@dataclass(frozen=True, slots=True)
class RawSplit:
    """Ảnh uint8 và nhãn nguyên 0..9 như trong file, chưa chuẩn hóa: (n, 28, 28) với MNIST, (n, 3, 32, 32)
    với CIFAR-10."""

    images: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        if self.images.dtype != np.uint8:
            raise ValueError(f"ảnh phải là uint8, nhận {self.images.dtype}")
        if self.images.shape[1:] not in CHANNEL_SHAPES:
            raise ValueError(
                "ảnh phải có shape (n, 28, 28) như MNIST hoặc (n, 3, 32, 32) như CIFAR-10, "
                f"nhận {self.images.shape}"
            )
        if self.labels.shape != (len(self.images),):
            raise ValueError(f"số nhãn {self.labels.shape} không khớp số ảnh {len(self.images)}")
        _check_labels(self.labels)


@dataclass(frozen=True, slots=True)
class ImageSet:
    """Ảnh float32 (n, C, H, W) đã chuẩn hóa và nhãn int64 (n,), nằm sẵn trong RAM."""

    x: torch.Tensor
    y: torch.Tensor


@dataclass(frozen=True, slots=True)
class Datasets:
    """Ba tập của một lượt; `train` là phần chia cho node (55k ảnh với MNIST, 45k với CIFAR-10)."""

    train: ImageSet
    validation: ImageSet
    test: ImageSet
    # Chỉ số của ảnh validation trong tập train gốc, tăng dần.
    validation_index: np.ndarray
    # mean và std (ddof = 0) điểm ảnh của `train` theo từng kênh, thang 0..1; dùng chuẩn hóa cả ba tập.
    mean: tuple[float, ...]
    std: tuple[float, ...]

    def black(self) -> tuple[float, ...]:
        """Giá trị đã chuẩn hóa của điểm ảnh đen (mức 0) ở từng kênh, đúng bằng giá trị trong `x`;
        augmentation crop_flip đệm bằng nó (ADR-012)."""
        return tuple(
            float(np.float32((0 - mean) / std)) for mean, std in zip(self.mean, self.std, strict=True)
        )

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
    """mean và std (ddof = 0) của mọi điểm ảnh, thang 0..1, tính từ histogram 256 mức."""
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
        raise ValueError(
            "mọi điểm ảnh của một kênh trong phần train bằng nhau (std = 0), không chuẩn hóa được"
        )
    return mean, std


def prepare(train: RawSplit, test: RawSplit, *, validation_size: int) -> Datasets:
    """Tách validation, chuẩn hóa từng kênh của ba tập bằng thống kê của phần chia cho node, rồi chuyển sang
    tensor."""
    if train.images.shape[1:] != test.images.shape[1:]:
        raise ValueError(
            f"train và test phải cùng shape ảnh, nhận {train.images.shape} và {test.images.shape}"
        )
    if validation_size >= len(train.labels):
        raise ValueError(
            f"validation_size = {validation_size} không chừa ảnh nào cho node "
            f"(train có {len(train.labels)} ảnh)"
        )
    validation_index = stratified_sample(train.labels, validation_size, seeding.data_rng(Stream.VAL_SPLIT))
    keep = np.ones(len(train.labels), dtype=bool)
    keep[validation_index] = False
    node_images = train.images[keep]
    node_channels = _channels(node_images)
    stats = [_pixel_stats(node_channels[:, c]) for c in range(node_channels.shape[1])]
    # Mỗi mức của một kênh ứng với đúng một giá trị float32, nên chuẩn hóa là một phép tra bảng cho mỗi kênh.
    tables = [((np.arange(256) / 255 - mean) / std).astype(np.float32) for mean, std in stats]

    def to_set(images: np.ndarray, labels: np.ndarray) -> ImageSet:
        channels = _channels(images)
        x = np.empty(channels.shape, dtype=np.float32)
        for c, table in enumerate(tables):
            x[:, c] = table[channels[:, c]]
        return ImageSet(x=torch.from_numpy(x), y=torch.from_numpy(labels.astype(np.int64)))

    return Datasets(
        train=to_set(node_images, train.labels[keep]),
        validation=to_set(train.images[validation_index], train.labels[validation_index]),
        test=to_set(test.images, test.labels),
        validation_index=validation_index,
        mean=tuple(mean for mean, _ in stats),
        std=tuple(std for _, std in stats),
    )


def eval_subset(datasets: Datasets, split: str, size: int) -> ImageSet:
    """Tập con phân tầng cố định `size` ảnh của `split`, cho learning curve (ADR-007); đánh giá cuối dùng
    đủ tập."""
    full = datasets.split(split)
    rng = seeding.data_rng(Stream.EVAL_SUBSET, EVAL_SUBSET_KEYS[split])
    index = torch.from_numpy(stratified_sample(full.y.numpy(), size, rng))
    return ImageSet(x=full.x[index], y=full.y[index])


def _synthetic(
    rng: np.random.Generator, prototypes: np.ndarray, shape: tuple[int, ...]
) -> tuple[RawSplit, RawSplit]:
    """Train và test tổng hợp: mỗi ảnh là 32 + 160 × mẫu của lớp (thang 0..1) cộng nhiễu Gauss σ = 48; lớp cân
    bằng, thứ tự ngẫu nhiên."""

    def draw(n: int) -> RawSplit:
        labels = rng.permutation(np.arange(n) % NUM_CLASSES)
        pixels = 32 + 160 * prototypes[labels] + rng.normal(0.0, 48.0, (n, *shape))
        return RawSplit(images=np.clip(pixels, 0, 255).astype(np.uint8), labels=labels.astype(np.int64))

    return draw(SYNTHETIC_TRAIN_SIZE), draw(SYNTHETIC_TEST_SIZE)


def synthetic_raw() -> tuple[RawSplit, RawSplit]:
    """Train và test tổng hợp cùng shape MNIST cho test và profile smoke, cố định theo DATA_SEED.

    Mỗi lớp có một mẫu cố định (khoảng 1/4 số điểm ảnh sáng), nên các lớp tách được và smoke thấy accuracy
    tăng.
    """
    rng = seeding.data_rng(Stream.SMOKE_DATA)
    return _synthetic(rng, rng.random((NUM_CLASSES, IMAGE_SIZE, IMAGE_SIZE)) < 0.25, MNIST_SHAPE)


def synthetic_cifar10_raw() -> tuple[RawSplit, RawSplit]:
    """Train và test tổng hợp cùng shape CIFAR-10 cho test và profile smoke_cifar10, cố định theo DATA_SEED.

    Mỗi lớp có một màu cố định (mức sáng riêng của từng kênh, như nhau ở mọi điểm ảnh), nên các lớp vẫn tách
    được sau khi cắt dời và lật ngang (augmentation crop_flip, ADR-012).
    """
    rng = seeding.data_rng(Stream.SMOKE_DATA, SYNTHETIC_CIFAR10_KEY)
    return _synthetic(rng, rng.random((NUM_CLASSES, 3, 1, 1)), CIFAR10_SHAPE)


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


def _read_cifar10_file(path: Path) -> RawSplit:
    """Một file của bản binary: đúng 10,000 bản ghi 3,073 byte."""
    expected = CIFAR10_RECORDS_PER_FILE * CIFAR10_RECORD_BYTES
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size != expected:
        raise ValueError(
            f"{path} phải dài đúng {expected:,} byte (10,000 bản ghi 3,073 byte), nhận {raw.size:,}; xóa "
            f"{path.parent} rồi tải lại bằng `python -m dfl download cifar10`"
        )
    records = raw.reshape(CIFAR10_RECORDS_PER_FILE, CIFAR10_RECORD_BYTES)
    images = np.ascontiguousarray(records[:, 1:]).reshape(CIFAR10_RECORDS_PER_FILE, *CIFAR10_SHAPE)
    return RawSplit(images=images, labels=records[:, 0].copy())


def load_cifar10_raw(root: Path | str) -> tuple[RawSplit, RawSplit]:
    """Train (50k, nối năm file data_batch theo thứ tự) và test (10k) của CIFAR-10 bản binary đã giải nén vào
    `root`; không tải gì (ENG-8)."""
    folder = Path(root) / CIFAR10_DIR
    missing = [name for name in (*CIFAR10_TRAIN_FILES, CIFAR10_TEST_FILE) if not (folder / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"không thấy {missing} trong {folder}; tải bằng `python -m dfl download cifar10 --root {root}`"
        )
    parts = [_read_cifar10_file(folder / name) for name in CIFAR10_TRAIN_FILES]
    train = RawSplit(
        images=np.concatenate([part.images for part in parts]),
        labels=np.concatenate([part.labels for part in parts]),
    )
    return train, _read_cifar10_file(folder / CIFAR10_TEST_FILE)


def download_cifar10(root: Path | str) -> None:
    """Tải bản binary của CIFAR-10 và giải nén vào `root`/cifar-10-batches-bin (torchvision kiểm md5 trước khi
    giải nén); đã đủ sáu file thì không tải lại."""
    from torchvision.datasets.utils import download_and_extract_archive

    folder = Path(root) / CIFAR10_DIR
    if all((folder / name).is_file() for name in (*CIFAR10_TRAIN_FILES, CIFAR10_TEST_FILE)):
        return
    download_and_extract_archive(CIFAR10_URL, str(root), md5=CIFAR10_MD5)


def load(dataset: str, *, root: Path | str = Path("data")) -> Datasets:
    """Ba tập của `dataset` (config.DATASETS). MNIST và CIFAR-10 đọc từ `root`, mặc định data/ như lệnh
    download."""
    if dataset == "mnist":
        train, test = load_mnist_raw(root)
        return prepare(train, test, validation_size=MNIST_VALIDATION_SIZE)
    if dataset == "cifar10":
        train, test = load_cifar10_raw(root)
        return prepare(train, test, validation_size=CIFAR10_VALIDATION_SIZE)
    if dataset == "synthetic":
        train, test = synthetic_raw()
        return prepare(train, test, validation_size=SYNTHETIC_VALIDATION_SIZE)
    if dataset == "synthetic_cifar10":
        train, test = synthetic_cifar10_raw()
        return prepare(train, test, validation_size=SYNTHETIC_VALIDATION_SIZE)
    raise ValueError(f"dataset phải là mnist, cifar10, synthetic hoặc synthetic_cifar10, nhận {dataset!r}")

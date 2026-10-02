"""Test cho src/dfl/data/datasets.py: tách validation phân tầng, chuẩn hóa, tập con đánh giá, dữ liệu tổng hợp
và đọc file MNIST qua torchvision (T03; KH §3.1; SPEC §5.2, §6, REQ-14).

Không dùng mạng và không cần MNIST thật (ENG-8): file MNIST giả được ghi theo định dạng IDX (số nguyên 32 bit
big-endian, rồi các byte dữ liệu), đúng tên file mà torchvision đọc sau khi tải.
"""

import struct
from pathlib import Path

import numpy as np
import pytest
import torch
import torchvision

from dfl import config, seeding
from dfl.data import datasets as ds
from dfl.seeding import Stream

# Số ảnh mỗi lớp của MNIST, để thử phân tầng ở đúng tỉ lệ thật mà không cần dữ liệu thật.
MNIST_TRAIN_COUNTS = (5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949)
MNIST_TEST_COUNTS = (980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009)


def labels_with_counts(counts: tuple[int, ...], seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).permutation(np.repeat(np.arange(10), counts)).astype(np.int64)


def raw_split(counts: tuple[int, ...], *, seed: int, offset: int = 0) -> ds.RawSplit:
    """Ảnh uint8 ngẫu nhiên; `offset` làm sáng cả tập, để thấy tập đó được chuẩn hóa bằng thống kê của
    tập khác."""
    rng = np.random.default_rng(seed)
    labels = labels_with_counts(counts, seed)
    images = rng.integers(offset, 200 + offset, (len(labels), 28, 28), dtype=np.uint8)
    return ds.RawSplit(images=images, labels=labels)


def striped_split(counts: tuple[int, ...]) -> ds.RawSplit:
    """Như raw_split nhưng rẻ ở cỡ 60k: ảnh đen có một hàng sáng theo nhãn (sinh 47M byte ngẫu nhiên mất
    0.6 s)."""
    labels = labels_with_counts(counts)
    images = np.zeros((len(labels), 28, 28), dtype=np.uint8)
    images[np.arange(len(labels)), labels] = 255
    return ds.RawSplit(images=images, labels=labels)


def write_idx(path: Path, array: np.ndarray) -> None:
    magic = 0x0800 | array.ndim  # 0x08: unsigned byte
    path.write_bytes(struct.pack(f">I{array.ndim}I", magic, *array.shape) + array.astype(np.uint8).tobytes())


def write_fake_mnist(root: Path, train: ds.RawSplit, test: ds.RawSplit) -> None:
    raw = root / "MNIST" / "raw"
    raw.mkdir(parents=True)
    for prefix, split in (("train", train), ("t10k", test)):
        write_idx(raw / f"{prefix}-images-idx3-ubyte", split.images)
        write_idx(raw / f"{prefix}-labels-idx1-ubyte", split.labels)


def class_counts(labels: np.ndarray | torch.Tensor) -> list[int]:
    return np.bincount(np.asarray(labels), minlength=10).tolist()


# ------------------------------------------------------------------------------------------------ phân tầng


def test_stratified_sample_apportions_by_largest_remainder() -> None:
    # 5 trên 10 mẫu: định mức chính xác 1.5, 1.5, 2.0; phần dư bằng nhau thì lớp có chỉ số nhỏ hơn được thêm.
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])

    chosen = ds.stratified_sample(labels, 5, seeding.data_rng(Stream.VAL_SPLIT))

    assert class_counts(labels[chosen])[:3] == [2, 1, 2]


@pytest.mark.parametrize(
    ("counts", "size", "stream"),
    [(MNIST_TRAIN_COUNTS, 5000, Stream.VAL_SPLIT), (MNIST_TEST_COUNTS, 2000, Stream.EVAL_SUBSET)],
    ids=["validation-5k-of-60k", "eval-2k-of-10k"],
)
def test_stratified_sample_keeps_class_proportions_within_one_image(
    counts: tuple[int, ...], size: int, stream: Stream
) -> None:
    labels = labels_with_counts(counts)

    chosen = ds.stratified_sample(labels, size, seeding.data_rng(stream))

    assert len(chosen) == size == len(np.unique(chosen))
    assert np.all(np.diff(chosen) > 0)
    for c, got in enumerate(class_counts(labels[chosen])):
        assert abs(got - size * counts[c] / sum(counts)) < 1


def test_stratified_sample_is_fixed_by_the_stream_and_differs_between_streams() -> None:
    labels = labels_with_counts(MNIST_TEST_COUNTS)

    first = ds.stratified_sample(labels, 2000, seeding.data_rng(Stream.EVAL_SUBSET, 0))
    again = ds.stratified_sample(labels, 2000, seeding.data_rng(Stream.EVAL_SUBSET, 0))
    other = ds.stratified_sample(labels, 2000, seeding.data_rng(Stream.EVAL_SUBSET, 1))

    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


@pytest.mark.parametrize(
    ("labels", "size", "message"),
    [
        (np.arange(10) % 10, 0, "size"),
        (np.arange(10) % 10, 11, "size"),
        (np.array([0, 1, 10]), 1, "0..9"),
        (np.array([0, -1, 2]), 1, "0..9"),
        (np.zeros((2, 2), dtype=np.int64), 1, "một chiều"),
    ],
    ids=["zero", "too-many", "label-10", "label-negative", "2d"],
)
def test_stratified_sample_rejects_bad_input(labels: np.ndarray, size: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ds.stratified_sample(labels, size, seeding.data_rng(Stream.VAL_SPLIT))


# ------------------------------------------------------------------------------------------------ prepare


@pytest.fixture(scope="module")
def small() -> tuple[ds.RawSplit, ds.RawSplit, ds.Datasets]:
    # Nhãn train để uint8 như byte trong file, để thấy prepare đổi nhãn sang int64.
    generated = raw_split((120,) * 10, seed=2)
    train = ds.RawSplit(images=generated.images, labels=generated.labels.astype(np.uint8))
    test = raw_split((30,) * 10, seed=3, offset=40)
    return train, test, ds.prepare(train, test, validation_size=100)


def test_prepare_holds_out_a_stratified_validation_split_from_the_data_stream(
    small: tuple[ds.RawSplit, ds.RawSplit, ds.Datasets],
) -> None:
    train, _, prepared = small
    expected = ds.stratified_sample(train.labels, 100, seeding.data_rng(Stream.VAL_SPLIT))
    rest = np.setdiff1d(np.arange(len(train.labels)), expected)

    assert np.array_equal(prepared.validation_index, expected)
    assert np.array_equal(prepared.validation.y.numpy(), train.labels[expected])
    assert np.array_equal(prepared.train.y.numpy(), train.labels[rest])  # giữ thứ tự gốc
    assert class_counts(prepared.validation.y) == [10] * 10


def test_prepare_normalizes_every_split_with_the_pixel_stats_of_the_node_part(
    small: tuple[ds.RawSplit, ds.RawSplit, ds.Datasets],
) -> None:
    # Arrange: thống kê tính độc lập trên đúng phần ảnh chia cho node, không gồm validation.
    train, test, prepared = small
    rest = np.setdiff1d(np.arange(len(train.labels)), prepared.validation_index)
    pixels = train.images[rest].astype(np.float64) / 255
    mean, std = pixels.mean(), pixels.std()

    # Assert
    assert prepared.mean == pytest.approx(mean, rel=1e-12)
    assert prepared.std == pytest.approx(std, rel=1e-12)
    for split, raw in (
        (prepared.train, train.images[rest]),
        (prepared.validation, train.images[prepared.validation_index]),
        (prepared.test, test.images),
    ):
        expected = torch.from_numpy((raw[:, None].astype(np.float64) / 255 - mean) / std).float()
        torch.testing.assert_close(split.x, expected)
    assert abs(float(prepared.train.x.double().mean())) < 1e-6
    assert (
        float(prepared.test.x.double().mean()) > 0.3
    )  # tập test sáng hơn: chuẩn hóa bằng thống kê của train


def test_pixel_stats_are_exact_when_counted_over_many_blocks(
    small: tuple[ds.RawSplit, ds.RawSplit, ds.Datasets], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Khối 7 ảnh: 1,100 ảnh của phần node đi qua 158 khối, khối cuối thiếu.
    train, test, expected = small
    monkeypatch.setattr(ds, "HISTOGRAM_BLOCK", 7)

    prepared = ds.prepare(train, test, validation_size=100)

    assert (prepared.mean, prepared.std) == (expected.mean, expected.std)
    assert prepared.mean == pytest.approx(
        train.images[np.setdiff1d(np.arange(1200), prepared.validation_index)].mean() / 255, rel=1e-12
    )


def test_prepared_images_are_float32_nchw_and_labels_int64(
    small: tuple[ds.RawSplit, ds.RawSplit, ds.Datasets],
) -> None:
    _, _, prepared = small

    for split, n in ((prepared.train, 1100), (prepared.validation, 100), (prepared.test, 300)):
        assert split.x.shape == (n, 1, 28, 28) and split.x.dtype == torch.float32
        assert split.y.shape == (n,) and split.y.dtype == torch.int64


@pytest.mark.parametrize(
    ("images", "labels", "message"),
    [
        (np.zeros((2, 28, 28), dtype=np.float32), np.zeros(2, dtype=np.int64), "uint8"),
        (np.zeros((2, 28, 27), dtype=np.uint8), np.zeros(2, dtype=np.int64), "28, 28"),
        (np.zeros((2, 28, 28), dtype=np.uint8), np.zeros(3, dtype=np.int64), "số nhãn"),
        (np.zeros((2, 28, 28), dtype=np.uint8), np.array([0, 10]), "0..9"),
    ],
    ids=["float-images", "bad-shape", "label-count", "label-range"],
)
def test_raw_split_rejects_malformed_arrays(images: np.ndarray, labels: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ds.RawSplit(images=images, labels=labels)


def test_prepare_rejects_constant_images_that_cannot_be_normalized() -> None:
    labels = labels_with_counts((3,) * 10)
    flat = ds.RawSplit(images=np.full((30, 28, 28), 7, dtype=np.uint8), labels=labels)

    with pytest.raises(ValueError, match="std = 0"):
        ds.prepare(flat, flat, validation_size=10)


def test_prepare_rejects_a_validation_split_that_leaves_no_node_data() -> None:
    train = raw_split((2,) * 10, seed=8)

    with pytest.raises(ValueError, match="validation_size"):
        ds.prepare(train, train, validation_size=20)


# ---------------------------------------------------------------------------------------------- tập đánh giá


def test_eval_subset_keys_cover_the_config_splits() -> None:
    assert set(ds.EVAL_SUBSET_KEYS) == set(config.SPLITS)


def test_split_names_select_the_matching_set(small: tuple[ds.RawSplit, ds.RawSplit, ds.Datasets]) -> None:
    _, _, prepared = small

    assert prepared.split("test") is prepared.test
    assert prepared.split("validation") is prepared.validation


@pytest.mark.parametrize("split", ["test", "validation"])
def test_eval_subset_is_a_fixed_stratified_subset_of_its_split(split: str) -> None:
    prepared = ds.load("synthetic")
    full = prepared.split(split)
    expected = ds.stratified_sample(
        full.y.numpy(), 100, seeding.data_rng(Stream.EVAL_SUBSET, ds.EVAL_SUBSET_KEYS[split])
    )

    subset = ds.eval_subset(prepared, split, 100)

    assert torch.equal(subset.y, full.y[expected]) and torch.equal(subset.x, full.x[expected])
    assert class_counts(subset.y) == [10] * 10
    assert torch.equal(ds.eval_subset(prepared, split, 100).x, subset.x)


def test_eval_subset_of_the_whole_split_is_the_whole_split() -> None:
    prepared = ds.load("synthetic")

    subset = ds.eval_subset(prepared, "test", 500)

    assert torch.equal(subset.x, prepared.test.x) and torch.equal(subset.y, prepared.test.y)


def test_unknown_split_raises_value_error() -> None:
    with pytest.raises(ValueError, match="split"):
        ds.load("synthetic").split("train")


# --------------------------------------------------------------------------------------------- nguồn dữ liệu


def test_synthetic_data_is_fixed_balanced_and_shaped_like_mnist() -> None:
    # SPEC §5.2, profile smoke: 2,000 ảnh train (200 trong đó làm validation) và 500 ảnh test.
    train, test = ds.synthetic_raw()
    again, _ = ds.synthetic_raw()
    prepared = ds.load("synthetic")

    assert np.array_equal(train.images, again.images) and np.array_equal(train.labels, again.labels)
    assert train.images.shape == (2000, 28, 28) and test.images.shape == (500, 28, 28)
    assert class_counts(train.labels) == [200] * 10 and class_counts(test.labels) == [50] * 10
    assert (len(prepared.train.y), len(prepared.validation.y), len(prepared.test.y)) == (1800, 200, 500)


def test_synthetic_classes_are_separable_by_their_mean_image() -> None:
    # Smoke phải học được: phân loại theo ảnh trung bình của lớp (tính trên train) đúng gần hết ảnh test.
    prepared = ds.load("synthetic")
    x, y = prepared.train.x.flatten(1), prepared.train.y
    centroids = torch.stack([x[y == c].mean(0) for c in range(10)])

    predicted = torch.cdist(prepared.test.x.flatten(1), centroids).argmin(1)

    assert float((predicted == prepared.test.y).float().mean()) > 0.9


def test_mnist_files_are_read_through_torchvision_like_its_documented_item_access(tmp_path: Path) -> None:
    # Bộ nạp đọc thuộc tính .data/.targets (UNVERIFIED theo tài liệu 0.29); oracle là __getitem__ có tài liệu.
    train, test = raw_split((3,) * 10, seed=4), raw_split((2,) * 10, seed=5)
    write_fake_mnist(tmp_path, train, test)

    loaded_train, loaded_test = ds.load_mnist_raw(tmp_path)

    for loaded, written, is_train in ((loaded_train, train, True), (loaded_test, test, False)):
        assert np.array_equal(loaded.images, written.images) and np.array_equal(loaded.labels, written.labels)
        documented = torchvision.datasets.MNIST(
            tmp_path, train=is_train, transform=torchvision.transforms.PILToTensor()
        )
        for i in (0, len(written.labels) - 1):
            image, label = documented[i]
            assert np.array_equal(image.numpy()[0], loaded.images[i]) and label == loaded.labels[i]


def test_load_mnist_splits_55k_for_nodes_and_5k_validation(tmp_path: Path) -> None:
    # REQ-14 trên dataset giả cùng kích thước và tỉ lệ lớp với MNIST.
    write_fake_mnist(tmp_path, striped_split(MNIST_TRAIN_COUNTS), striped_split(MNIST_TEST_COUNTS))

    prepared = ds.load("mnist", root=tmp_path)

    assert (len(prepared.train.y), len(prepared.validation.y), len(prepared.test.y)) == (
        55_000,
        5_000,
        10_000,
    )
    for c, got in enumerate(class_counts(prepared.validation.y)):
        assert abs(got - MNIST_TRAIN_COUNTS[c] / 12) < 1


def test_download_asks_torchvision_for_both_splits_with_download_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[Path, bool, bool]] = []
    monkeypatch.setattr(
        ds.torchvision.datasets,
        "MNIST",
        lambda root, *, train, download: calls.append((root, train, download)),
    )

    ds.download_mnist(tmp_path)

    assert calls == [(tmp_path, True, True), (tmp_path, False, True)]


def test_missing_mnist_files_raise_file_not_found_with_the_download_hint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="python -m dfl download mnist"):
        ds.load("mnist", root=tmp_path)


def test_unknown_dataset_raises_value_error() -> None:
    with pytest.raises(ValueError, match="dataset"):
        ds.load("cifar10")

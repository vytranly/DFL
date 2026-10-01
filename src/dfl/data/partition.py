"""Chia chỉ số mẫu train cho N node theo P0–P3 của NIID-Bench (module data; KH §3.2; T-INV-8).

Chỉ dùng numpy và không import `dfl.seeding` (vốn nạp torch): Generator do sim truyền vào, lấy từ luồng
"partition" (contract pure-no-torch, SPEC §3.2). Thứ tự tiêu thụ số ngẫu nhiên giống hệt hàm tham chiếu sao
từ docs/spikes/ (tests/reference/), để test golden so khớp tuyệt đối.

Khác NIID-Bench có chủ đích: min_size = 64 (= 2B) thay vì 10; lấy mẫu lại khi mọi xác suất còn lại bằng 0
(code gốc sinh NaN); lấy mẫu lại tối đa 1,000 lần rồi báo lỗi (KH §3.2, SPEC REQ-02).
numpy.random.Generator: https://numpy.org/doc/2.5/reference/random/generator.html
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dfl.contracts import PartitionResult

NUM_CLASSES = 10
DEFAULT_MIN_SIZE = 64
# Số lần lấy mẫu lại tối đa (không tính lần thử đầu); vượt thì cấu hình coi như không thực hiện được.
MAX_RETRIES = 1_000


def _check_inputs(labels: np.ndarray, num_nodes: int) -> None:
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels phải là mảng 1-D số nguyên")
    if len(labels) and (labels.min() < 0 or labels.max() >= NUM_CLASSES):
        raise ValueError(f"nhãn phải nằm trong 0..{NUM_CLASSES - 1}")
    if num_nodes < 1:
        raise ValueError("num_nodes phải ≥ 1")


def _check_min_size(min_size: int) -> None:
    if min_size < 1:
        raise ValueError("min_size phải ≥ 1")


def _result(parts: list[np.ndarray], retries: int, min_size: int) -> PartitionResult:
    if min(len(p) for p in parts) < min_size:
        raise ValueError(f"có node nhỏ hơn min_size = {min_size}; cấu hình không thực hiện được")
    return PartitionResult(parts=tuple(np.asarray(p, dtype=np.int64) for p in parts), retries=retries)


def _too_many_retries(what: str) -> ValueError:
    return ValueError(f"{what}: đã lấy mẫu lại {MAX_RETRIES:,} lần mà vẫn có node nhỏ hơn min_size")


@dataclass(frozen=True, slots=True)
class IIDPartitioner:
    """P0: hoán vị ngẫu nhiên rồi chia đều."""

    min_size: int = DEFAULT_MIN_SIZE

    def __post_init__(self) -> None:
        _check_min_size(self.min_size)

    def __call__(self, labels: np.ndarray, num_nodes: int, rng: np.random.Generator) -> PartitionResult:
        _check_inputs(labels, num_nodes)
        return _result(np.array_split(rng.permutation(len(labels)), num_nodes), 0, self.min_size)


@dataclass(frozen=True, slots=True)
class DirichletPartitioner:
    """P1, NIID-Bench `noniid-labeldir`: với mỗi lớp, p ~ Dir_N(α).

    Node đã đạt kích thước trung bình thì không nhận thêm ("balance"). Lấy mẫu lại toàn bộ nếu mọi xác suất
    còn lại bằng 0, hoặc có node nhỏ hơn min_size.
    """

    alpha: float
    min_size: int = DEFAULT_MIN_SIZE

    def __post_init__(self) -> None:
        if not self.alpha > 0:
            raise ValueError("alpha phải > 0")
        _check_min_size(self.min_size)

    def __call__(self, labels: np.ndarray, num_nodes: int, rng: np.random.Generator) -> PartitionResult:
        _check_inputs(labels, num_nodes)
        cap = len(labels) / num_nodes
        class_indices = [np.flatnonzero(labels == k) for k in range(NUM_CLASSES)]
        for attempt in range(MAX_RETRIES + 1):
            parts: list[list[int]] = [[] for _ in range(num_nodes)]
            for idx_k in class_indices:
                idx = rng.permutation(idx_k)
                p = rng.dirichlet(np.full(num_nodes, self.alpha))
                p = np.array([pj * (len(nodes) < cap) for pj, nodes in zip(p, parts, strict=True)])
                if p.sum() == 0:  # tràn số khi α rất nhỏ: bỏ lần thử này
                    break
                p = p / p.sum()
                cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
                for node, chunk in enumerate(np.split(idx, cuts)):
                    parts[node].extend(chunk.tolist())
            else:
                if min(len(nodes) for nodes in parts) >= self.min_size:
                    return _result([np.array(nodes) for nodes in parts], attempt, self.min_size)
        raise _too_many_retries(f"Dirichlet α = {self.alpha}")


@dataclass(frozen=True, slots=True)
class LabelKPartitioner:
    """P2, NIID-Bench `noniid-#label{k}`: nhãn đầu của node i là i mod 10.

    k − 1 nhãn còn lại rút ngẫu nhiên, không trùng; mẫu của mỗi nhãn chia đều cho các node sở hữu nhãn đó.
    Cần N ≥ 10 để mọi nhãn có chủ (G23).
    """

    k: int
    min_size: int = DEFAULT_MIN_SIZE

    def __post_init__(self) -> None:
        if not 1 <= self.k <= NUM_CLASSES:
            raise ValueError(f"k phải nằm trong 1..{NUM_CLASSES}")
        _check_min_size(self.min_size)

    def __call__(self, labels: np.ndarray, num_nodes: int, rng: np.random.Generator) -> PartitionResult:
        _check_inputs(labels, num_nodes)
        if num_nodes < NUM_CLASSES:
            raise ValueError(f"#C = k cần N ≥ 10 để mọi nhãn có node sở hữu, nhận N = {num_nodes} (G23)")
        owned: list[list[int]] = []
        for node in range(num_nodes):
            node_labels = [node % NUM_CLASSES]
            while len(node_labels) < self.k:
                c = int(rng.integers(NUM_CLASSES))
                if c not in node_labels:
                    node_labels.append(c)
            owned.append(node_labels)
        parts: list[list[int]] = [[] for _ in range(num_nodes)]
        for c in range(NUM_CLASSES):
            owners = [node for node in range(num_nodes) if c in owned[node]]
            idx = rng.permutation(np.flatnonzero(labels == c))
            for node, chunk in zip(owners, np.array_split(idx, len(owners)), strict=True):
                parts[node].extend(chunk.tolist())
        return _result([np.array(nodes) for nodes in parts], 0, self.min_size)


@dataclass(frozen=True, slots=True)
class QuantityPartitioner:
    """P3, NIID-Bench `iid-diff-quantity`: nhãn IID, phần của node j là q_j với q ~ Dir_N(β); lấy mẫu lại
    (cả q lẫn hoán vị) nếu có node < min_size."""

    beta: float
    min_size: int = DEFAULT_MIN_SIZE

    def __post_init__(self) -> None:
        if not self.beta > 0:
            raise ValueError("beta phải > 0")
        _check_min_size(self.min_size)

    def __call__(self, labels: np.ndarray, num_nodes: int, rng: np.random.Generator) -> PartitionResult:
        _check_inputs(labels, num_nodes)
        for attempt in range(MAX_RETRIES + 1):
            cuts = (np.cumsum(rng.dirichlet(np.full(num_nodes, self.beta))) * len(labels)).astype(int)[:-1]
            parts = np.split(rng.permutation(len(labels)), cuts)
            if min(len(p) for p in parts) >= self.min_size:
                return _result(parts, attempt, self.min_size)
        raise _too_many_retries(f"quantity β = {self.beta}")


@dataclass(frozen=True, slots=True)
class ShardsPartitioner:
    """P2', shards của McMahan et al. 2017; chỉ dùng trong `describe`, không chạy trong lưới (SPEC QS-11).

    Sắp chỉ số theo nhãn (sắp xếp ổn định), cắt thành N × s mảnh gần bằng nhau (s = shards_per_node), rồi
    rút một hoán vị của các mảnh: node i nhận các mảnh ở vị trí i·s … i·s + s − 1 của hoán vị đó.
    Chỉ rút đúng một `rng.permutation`. Kích thước node đều; số lớp mỗi node có thể là 1 hoặc nhiều hơn 2,
    nên P2' không tương đương #C = 2 (SPEC §1.4).
    """

    shards_per_node: int = 2
    min_size: int = DEFAULT_MIN_SIZE

    def __post_init__(self) -> None:
        if self.shards_per_node < 1:
            raise ValueError("shards_per_node phải ≥ 1")
        _check_min_size(self.min_size)

    def __call__(self, labels: np.ndarray, num_nodes: int, rng: np.random.Generator) -> PartitionResult:
        _check_inputs(labels, num_nodes)
        s = self.shards_per_node
        shards = np.array_split(np.argsort(labels, kind="stable"), num_nodes * s)
        order = rng.permutation(num_nodes * s)
        parts = [np.concatenate([shards[j] for j in order[i * s : (i + 1) * s]]) for i in range(num_nodes)]
        return _result(parts, 0, self.min_size)


def assign_positions(result: PartitionResult, rng: np.random.Generator) -> PartitionResult:
    """Gán các phần vào vị trí node bằng một hoán vị ngẫu nhiên (KH §3.2), để tránh cấu trúc nhân tạo.

    Ví dụ: hai node kề nhau trên ring luôn khác nhãn. Vị trí p nhận phần `perm[p]`, với perm là đúng một
    `rng.permutation(N)` của luồng placement. Cùng seed thì cùng hoán vị cho mọi topology, vì luồng
    placement không phụ thuộc topology (G9).
    """
    perm = rng.permutation(len(result.parts))
    return PartitionResult(parts=tuple(result.parts[int(i)] for i in perm), retries=result.retries)

"""Hàm chia dữ liệu tham chiếu cho test golden (T-INV-8), sao từ docs/spikes/spike_topology_partition.py.

Khác bản spike ở hai điểm, không điểm nào đổi thứ tự tiêu thụ số ngẫu nhiên:
- nhận mảng nhãn `labels` làm tham số thay cho biến toàn cục LABELS;
- `zip(..., strict=True)` theo luật ruff B905.
Độ trung thành của spike với NIID-Bench dựa trên đọc mã (DIGEST §2.4); test golden chỉ kiểm bản cài
khớp spike.
"""

import numpy as np

MNIST_TRAIN_CLASS_COUNTS = np.array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949])
VALIDATION_SIZE = 5_000


def mnist_train_labels() -> np.ndarray:
    """Nhãn của 55k mẫu train sau khi tách 5k validation phân tầng, theo số đếm lớp thật của MNIST."""
    counts = MNIST_TRAIN_CLASS_COUNTS
    held = np.floor(counts * VALIDATION_SIZE / counts.sum()).astype(int)
    order = np.argsort(-(counts * VALIDATION_SIZE / counts.sum() - held))
    held[order[: VALIDATION_SIZE - held.sum()]] += 1
    return np.repeat(np.arange(10), counts - held)


def niid_bench_iid(labels: np.ndarray, rng: np.random.Generator, n_nodes: int) -> list[np.ndarray]:
    return np.array_split(rng.permutation(len(labels)), n_nodes)


def niid_bench_dirichlet(
    labels: np.ndarray, alpha: float, rng: np.random.Generator, n_nodes: int, min_size: int
) -> tuple[list[np.ndarray], int]:
    cap = len(labels) / n_nodes
    attempts = 0
    while True:
        attempts += 1
        parts: list[list[int]] = [[] for _ in range(n_nodes)]
        for k in range(10):
            idx = rng.permutation(np.flatnonzero(labels == k))
            p = rng.dirichlet(np.full(n_nodes, alpha))
            p = np.array([pj * (len(part) < cap) for pj, part in zip(p, parts, strict=True)])
            if p.sum() == 0:
                break
            p = p / p.sum()
            cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
            for i, chunk in enumerate(np.split(idx, cuts)):
                parts[i].extend(chunk.tolist())
        else:
            if min(len(part) for part in parts) >= min_size:
                return [np.array(part) for part in parts], attempts


def niid_bench_label_k(
    labels: np.ndarray, k: int, rng: np.random.Generator, n_nodes: int
) -> list[np.ndarray]:
    owned: list[list[int]] = []
    for node in range(n_nodes):
        node_labels = [node % 10]
        while len(node_labels) < k:
            c = int(rng.integers(10))
            if c not in node_labels:
                node_labels.append(c)
        owned.append(node_labels)
    parts: list[list[int]] = [[] for _ in range(n_nodes)]
    for c in range(10):
        owners = [node for node in range(n_nodes) if c in owned[node]]
        idx = rng.permutation(np.flatnonzero(labels == c))
        for node, chunk in zip(owners, np.array_split(idx, len(owners)), strict=True):
            parts[node].extend(chunk.tolist())
    return [np.array(part) for part in parts]


def niid_bench_quantity(
    labels: np.ndarray, beta: float, rng: np.random.Generator, n_nodes: int, min_size: int
) -> list[np.ndarray]:
    while True:
        cuts = (np.cumsum(rng.dirichlet(np.full(n_nodes, beta))) * len(labels)).astype(int)[:-1]
        parts = np.split(rng.permutation(len(labels)), cuts)
        if min(len(part) for part in parts) >= min_size:
            return parts

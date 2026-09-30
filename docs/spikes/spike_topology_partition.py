"""Spike (khảo sát nhanh) phục vụ kế hoạch Đề 2 — KHÔNG phải code sản phẩm.

Tái tạo các bảng số liệu trong docs/KE_HOACH_DE_TAI_2.md (§3.2, §3.3, Phụ lục A):
  1. Spectral gap của ma trận trộn Metropolis–Hastings theo topology (N = 20), kèm dải theo seed
     cho các đồ thị ngẫu nhiên.
  2. Mức Non-IID theo cách chia dữ liệu, cài đúng như NIID-Bench (có dòng "balance" của Dirichlet;
     #C=k với nhãn đầu i % K + nhãn ngẫu nhiên), trên tập train 55k (MNIST 60k trừ 5k validation
     phân tầng). Khác NIID-Bench duy nhất ở min_size = 64 (= 2B) thay vì 10. Dùng số mẫu mỗi lớp thật
     của MNIST, không cần tải dữ liệu. Cột shift_TV là độ lệch giữa phân bố nhãn trộn đều theo node
     (mục tiêu mà DFL với trọng số ngẫu nhiên kép thực sự tối ưu) và phân bố toàn cục.

Chạy:
  pip install "numpy>=2" "networkx>=3.4"
  python docs/spikes/spike_topology_partition.py              # 5 seed (nhanh)
  python docs/spikes/spike_topology_partition.py --seeds 20   # 20 seed (dải ở Phụ lục A)
"""
import argparse
import math

import networkx as nx
import numpy as np

N = 20
BATCH = 32
MIN_SIZE = 2 * BATCH  # dự án dùng 64 thay vì 10 của NIID-Bench để node nào cũng đủ ít nhất 2 batch
MODEL_PARAMS = {"SmallCNN": 46_730, "2NN-MLP (McMahan 2017)": 199_210, "CNN (McMahan 2017)": 1_663_370}
MSG_BYTES = 4 * MODEL_PARAMS["SmallCNN"]  # float32
MNIST_TRAIN_CLASS_COUNTS = np.array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949])
VALIDATION_SIZE = 5_000


# ----------------------------------------------------------------------------- topology
def metropolis_hastings(graph: nx.Graph) -> np.ndarray:
    """W_ij = 1 / (1 + max(d_i, d_j)) trên cạnh, W_ii = 1 - tổng hàng. Đối xứng, ngẫu nhiên kép."""
    n = graph.number_of_nodes()
    w = np.zeros((n, n))
    deg = dict(graph.degree())
    for i, j in graph.edges():
        w[i, j] = w[j, i] = 1.0 / (1 + max(deg[i], deg[j]))
    w[np.diag_indices(n)] = 1.0 - w.sum(axis=1)
    return w


def second_largest_modulus(w: np.ndarray) -> float:
    return float(np.sort(np.abs(np.linalg.eigvals(w)))[::-1][1])


def connected_erdos_renyi(n: int, p: float, seed: int) -> nx.Graph:
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        graph = nx.erdos_renyi_graph(n, p, seed=int(rng.integers(2**31)))
        if nx.is_connected(graph):
            return graph
    raise RuntimeError("không tạo được đồ thị Erdős–Rényi liên thông")


def exponential_graph(n: int) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    k = 0
    while 2**k < n:
        for i in range(n):
            graph.add_edge(i, (i + 2**k) % n)
        k += 1
    return graph


def report_topologies(seeds: range) -> None:
    print(f"== Topology, N={N}, trọng số Metropolis–Hastings, 1 message SmallCNN = {MSG_BYTES / 1e6:.3f} MB")
    print(f"{'topology':34s} {'deg':>5s} {'diam':>4s} {'rho':>6s} {'gap':>6s} {'rounds->1%':>10s} {'msg/rd':>6s} {'MB/rd':>6s}")
    fixed = {
        "ring": nx.cycle_graph(N),
        "random 4-regular (seed 0)": nx.random_regular_graph(4, N, seed=0),
        "Erdos-Renyi p=0.2 (seed 0)": connected_erdos_renyi(N, 0.2, 0),
        "Erdos-Renyi p=0.3 (seed 0)": connected_erdos_renyi(N, 0.3, 0),
        "exponential (static, undirected)": exponential_graph(N),
        "fully-connected": nx.complete_graph(N),
    }
    for name, graph in fixed.items():
        w = metropolis_hastings(graph)
        assert np.allclose(w.sum(0), 1) and np.allclose(w.sum(1), 1) and np.allclose(w, w.T) and (w >= -1e-12).all()
        rho = second_largest_modulus(w)
        messages = 2 * graph.number_of_edges()
        rounds = math.ceil(math.log(0.01) / math.log(rho)) if rho > 1e-12 else 1
        print(f"{name:34s} {messages / N:5.2f} {nx.diameter(graph):4d} {rho:6.3f} {1 - rho:6.3f} {rounds:10d} "
              f"{messages:6d} {messages * MSG_BYTES / 1e6:6.2f}")
    # Gossip ngẫu nhiên theo cặp trên K_N (mỗi round một perfect matching đều trên K_N, lịch sinh từ seed chung):
    # W^2 = W nên E[W^T W] = E[W], lambda_2(E[W]) = 1/2 - 1/(2(N-1)); hệ số co kỳ vọng (chuẩn L2) = sqrt(lambda_2).
    rho = math.sqrt(0.5 - 1 / (2 * (N - 1)))
    print(f"{'random pairwise gossip on K_N':34s} {1.0:5.2f} {'-':>4s} {rho:6.3f} {1 - rho:6.3f} "
          f"{math.ceil(math.log(0.01) / math.log(rho)):10d} {N:6d} {N * MSG_BYTES / 1e6:6.2f}  (kỳ vọng)")
    print(f"{'star / FedAvg có server (tham chiếu)':34s} {'-':>5s} {2:4d} {0.0:6.3f} {1.0:6.3f} {1:10d} "
          f"{2 * N:6d} {2 * N * MSG_BYTES / 1e6:6.2f}  (server nhận {N} + gửi {N})")
    print("Kích thước model (float32):", {k: f"{4 * v / 1e6:.3f} MB" for k, v in MODEL_PARAMS.items()})
    print(f"\n-- Dải spectral gap theo {len(seeds)} seed (đồ thị ngẫu nhiên)")
    for name, make in {
        "Erdos-Renyi p=0.2": lambda s: connected_erdos_renyi(N, 0.2, s),
        "Erdos-Renyi p=0.3": lambda s: connected_erdos_renyi(N, 0.3, s),
        "random 4-regular": lambda s: nx.random_regular_graph(4, N, seed=s),
    }.items():
        gaps = [1 - second_largest_modulus(metropolis_hastings(make(s))) for s in seeds]
        print(f"{name:34s} gap mean {np.mean(gaps):.3f}  min {min(gaps):.3f}  max {max(gaps):.3f}")


# ----------------------------------------------------------------------------- partitions
def stratified_train_labels(seed: int = 0) -> np.ndarray:
    """Nhãn của 55k mẫu train sau khi tách 5k validation phân tầng (theo tỉ lệ lớp)."""
    counts = MNIST_TRAIN_CLASS_COUNTS
    held = np.floor(counts * VALIDATION_SIZE / counts.sum()).astype(int)
    order = np.argsort(-(counts * VALIDATION_SIZE / counts.sum() - held))
    held[order[: VALIDATION_SIZE - held.sum()]] += 1  # làm tròn để tổng đúng 5000
    return np.repeat(np.arange(10), counts - held)


LABELS = stratified_train_labels()
GLOBAL_DIST = np.bincount(LABELS, minlength=10) / len(LABELS)


def js_divergence_bits(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def heterogeneity(parts: list[np.ndarray]) -> dict[str, float]:
    sizes = np.array([len(ix) for ix in parts])
    dists = np.array([np.bincount(LABELS[ix], minlength=10) / len(ix) for ix in parts])
    equal_weight_mix = dists.mean(axis=0)  # phân bố nhãn mà DFL (trọng số ngẫu nhiên kép) thực sự tối ưu
    return {
        "js": float(np.mean([js_divergence_bits(d, GLOBAL_DIST) for d in dists])),
        "tv": float(np.mean([0.5 * np.abs(d - GLOBAL_DIST).sum() for d in dists])),
        "classes_ge_5pct": float(np.mean([(d >= 0.05).sum() for d in dists])),
        "n_min": float(sizes.min()),
        "n_max": float(sizes.max()),
        "cv_n": float(sizes.std() / sizes.mean()),
        "objective_shift_tv": float(0.5 * np.abs(equal_weight_mix - GLOBAL_DIST).sum()),
    }


def niid_bench_dirichlet(alpha: float, rng: np.random.Generator, n_nodes: int = N,
                         min_size: int = MIN_SIZE) -> tuple[list[np.ndarray], int]:
    """NIID-Bench `noniid-labeldir`: với mỗi lớp, p ~ Dir_N(alpha); node đã đạt kích thước trung bình
    thì không nhận thêm ("balance"); lấy mẫu lại toàn bộ nếu có node < min_size."""
    cap = len(LABELS) / n_nodes
    attempts = 0
    while True:
        attempts += 1
        parts: list[list[int]] = [[] for _ in range(n_nodes)]
        for k in range(10):
            idx = rng.permutation(np.flatnonzero(LABELS == k))
            p = rng.dirichlet(np.full(n_nodes, alpha))
            p = np.array([pj * (len(part) < cap) for pj, part in zip(p, parts)])
            if p.sum() == 0:  # mọi node còn chỗ đều nhận xác suất 0 (tràn số khi alpha rất nhỏ) -> lấy mẫu lại
                break
            p = p / p.sum()
            cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
            for i, chunk in enumerate(np.split(idx, cuts)):
                parts[i].extend(chunk.tolist())
        else:
            if min(len(part) for part in parts) >= min_size:
                return [np.array(part) for part in parts], attempts


def niid_bench_label_k(k: int, rng: np.random.Generator, n_nodes: int = N) -> list[np.ndarray]:
    """NIID-Bench `noniid-#label{k}`: nhãn đầu = i % 10 (mọi nhãn đều có chủ khi N >= 10), k-1 nhãn còn lại
    chọn ngẫu nhiên, không trùng; mẫu của mỗi nhãn chia đều cho các node sở hữu nhãn đó."""
    owned: list[list[int]] = []
    for node in range(n_nodes):
        labels = [node % 10]
        while len(labels) < k:
            c = int(rng.integers(10))
            if c not in labels:
                labels.append(c)
        owned.append(labels)
    parts: list[list[int]] = [[] for _ in range(n_nodes)]
    for c in range(10):
        owners = [node for node in range(n_nodes) if c in owned[node]]
        idx = rng.permutation(np.flatnonzero(LABELS == c))
        for node, chunk in zip(owners, np.array_split(idx, len(owners))):
            parts[node].extend(chunk.tolist())
    return [np.array(part) for part in parts]


def niid_bench_quantity(beta: float, rng: np.random.Generator, n_nodes: int = N,
                        min_size: int = MIN_SIZE) -> list[np.ndarray]:
    """NIID-Bench `iid-diff-quantity`: nhãn IID, kích thước q ~ Dir_N(beta)."""
    while True:
        cuts = (np.cumsum(rng.dirichlet(np.full(n_nodes, beta))) * len(LABELS)).astype(int)[:-1]
        parts = np.split(rng.permutation(len(LABELS)), cuts)
        if min(len(part) for part in parts) >= min_size:
            return parts


COLUMNS = ["js", "tv", "classes_ge_5pct", "n_min", "n_max", "cv_n", "objective_shift_tv"]


def fmt(name: str, rows: list[dict[str, float]], extra: str = "-") -> str:
    mean = {c: np.mean([r[c] for r in rows]) for c in COLUMNS}
    return (f"{name:24s} {mean['js']:8.3f} {mean['tv']:6.3f} {mean['classes_ge_5pct']:9.2f} {mean['n_min']:6.0f} "
            f"{mean['n_max']:6.0f} {mean['cv_n']:6.2f} {mean['objective_shift_tv']:9.3f} {extra:>9s}")


def spread(name: str, rows: list[dict[str, float]]) -> str:
    def rng_of(c: str) -> str:
        vals = [r[c] for r in rows]
        return f"{min(vals):.3f}–{max(vals):.3f}"
    return f"{name:24s} JS {rng_of('js'):13s} #lớp≥5% {rng_of('classes_ge_5pct'):13s} CV {rng_of('cv_n'):13s} shift {rng_of('objective_shift_tv')}"


def report_partitions(seeds: range) -> None:
    streams = {s: np.random.default_rng(np.random.SeedSequence(s)) for s in seeds}
    print(f"\n== Cách chia (NIID-Bench, min_size={MIN_SIZE}), N={N}, train {len(LABELS)} mẫu, trung bình {len(seeds)} seed")
    print(f"{'partition':24s} {'JS(bit)':>8s} {'TV':>6s} {'#lớp>=5%':>9s} {'n_min':>6s} {'n_max':>6s} {'CV(n)':>6s} "
          f"{'shift_TV':>9s} {'resample':>9s}")
    iid = [heterogeneity(np.array_split(streams[s].permutation(len(LABELS)), N)) for s in seeds]
    print(fmt("IID (chia đều)", iid))
    table: dict[str, list[dict[str, float]]] = {}
    for alpha in [100, 1.0, 0.5, 0.1, 0.05]:
        runs = [niid_bench_dirichlet(alpha, np.random.default_rng([s, 1])) for s in seeds]
        table[f"Dirichlet alpha={alpha}"] = rows = [heterogeneity(p) for p, _ in runs]
        print(fmt(f"Dirichlet alpha={alpha}", rows, f"{np.mean([a for _, a in runs]):.1f}"))
    for k in [5, 3, 2, 1]:
        table[f"#C={k}"] = rows = [heterogeneity(niid_bench_label_k(k, np.random.default_rng([s, 2]))) for s in seeds]
        print(fmt(f"#C={k} nhãn/node", rows))
    for beta in [5.0, 1.0, 0.5]:
        table[f"quantity beta={beta}"] = rows = [heterogeneity(niid_bench_quantity(beta, np.random.default_rng([s, 3])))
                                                 for s in seeds]
        print(fmt(f"quantity beta={beta}", rows))
    print(f"\n-- Dải theo {len(seeds)} seed")
    for name, rows in table.items():
        print(spread(name, rows))
    print(f"\n-- Vì sao min_size = {MIN_SIZE}: với min_size = 10 (mặc định NIID-Bench), số seed có node < B={BATCH} / < 2B;"
          f" và số lần lấy mẫu lại khi min_size = {MIN_SIZE}")
    for n_nodes, alpha in [(20, 0.1), (20, 0.05), (50, 0.1)]:
        small = small2 = 0
        attempts = []
        for s in seeds:
            parts, _ = niid_bench_dirichlet(alpha, np.random.default_rng([s, 1]), n_nodes=n_nodes, min_size=10)
            smallest = min(len(p) for p in parts)
            small += smallest < BATCH
            small2 += smallest < 2 * BATCH
            attempts.append(niid_bench_dirichlet(alpha, np.random.default_rng([s, 4]), n_nodes=n_nodes)[1])
        print(f"N={n_nodes} alpha={alpha}: seed có node < {BATCH}: {small}/{len(seeds)}; < {2 * BATCH}: {small2}/{len(seeds)}; "
              f"lấy mẫu lại (min_size={MIN_SIZE}) trung bình {np.mean(attempts):.1f}, tối đa {max(attempts)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    assert MNIST_TRAIN_CLASS_COUNTS.sum() == 60_000 and len(LABELS) == 55_000
    report_topologies(range(args.seeds))
    report_partitions(range(args.seeds))

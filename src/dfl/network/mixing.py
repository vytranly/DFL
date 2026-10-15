"""Ma trận trộn W^(t) và spectral gap (module network; KH §3.3; ADR-005; INV-4).

Chỉ phụ thuộc numpy/networkx để test chạy trong mili-giây và không kéo theo torch (contract pure-no-torch).
Hàm ở đây không kiểm tính liên thông: matching của gossip vốn không liên thông. Tính liên thông của đồ thị
tĩnh do TopologySchedule kiểm lúc dựng (T06).

numpy.linalg.eigvalsh (ma trận đối xứng): https://numpy.org/doc/2.5/reference/generated/numpy.linalg.eigvalsh.html
numpy.linalg.eigvals (ma trận bất kỳ): https://numpy.org/doc/2.5/reference/generated/numpy.linalg.eigvals.html
networkx.Graph.degree:
https://networkx.org/documentation/networkx-3.7/reference/classes/generated/networkx.Graph.degree.html
networkx.number_of_selfloops:
https://networkx.org/documentation/networkx-3.7/reference/generated/networkx.classes.function.number_of_selfloops.html
networkx.to_numpy_array (weight=None: mọi cạnh có giá trị 1, bỏ qua thuộc tính cạnh):
https://networkx.org/documentation/networkx-3.7/reference/generated/networkx.convert_matrix.to_numpy_array.html
"""

from __future__ import annotations

import networkx as nx
import numpy as np

OBJECTIVE_SCHEMES = ("metropolis_hastings", "uniform", "size_weighted", "size_corrected")


def _check_graph(graph: nx.Graph) -> int:
    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("đồ thị giao tiếp phải vô hướng và không có đa cạnh (INV-4)")
    n = graph.number_of_nodes()
    if set(graph.nodes) != set(range(n)):
        raise ValueError("node phải được đánh số 0..N-1 để khớp chỉ số hàng của W")
    if nx.number_of_selfloops(graph) > 0:
        raise ValueError("không được có self-loop: node không tự gửi message cho mình (G7)")
    return n


def metropolis_hastings(graph: nx.Graph) -> np.ndarray:
    """W[i, j] = 1 / (1 + max(d_i, d_j)) trên mỗi cạnh; phần còn lại của hàng dồn vào W[i, i].

    Vì sao MH: W đối xứng và ngẫu nhiên kép nên trung bình các model được bảo toàn, và mỗi node tính được
    hàng của mình chỉ từ bậc của láng giềng (Xiao & Boyd 2004; ADR-005). Trên một matching, mỗi cặp nhận
    W = ½ và node không được ghép giữ W_ii = 1 (ADR-009).
    """
    n = _check_graph(graph)
    degree = dict(graph.degree())
    weights = np.zeros((n, n))
    for i, j in graph.edges():
        weights[i, j] = weights[j, i] = 1.0 / (1 + max(degree[i], degree[j]))
    np.fill_diagonal(weights, 1.0 - weights.sum(axis=1))
    return weights


def uniform(graph: nx.Graph) -> np.ndarray:
    """W[i, j] = 1 / (d_i + 1) cho j là láng giềng hoặc chính i: chỉ ngẫu nhiên theo hàng.

    Trên đồ thị không đều, mạng đồng thuận về trung bình có trọng số tỉ lệ với d_j + 1 (ADR-005), nên ablation
    này chỉ chạy trên ER; trên đồ thị đều nó trùng MH.
    """
    n = _check_graph(graph)
    # weight=None: W chỉ phụ thuộc cấu trúc đồ thị, không đọc thuộc tính cạnh `weight` (review C3 P3.7, P2-1).
    weights = nx.to_numpy_array(graph, nodelist=range(n), dtype=float, weight=None) + np.eye(n)
    return weights / weights.sum(axis=1, keepdims=True)


def size_weighted(graph: nx.Graph, node_sizes: np.ndarray) -> np.ndarray:
    """W[i, j] = n_j / n trên đồ thị đầy đủ: đúng bằng bước tổng hợp của FedAvg (B3, T-INV-3).

    Trên đồ thị không đầy đủ, chuẩn hóa n_j theo hàng không cho mục tiêu Σ n_j/n F_j (G19), nên bị từ chối.
    """
    n = _check_graph(graph)
    if graph.number_of_edges() != n * (n - 1) // 2:
        raise ValueError("size-weighted chỉ dùng trên đồ thị đầy đủ (FC); xem ADR-005")
    sizes = np.asarray(node_sizes, dtype=float)
    if sizes.shape != (n,) or (sizes <= 0).any():
        raise ValueError("node_sizes phải có đúng N phần tử dương")
    return np.tile(sizes / sizes.sum(), (n, 1))


def identity(num_nodes: int) -> np.ndarray:
    """W = I của baseline B2 (chỉ huấn luyện cục bộ); ngoại lệ duy nhất của INV-4 về liên thông."""
    return np.eye(num_nodes)


def spectral_gap(weights: np.ndarray) -> float:
    """1 − |λ₂|, với |λ₂| là môđun lớn thứ hai của các trị riêng của W.

    W đối xứng (MH, matching) dùng eigvalsh; W không đối xứng (uniform) phải dùng eigvals.
    """
    if np.allclose(weights, weights.T):
        moduli = np.abs(np.linalg.eigvalsh(weights))
    else:
        moduli = np.abs(np.linalg.eigvals(weights))
    if len(moduli) < 2:
        raise ValueError("spectral gap cần ít nhất 2 node")
    return float(1.0 - np.sort(moduli)[-2])


def objective_weights(scheme: str, graph: nx.Graph, node_sizes: np.ndarray) -> np.ndarray:
    """Trọng số π của mục tiêu mà lượt chạy thực sự tối ưu (ADR-005), ghi vào manifest (`objective_weights`).

    MH: 1/N; uniform: tỉ lệ d_j + 1; size-weighted và size-corrected: n_j/n.
    """
    n = _check_graph(graph)
    if scheme == "metropolis_hastings":
        return np.full(n, 1.0 / n)
    if scheme == "uniform":
        weights = np.array([graph.degree(j) + 1 for j in range(n)], dtype=float)
        return weights / weights.sum()
    if scheme in ("size_weighted", "size_corrected"):
        sizes = np.asarray(node_sizes, dtype=float)
        return sizes / sizes.sum()
    raise ValueError(f"scheme phải là một trong {OBJECTIVE_SCHEMES}, nhận {scheme!r}")

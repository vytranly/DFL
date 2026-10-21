"""Bậc của node, node giữ lớp hiếm và độ đo của H9 cho E7 (module analysis; T30d; ADR-014).

- Bậc tính từ `graph_edges` của manifest, không dựng lại đồ thị từ cấu hình: analysis chỉ được import
  `dfl.contracts` (contract `analysis-reads-only`), và sửa code topology sau này không được âm thầm làm sai
  hình.
- Node giữ lớp hiếm là node có mẫu của lớp 5–9 theo `node_class_counts` của manifest, nên không phụ thuộc quy
  tắc vị trí của `src/`.
- Lượt không có `graph_edges` (ghi trước ADR-014, ví dụ lưới lõi) vẫn có bậc nếu topology tĩnh: mỗi round
  node i gửi và nhận đúng deg(i) message, nên deg(i) = `node_messages_cum`[i] / (2 · round) (ADR-015).
- Loại bậc theo định nghĩa của GVHD (góp ý buổi 2): hub là node có từ 6 kết nối, leaf là node có 1–2 kết nối.
- H9 (thẻ docs/experiments/E7.md): độ đo là trung bình `final_acc_per_node` (đủ tập test) của node không giữ
  lớp hiếm; phép so sánh chính là hiệu hub − edge ghép cặp theo seed (cùng partition, đồ thị, w⁰; G9).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from analysis.aggregate import AnalysisError, Run

# Phải trùng dfl.data.partition.RARE_CLASSES; analysis không import được dfl.data, nên
# tests/integration/test_analysis_smoke.py giữ hai nơi trùng nhau.
RARE_CLASSES = (5, 6, 7, 8, 9)
PLACEMENTS = ("hub", "edge", "random")
# Topology có đồ thị cố định qua mọi round, nên tải message cho đúng bậc.
STATIC_KINDS = ("ring", "lattice", "erdos_renyi", "barabasi_albert", "fully_connected", "star")
# Phải trùng HUB_DEGREE, LEAF_DEGREE của dfl.sim.describe (test tích hợp giữ hai nơi trùng nhau).
HUB_DEGREE = 6
LEAF_DEGREE = 2
DEGREE_CLASSES = ("hub", "giữa", "leaf")


def placement(run: Run) -> str:
    return str(run.config["partition"]["placement"])


def degrees(run: Run) -> np.ndarray:
    """Bậc của từng node theo `graph_edges`, hay suy từ tải message khi lượt ghi trước ADR-014; báo lỗi khi đồ
    thị đổi mỗi round (gossip) hay lượt không có cả hai."""
    edges = run.manifest.get("graph_edges")
    if edges is None:
        return _degrees_from_messages(run)
    result = np.zeros(int(run.config["num_nodes"]), dtype=int)
    for u, v in edges:
        result[u] += 1
        result[v] += 1
    return result


def _degrees_from_messages(run: Run) -> np.ndarray:
    """deg(i) = node_messages_cum[i] / (2 · round) ở mốc cuối, chỉ với topology tĩnh (ADR-015)."""
    last = run.rounds[-1] if run.rounds else {}
    if run.config["topology"]["kind"] not in STATIC_KINDS or "node_messages_cum" not in last:
        raise AnalysisError(
            f"lượt {run.run_id} không có graph_edges (đồ thị đổi mỗi round, đường server, hay lượt ghi trước "
            "ADR-014, không có tải message theo node), nên không tính được bậc"
        )
    n = int(run.config["num_nodes"])
    degree = np.array(last["node_messages_cum"][:n], dtype=float) / (2 * int(last["round"]))
    if not np.allclose(degree, np.round(degree)):
        raise AnalysisError(f"lượt {run.run_id}: tải message không cho bậc nguyên ({degree.tolist()})")
    return np.round(degree).astype(int)


def edges(run: Run) -> list[list[int]]:
    """Cạnh của đồ thị tĩnh từ manifest (P15 vẽ cạnh, nên không suy được từ tải message)."""
    graph = run.manifest.get("graph_edges")
    if graph is None:
        raise AnalysisError(
            f"lượt {run.run_id} không có graph_edges (ghi trước ADR-014), nên không vẽ được mạng"
        )
    return [list(edge) for edge in graph]


def degree_class(degree: int) -> str:
    """hub (bậc ≥ 6), leaf (bậc 1–2) hay giữa (3–5), theo định nghĩa của GVHD."""
    if degree >= HUB_DEGREE:
        return "hub"
    return "leaf" if 1 <= degree <= LEAF_DEGREE else "giữa"


def holders(run: Run) -> np.ndarray:
    """Mặt nạ N phần tử: node có ít nhất một mẫu của lớp 5–9."""
    counts = np.array(run.manifest["node_class_counts"])
    return counts[:, list(RARE_CLASSES)].sum(axis=1) > 0


def js_mean(run: Run) -> float:
    """Trung bình qua node của JS(P_i ‖ P) (bit) từ `node_class_counts`, như `dfl.data.heterogeneity`: mức
    Non-IID của cách chia, dùng để xếp các mức dữ liệu khi chọn hàng của P16 và mức của P17 (ADR-015)."""
    counts = np.array(run.manifest["node_class_counts"], dtype=float)
    dists = counts / counts.sum(axis=1, keepdims=True)
    pooled = counts.sum(axis=0) / counts.sum()
    middle = 0.5 * (dists + pooled)

    def kl(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ratio = np.divide(a, b, out=np.ones_like(a), where=a > 0)
        return (a * np.log2(ratio)).sum(axis=-1)

    return float(np.mean(0.5 * kl(dists, middle) + 0.5 * kl(np.broadcast_to(pooled, dists.shape), middle)))


def final_accuracy(run: Run) -> np.ndarray:
    return np.array(run.summary["final_acc_per_node"], dtype=float)


def final_byte_load(run: Run) -> np.ndarray:
    """Tải byte cộng dồn (gửi + nhận) của N node ở dòng cuối của rounds.jsonl, bỏ server ảo nếu có."""
    n = int(run.config["num_nodes"])
    return np.array(run.rounds[-1]["node_bytes_cum"][:n], dtype=float)


def non_holder_accuracy(run: Run) -> float:
    """Độ đo của H9: accuracy cuối trung bình của node không giữ lớp hiếm."""
    return float(final_accuracy(run)[~holders(run)].mean())


def hub_minus_edge(runs: Sequence[Run]) -> dict[int, float]:
    """Seed → độ đo H9 của ô hub trừ ô edge, chỉ với seed có đủ cả hai ô `ok`."""
    by_seed: dict[int, dict[str, float]] = defaultdict(dict)
    for run in runs:
        if run.status == "ok" and placement(run) in ("hub", "edge"):
            by_seed[run.seed][placement(run)] = non_holder_accuracy(run)
    return {
        seed: cells["hub"] - cells["edge"]
        for seed, cells in sorted(by_seed.items())
        if {"hub", "edge"} <= set(cells)
    }

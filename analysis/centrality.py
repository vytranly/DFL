"""Bậc của node, node giữ lớp hiếm và độ đo của H9 cho E7 (module analysis; T30d; ADR-014).

- Bậc tính từ `graph_edges` của manifest, không dựng lại đồ thị từ cấu hình: analysis chỉ được import
  `dfl.contracts` (contract `analysis-reads-only`), và sửa code topology sau này không được âm thầm làm sai
  hình.
- Node giữ lớp hiếm là node có mẫu của lớp 5–9 theo `node_class_counts` của manifest, nên không phụ thuộc quy
  tắc vị trí của `src/`.
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


def placement(run: Run) -> str:
    return str(run.config["partition"]["placement"])


def degrees(run: Run) -> np.ndarray:
    """Bậc của từng node theo `graph_edges`; báo lỗi khi lượt không có đồ thị tĩnh hay ghi trước ADR-014."""
    edges = run.manifest.get("graph_edges")
    if edges is None:
        raise AnalysisError(
            f"lượt {run.run_id} không có graph_edges (đồ thị đổi mỗi round, đường server, hay lượt ghi trước "
            "ADR-014), nên không tính được bậc"
        )
    result = np.zeros(int(run.config["num_nodes"]), dtype=int)
    for u, v in edges:
        result[u] += 1
        result[v] += 1
    return result


def holders(run: Run) -> np.ndarray:
    """Mặt nạ N phần tử: node có ít nhất một mẫu của lớp 5–9."""
    counts = np.array(run.manifest["node_class_counts"])
    return counts[:, list(RARE_CLASSES)].sum(axis=1) > 0


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

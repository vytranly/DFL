"""Chi phí giao tiếp theo loại (message, byte, thời gian) và theo node, cho P12 và P13 (module analysis; T29c;
ADR-013; SPEC REQ-05, REQ-10).

- **Message và byte** lấy từ rounds.jsonl, nên tất định. Byte = message × `nbytes`, mà `nbytes` không đổi
  trong một lượt; hai loại chỉ khác nhau khi so các model khác cỡ (E8) hay so với đường server.
- **Thời gian** lấy từ timing.jsonl, không tất định. Mỗi pha được cộng qua mọi lần ghi rồi chia cho số round
  của lần ghi cuối. "Khác" là phần của `wall_time_s` không thuộc pha nào (chờ ở barrier của mặt phẳng đo,
  lập kế hoạch round), chặn dưới ở 0: ở process runtime, eval của observer chạy song song với train của node,
  nên tổng các pha có thể vượt `wall_time_s`. `comm_s` bằng 0 ở runtime inprocess (ADR-010).
- **Theo node** (ADR-013): tải (gửi + nhận) ở mốc cuối chia cho số round. Ở đường server, phần tử thứ N là
  server ảo; nó được tách riêng để không lẫn vào phân bố của các node dữ liệu.
- Lượt ghi trước ADR-013 không có các dãy theo node: hàm theo node báo lỗi rõ thay vì vẽ dở.
- **Thời gian để so số round với thời gian thực** (ADR-015; góp ý buổi 2 của GVHD):
  - *đo được*: giây tích lũy của các pha train, comm, combine (không tính đánh giá, vốn chỉ là đo lường). Ở
    runtime inprocess các node chạy lần lượt, nên đây là tổng qua node, không phải thời gian của hệ phân tán;
  - *ước tính*: mỗi khoảng giữa hai mốc tốn max_i(train_s_per_node) + max_i(byte gửi + nhận của node i) / B.
    Giả định các node huấn luyện song song, gửi và nhận dùng chung băng thông B, bỏ qua độ trễ và thời gian
    tổng hợp. Lấy max của tổng trong khoảng thay cho tổng của max từng round: đúng với byte của topology tĩnh
    (tải mỗi round không đổi), xấp xỉ với thời gian huấn luyện (gần như bằng nhau giữa các round).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from analysis.aggregate import AnalysisError, Run
from dfl.contracts import TimingRecord

PHASES = ("train", "comm", "combine", "eval")
# Byte mỗi giây của 1 Mbit/s; băng thông tham chiếu của P17 và của cột s@x_est (ADR-015).
MBIT_BYTES = 1e6 / 8
REFERENCE_MBIT = 10.0


@dataclass(frozen=True)
class CostSeries:
    """Message và byte tích lũy của một lượt theo bước cục bộ, ở các mốc đánh giá."""

    steps: np.ndarray
    messages: np.ndarray
    bytes: np.ndarray


@dataclass(frozen=True)
class NodeLoad:
    """Tải mỗi round của từng node dữ liệu (N phần tử), và của server ảo nếu lượt đi đường server."""

    messages: np.ndarray
    bytes: np.ndarray
    server_messages: float | None
    server_bytes: float | None


def cost_series(run: Run) -> CostSeries:
    return CostSeries(
        steps=np.array([r["local_steps"] for r in run.rounds]),
        messages=np.array([r["messages_cum"] for r in run.rounds]),
        bytes=np.array([r["bytes_cum"] for r in run.rounds]),
    )


def _timing(run: Run) -> list[TimingRecord]:
    if not run.timing:
        raise AnalysisError(f"lượt {run.run_id} không có timing.jsonl, nên không có thời gian")
    return run.timing


def phase_seconds_per_round(run: Run) -> dict[str, float]:
    """Giây mỗi round của từng pha, trung bình trên cả lượt, cùng phần "khác" của wall time."""
    timing = _timing(run)
    rounds = timing[-1]["round"]
    phases = {name: sum(t[f"{name}_s"] for t in timing) / rounds for name in PHASES}
    phases["other"] = max(0.0, timing[-1]["wall_time_s"] / rounds - sum(phases.values()))
    return phases


def _node_vector(run: Run, values: list[int], name: str) -> list[int]:
    n = run.config["num_nodes"]
    expected = n + 1 if run.config["topology"]["kind"] == "server" else n
    if len(values) != expected:
        raise AnalysisError(
            f"lượt {run.run_id}: {name} có {len(values)} phần tử, cần {expected} (N = {n}"
            f"{', thêm server ảo' if expected > n else ''})"
        )
    return values


def node_load(run: Run) -> NodeLoad:
    """Tải message và byte mỗi round của từng node, ở mốc cuối."""
    last = run.rounds[-1]
    if "node_messages_cum" not in last or "node_bytes_cum" not in last:
        raise AnalysisError(
            f"lượt {run.run_id} không có tải theo node (node_messages_cum, node_bytes_cum); lượt được ghi "
            "trước ADR-013, cần chạy lại ở code_version mới"
        )
    n, rounds = run.config["num_nodes"], last["round"]
    messages = np.array(_node_vector(run, last["node_messages_cum"], "node_messages_cum")) / rounds
    nbytes = np.array(_node_vector(run, last["node_bytes_cum"], "node_bytes_cum")) / rounds
    server = len(messages) > n
    return NodeLoad(
        messages=messages[:n],
        bytes=nbytes[:n],
        server_messages=float(messages[n]) if server else None,
        server_bytes=float(nbytes[n]) if server else None,
    )


def node_seconds_per_round(run: Run) -> tuple[np.ndarray, np.ndarray]:
    """(train, comm): giây mỗi round của từng node, N phần tử."""
    timing = _timing(run)
    if any("train_s_per_node" not in t or "comm_s_per_node" not in t for t in timing):
        raise AnalysisError(
            f"lượt {run.run_id} không có thời gian theo node (train_s_per_node, comm_s_per_node); lượt được "
            "ghi trước ADR-013, cần chạy lại ở code_version mới"
        )
    rounds = timing[-1]["round"]
    train = np.sum([t["train_s_per_node"] for t in timing], axis=0) / rounds
    comm = np.sum([t["comm_s_per_node"] for t in timing], axis=0) / rounds
    return train, comm


@dataclass(frozen=True)
class TimeSeries:
    """Thời gian của một lượt ở các mốc đánh giá (ADR-015)."""

    steps: np.ndarray
    rounds: np.ndarray
    # Giây tích lũy đo được của các pha train, comm, combine.
    measured: np.ndarray
    # Giây tích lũy của node huấn luyện chậm nhất trong từng khoảng giữa hai mốc.
    compute: np.ndarray
    # Byte (gửi + nhận) của node bận nhất trong từng khoảng, kể cả server ảo; không tích lũy.
    busiest_bytes: np.ndarray


def time_series(run: Run) -> TimeSeries:
    """Thời gian đo và các thành phần của thời gian ước tính, ở các mốc của rounds.jsonl."""
    timing = _timing(run)
    if [t["round"] for t in timing] != [r["round"] for r in run.rounds]:
        raise AnalysisError(
            f"lượt {run.run_id}: timing.jsonl không ghi đúng các round của rounds.jsonl, nên không ghép được "
            "thời gian với accuracy"
        )
    if any("train_s_per_node" not in t for t in timing) or any("node_bytes_cum" not in r for r in run.rounds):
        raise AnalysisError(
            f"lượt {run.run_id} không có thời gian huấn luyện hay tải byte theo node; lượt được ghi trước "
            "ADR-013, cần chạy lại ở code_version mới"
        )
    loads = np.array([r["node_bytes_cum"] for r in run.rounds], dtype=float)
    return TimeSeries(
        steps=np.array([r["local_steps"] for r in run.rounds]),
        rounds=np.array([r["round"] for r in run.rounds]),
        measured=np.cumsum([t["train_s"] + t["comm_s"] + t["combine_s"] for t in timing]),
        compute=np.cumsum([max(t["train_s_per_node"]) for t in timing]),
        busiest_bytes=np.diff(loads, axis=0, prepend=np.zeros((1, loads.shape[1]))).max(axis=1),
    )


def estimated_seconds(series: TimeSeries, *, mbit: float) -> np.ndarray:
    """Giây tích lũy của hệ phân tán ước tính ở băng thông `mbit` Mbit/s, tại từng mốc."""
    if not mbit > 0:
        raise ValueError(f"băng thông phải > 0, nhận {mbit}")
    return series.compute + np.cumsum(series.busiest_bytes) / (mbit * MBIT_BYTES)


def has_time(run: Run) -> bool:
    """Lượt có đủ dữ liệu cho time_series (timing.jsonl khớp rounds và các dãy theo node của ADR-013)."""
    return (
        bool(run.timing)
        and [t["round"] for t in run.timing] == [r["round"] for r in run.rounds]
        and all("train_s_per_node" in t for t in run.timing)
        and all("node_bytes_cum" in r for r in run.rounds)
    )


def single_runtime(runs: Sequence[Run]) -> str:
    """Runtime chung của các lượt: thời gian của inprocess và processes không cùng nghĩa (ADR-010,
    ADR-013)."""
    runtimes = sorted({run.config["runtime"] for run in runs})
    if len(runtimes) != 1:
        raise AnalysisError(
            f"một hình về chi phí giao tiếp trộn {len(runtimes)} runtime ({runtimes}); thời gian của chúng "
            "không cùng nghĩa (ADR-013)"
        )
    return runtimes[0]

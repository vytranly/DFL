"""
Khối 5 — metrics.py
Module Đánh giá: theo dõi F1-score, convergence speed,
communication cost, variance giữa các client theo từng round.
"""

import numpy as np
import pandas as pd


class MetricsTracker:
    """
    Ghi lại kết quả đánh giá sau mỗi round.

    Các thông số theo dõi:
      - mean_f1, std_f1       : F1-macro trung bình và độ lệch chuẩn giữa các client
      - mean_acc, std_acc     : Accuracy trung bình và độ lệch chuẩn
      - mean_loss             : Loss trung bình
      - comm_cost_mb          : Chi phí giao tiếp tích lũy (MB)
      - convergence_round     : Round đầu tiên đạt target F1
    """

    def __init__(self, num_clients: int, target_f1: float = 0.75):
        self.num_clients   = num_clients
        self.target_f1     = target_f1
        self.history       = []               # list of dict, 1 dict per round
        self.total_bytes   = 0                # tổng bytes đã truyền
        self._conv_round   = -1
        # Theo yeu cau co (43:45): do thoi gian thuc moi round
        self.total_time_s  = 0.0
        self.round_times   = []
        # Theo yeu cau co (52:22): tach comm cost rieng tung node
        self.node_comm_kb  = {}

    # ── Ghi kết quả 1 round ──────────────────────────────────────────
    def log_round(self, round_num: int, per_client_metrics: list, **kwargs):
        """
        per_client_metrics: list of (accuracy, f1, loss) — 1 tuple/client
        """
        accs   = [m[0] for m in per_client_metrics]
        f1s    = [m[1] for m in per_client_metrics]
        losses = [m[2] for m in per_client_metrics]

        mean_f1  = float(np.mean(f1s))
        mean_acc = float(np.mean(accs))

        # Kiểm tra convergence lần đầu
        if self._conv_round == -1 and mean_f1 >= self.target_f1:
            self._conv_round = round_num

        row = {
            "round"       : round_num,
            "mean_acc"    : mean_acc,
            "std_acc"     : float(np.std(accs)),
            "min_acc"     : float(np.min(accs)),
            "max_acc"     : float(np.max(accs)),
            "mean_f1"     : mean_f1,
            "std_f1"      : float(np.std(f1s)),
            "mean_loss"   : float(np.mean(losses)),
            "comm_cost_mb": self.total_bytes / (1024 ** 2),
            "round_time_s": kwargs.get("round_time_s", 0.0),
            "cum_time_s"  : self.total_time_s,
        }
        # Lưu F1 từng client riêng để vẽ variance heatmap
        for i, f1 in enumerate(f1s):
            row[f"f1_client_{i}"] = float(f1)
        for i, acc in enumerate(accs):
            row[f"acc_client_{i}"] = float(acc)

        self.history.append(row)

    # ── Cập nhật communication cost ───────────────────────────────────
    def add_comm_bytes(self, bytes_this_round: int):
        self.total_bytes += bytes_this_round

    def add_round_time(self, seconds: float):
        """Do thoi gian thuc 1 round de so sanh mang day vs thua.
        (Theo yeu cau co transcript 43:45)"""
        self.total_time_s += seconds
        self.round_times.append(seconds)

    def add_node_comm(self, node_id: int, kb: float):
        """Ghi comm cost rieng tung node (KB gui di).
        (Theo yeu cau co transcript 52:22 -- tach rieng comm cost)"""
        self.node_comm_kb[node_id] = self.node_comm_kb.get(node_id, 0.0) + kb

    def add_node_train_time(self, node_id: int, seconds: float):
        """Ghi training time rieng tung node (giay wall-clock).
        Training cost ty le voi (num_samples x local_epochs) -- node co
        nhieu mau hon se mat nhieu thoi gian hon (size imbalance ro rang).
        (Cho thay tac dung cua dirichlet_imbalanced, bo sung cho transcript 52:22)"""
        if not hasattr(self, 'node_train_time_s'):
            self.node_train_time_s = {}
        self.node_train_time_s[node_id] = \
            self.node_train_time_s.get(node_id, 0.0) + seconds

    def get_node_train_summary(self):
        """Tom tat training time tung node."""
        d = getattr(self, 'node_train_time_s', {})
        if not d:
            return {}
        avg = sum(d.values()) / len(d)
        return {
            "per_node_s": dict(d),
            "avg_s"     : round(avg, 4),
            "max_s"     : round(max(d.values()), 4),
            "min_s"     : round(min(d.values()), 4),
        }

    def get_node_comm_summary(self):
        """Tom tat comm cost tung node -- phan biet Hub vs Leaf."""
        if not self.node_comm_kb:
            return {}
        avg = sum(self.node_comm_kb.values()) / len(self.node_comm_kb)
        return {
            "per_node_kb": dict(self.node_comm_kb),
            "avg_kb"     : round(avg, 2),
            "max_kb"     : round(max(self.node_comm_kb.values()), 2),
            "min_kb"     : round(min(self.node_comm_kb.values()), 2),
        }

    # ── Truy vấn ─────────────────────────────────────────────────────
    @property
    def convergence_round(self) -> int:
        """Round đầu tiên đạt target_f1. -1 nếu chưa đạt."""
        return self._conv_round

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

    def save_csv(self, path: str):
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        print(f"[Metrics] Saved -> {path}")

    def summary(self) -> dict:
        """Tóm tắt kết quả cuối cùng."""
        if not self.history:
            return {}
        df = self.to_dataframe()
        last = self.history[-1]
        return {
            "final_f1"          : round(last["mean_f1"],  4),
            "final_acc"         : round(last["mean_acc"], 4),
            "final_std_f1"      : round(last["std_f1"],   4),
            "convergence_round" : self._conv_round,
            "total_comm_mb"     : round(self.total_bytes / (1024**2), 2),
            "best_f1"           : round(df["mean_f1"].max(), 4),
            "avg_round_time_s"  : round(sum(self.round_times)/max(len(self.round_times),1), 2),
            "total_time_s"      : round(self.total_time_s, 2),
        }

    def print_summary(self):
        s = self.summary()
        print("\n" + "="*50)
        print("  KẾT QUẢ TỔNG HỢP")
        print("="*50)
        print(f"  Final F1-macro     : {s.get('final_f1','N/A')}")
        print(f"  Final Accuracy     : {s.get('final_acc','N/A')}")
        print(f"  Final Std (F1)     : {s.get('final_std_f1','N/A')}")
        print(f"  Convergence round  : {s.get('convergence_round','N/A')}")
        print(f"  Total comm cost    : {s.get('total_comm_mb','N/A')} MB")
        print(f"  Best F1            : {s.get('best_f1','N/A')}")
        print(f"  Avg round time     : {s.get('avg_round_time_s','N/A')} s")
        print(f"  Total time         : {s.get('total_time_s','N/A')} s")
        print("="*50)

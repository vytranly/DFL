"""
Khối 6 — visualizer.py
Module Phân tích & Báo cáo: vẽ các biểu đồ phân tích kết quả.

Các biểu đồ:
  1. plot_f1_curves      — F1-score theo round (learning curve)
  2. plot_comm_cost      — Communication cost tích lũy theo round
  3. plot_variance       — Variance (std F1) giữa các client theo round
  4. plot_final_bar      — Bar chart so sánh final F1 nhiều thí nghiệm
  5. plot_client_heatmap — Heatmap F1 từng client theo round
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# ── 1. F1-score theo round ────────────────────────────────────────────
def plot_f1_curves(results: dict, title: str, save_path: str):
    """
    results: {label: DataFrame}  — mỗi DataFrame có cột 'round', 'mean_f1', 'std_f1'
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for label, df in results.items():
        ax.plot(df["round"], df["mean_f1"], marker='o', markersize=3,
                linewidth=1.8, label=label)
        if "std_f1" in df.columns:
            ax.fill_between(df["round"],
                            df["mean_f1"] - df["std_f1"],
                            df["mean_f1"] + df["std_f1"],
                            alpha=0.12)

    ax.axhline(0.75, color='red', linestyle='--', linewidth=1,
               label='Target F1=0.75')
    ax.set_xlabel("Round")
    ax.set_ylabel("F1-score (macro)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    _save(fig, save_path)


# ── 2. Communication cost ─────────────────────────────────────────────
def plot_comm_cost(results: dict, title: str, save_path: str):
    """
    results: {label: DataFrame}  — cột 'round', 'comm_cost_mb'
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    for label, df in results.items():
        ax.plot(df["round"], df["comm_cost_mb"], marker='s',
                markersize=3, linewidth=1.8, label=label)

    ax.set_xlabel("Round")
    ax.set_ylabel("Comm Cost (MB, tích lũy)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    _save(fig, save_path)


# ── 3. Variance giữa các client ───────────────────────────────────────
def plot_variance(results: dict, title: str, save_path: str):
    """
    results: {label: DataFrame}  — cột 'round', 'std_f1'
    Variance thấp = các client đồng đều = hệ thống hoạt động tốt.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    for label, df in results.items():
        ax.plot(df["round"], df["std_f1"], marker='D',
                markersize=3, linewidth=1.8, label=label)

    ax.set_xlabel("Round")
    ax.set_ylabel("Std F1 giữa các client")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    _save(fig, save_path)


# ── 4. Bar chart so sánh final F1 ────────────────────────────────────
def plot_final_bar(summaries: list, group_by: str, title: str, save_path: str):
    """
    summaries: list of dict, mỗi dict có key group_by và 'final_f1'
    group_by : tên cột làm nhãn (ví dụ 'algorithm', 'topology', 'alpha')
    """
    labels = [str(s[group_by]) for s in summaries]
    f1s    = [s["final_f1"]    for s in summaries]
    convs  = [s["convergence_round"] for s in summaries]

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # F1 bar
    bars = axes[0].bar(x, f1s, color='steelblue', edgecolor='black', alpha=0.8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=15)
    axes[0].set_ylabel("Final F1-macro")
    axes[0].set_title(f"{title} — Final F1")
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(0.75, color='red', linestyle='--', linewidth=1)
    for bar, v in zip(bars, f1s):
        axes[0].text(bar.get_x()+bar.get_width()/2, v+0.01,
                     f"{v:.3f}", ha='center', fontsize=8)

    # Convergence bar
    conv_vals = [c if c != -1 else 0 for c in convs]
    colors    = ['salmon' if c == -1 else 'mediumseagreen' for c in convs]
    bars2 = axes[1].bar(x, conv_vals, color=colors, edgecolor='black', alpha=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=15)
    axes[1].set_ylabel("Convergence Round")
    axes[1].set_title(f"{title} — Convergence Speed")
    for bar, v, c in zip(bars2, conv_vals, convs):
        label = "N/A" if c == -1 else str(c)
        axes[1].text(bar.get_x()+bar.get_width()/2, v+0.3,
                     label, ha='center', fontsize=8)

    plt.tight_layout()
    _save(fig, save_path)


# ── 5. Client heatmap ────────────────────────────────────────────────
def plot_client_heatmap(df: pd.DataFrame, num_clients: int,
                        title: str, save_path: str):
    """
    Heatmap F1 từng client theo round.
    Màu đậm = F1 cao, màu nhạt = F1 thấp.
    """
    client_cols = [f"f1_client_{i}" for i in range(num_clients)
                   if f"f1_client_{i}" in df.columns]
    if not client_cols:
        print("[visualizer] Không có cột f1_client_i, bỏ qua heatmap")
        return

    matrix = df[client_cols].T.values   # (num_clients, num_rounds)
    rounds = df["round"].tolist()

    fig, ax = plt.subplots(figsize=(max(8, len(rounds)//2), num_clients + 1))
    sns.heatmap(matrix, ax=ax,
                xticklabels=[str(r) for r in rounds],
                yticklabels=[f"Client {i}" for i in range(num_clients)],
                vmin=0, vmax=1, cmap="YlGn",
                annot=(len(rounds) <= 20), fmt=".2f", linewidths=0.3)
    ax.set_xlabel("Round")
    ax.set_ylabel("Client")
    ax.set_title(f"{title} — F1 từng client theo round")
    plt.tight_layout()
    _save(fig, save_path)


# ── Helper ────────────────────────────────────────────────────────────
def _save(fig, path: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"[Visualizer] Saved -> {path}")

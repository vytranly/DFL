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


# ── Hub vs Leaf comparison (Theo yeu cau co transcript 46:03) ────────
def plot_hub_leaf(df, nodes_info, title, save_path, client_sizes=None):
    """
    Ve bieu do so sanh F1 giua Hub node va Leaf node theo tung round.
    F1 tinh bang weighted mean theo so mau (neu co client_sizes),
    hoac simple mean neu khong co.
    """
    import numpy as np

    hub_ids  = [k for k,v in nodes_info.items() if v["type"] == "Hub"]
    leaf_ids = [k for k,v in nodes_info.items() if v["type"] == "Leaf"]
    rounds   = df["round"].tolist()

    def weighted_f1(ids):
        cols = [f"f1_client_{i}" for i in ids if f"f1_client_{i}" in df.columns]
        if not cols:
            return [0] * len(rounds)
        if client_sizes:
            weights = np.array([client_sizes.get(i, 1) for i in ids
                                 if f"f1_client_{i}" in df.columns], dtype=float)
            total_w = weights.sum()
            if total_w > 0:
                # weighted mean theo so mau
                result = []
                for _, row in df.iterrows():
                    f1_vals = np.array([row.get(f"f1_client_{i}", 0)
                                        for i in ids if f"f1_client_{i}" in df.columns])
                    result.append(float((f1_vals * weights).sum() / total_w))
                return result
        # fallback: simple mean
        return df[cols].mean(axis=1).tolist()

    def weighted_f1_all():
        """Weighted mean toan bo node."""
        all_ids = list(nodes_info.keys())
        cols    = [f"f1_client_{i}" for i in all_ids if f"f1_client_{i}" in df.columns]
        if not cols:
            return df["mean_f1"].tolist()
        if client_sizes:
            weights = np.array([client_sizes.get(i, 1) for i in all_ids
                                 if f"f1_client_{i}" in df.columns], dtype=float)
            total_w = weights.sum()
            if total_w > 0:
                result = []
                for _, row in df.iterrows():
                    f1_vals = np.array([row.get(f"f1_client_{i}", 0)
                                        for i in all_ids
                                        if f"f1_client_{i}" in df.columns])
                    result.append(float((f1_vals * weights).sum() / total_w))
                return result
        return df["mean_f1"].tolist()

    hub_f1   = weighted_f1(hub_ids)
    leaf_f1  = weighted_f1(leaf_ids)
    mean_all = weighted_f1_all()

    label_suffix = " (weighted)" if client_sizes else " (simple mean)"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    axes[0].plot(rounds, hub_f1,  "r-o", lw=2, markersize=4,
                 label=f"Hub  nodes {hub_ids}{label_suffix}")
    axes[0].plot(rounds, leaf_f1, "b--s", lw=2, markersize=4,
                 label=f"Leaf nodes {leaf_ids}{label_suffix}")
    axes[0].plot(rounds, mean_all, "g:", lw=1.5,
                 label=f"Mean (all){label_suffix}")
    axes[0].axhline(0.75, color="gray", ls="--", lw=1, label="Target F1=0.75")
    axes[0].set_xlabel("Round"); axes[0].set_ylabel("F1-score (macro)")
    axes[0].set_title("Hub vs Leaf: F1 theo Round")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3); axes[0].set_ylim(0, 1.05)

    # --- Plot 2: Bar chart F1 cuoi cung tung node ---
    last = df.iloc[-1]
    all_ids = sorted(nodes_info.keys())
    f1s     = [last.get(f"f1_client_{i}", 0) for i in all_ids]
    colors  = ["salmon" if nodes_info[i]["type"] == "Hub" else "steelblue"
               for i in all_ids]
    degs    = [nodes_info[i]["degree"] for i in all_ids]

    bars = axes[1].bar(all_ids, f1s, color=colors, edgecolor="black", alpha=0.85)
    axes[1].axhline(float(np.mean(f1s)), color="green", ls="--", lw=1.5,
                    label=f"Mean={np.mean(f1s):.3f}")
    axes[1].set_xlabel("Node ID"); axes[1].set_ylabel("F1 (final round)")
    axes[1].set_title("F1 tung node (Red=Hub, Blue=Leaf)")
    axes[1].legend(fontsize=8); axes[1].grid(True, axis="y", alpha=0.3)
    for bar, f1, d in zip(bars, f1s, degs):
        axes[1].text(bar.get_x()+bar.get_width()/2, f1+0.005,
                     f"{f1:.3f}\ndeg={d}", ha="center", fontsize=7)

    plt.tight_layout()
    _save(fig, save_path)


# ── Per-node Comm vs Training cost (bo sung cho transcript 52:22) ────
def plot_node_costs(nodes_info, node_comm_kb, node_train_s, client_sizes,
                    title, save_path):
    """
    Ve 2 subplot bar chart: comm cost va training cost tung node.

    Yeu cau (notification): comm cost va training cost la 2 TRUC DOC LAP --
      - Comm theo DEGREE (topology): Hub cao, Leaf thap
      - Training theo SO MAU (data size): node nhieu mau cao, it mau thap
    Sap xep node theo degree giam dan (Hub truoc): bar Comm se giam dan,
    nhung bar Training KHONG giam dan (vi training theo size, khong theo degree)
    -> khoi "hub it mau = comm cao + training thap" va
       "leaf nhieu mau = comm thap + training cao" hien ro.
    Mau do = Hub, mau xanh = Leaf.
    """
    # Sap xep node theo degree giam dan, roi theo so mau giam dan (Hub truoc)
    node_ids = sorted(nodes_info.keys(),
                      key=lambda i: (-nodes_info[i]['degree'],
                                     -client_sizes.get(i, 0)))
    comm_kb  = [node_comm_kb.get(i, 0.0) for i in node_ids]
    train_s  = [node_train_s.get(i, 0.0) for i in node_ids]
    sizes    = [client_sizes.get(i, 0) for i in node_ids]
    degs     = [nodes_info[i]['degree'] for i in node_ids]
    colors   = ["salmon" if nodes_info[i]["type"] == "Hub" else "steelblue"
                for i in node_ids]
    xlabels  = [f"N{i}\n{nodes_info[i]['type'][0]}{nodes_info[i]['degree']}"
                for i in node_ids]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    # --- Comm cost (KB) -- theo degree, giam dan ---
    bars = axes[0].bar(range(len(node_ids)), comm_kb, color=colors,
                       edgecolor="black", alpha=0.85)
    axes[0].set_xticks(range(len(node_ids)))
    axes[0].set_xticklabels(xlabels, fontsize=8)
    axes[0].set_xlabel("Node (sap xep theo degree giam dan -> Hub truoc)")
    axes[0].set_ylabel("Comm cost (KB tich luy)")
    axes[0].set_title("Communication cost -- theo DEGREE (Hub cao, Leaf thap)")
    axes[0].grid(True, axis="y", alpha=0.3)
    for bar, v, sz in zip(bars, comm_kb, sizes):
        axes[0].text(bar.get_x()+bar.get_width()/2, v+max(comm_kb)*0.01,
                     f"{v:.0f}\nn={sz}", ha="center", fontsize=6)

    # --- Training cost (s) -- theo so mau, KHONG theo degree ---
    bars2 = axes[1].bar(range(len(node_ids)), train_s, color=colors,
                        edgecolor="black", alpha=0.85)
    axes[1].set_xticks(range(len(node_ids)))
    axes[1].set_xticklabels(xlabels, fontsize=8)
    axes[1].set_xlabel("Node (cung thu tu -> de thay training KHONG theo degree)")
    axes[1].set_ylabel("Training time (s tich luy)")
    axes[1].set_title("Training cost -- theo SO MAU (node nhieu mau = cao)")
    axes[1].grid(True, axis="y", alpha=0.3)
    for bar, v, sz in zip(bars2, train_s, sizes):
        axes[1].text(bar.get_x()+bar.get_width()/2, v+max(train_s)*0.01,
                     f"{v:.2f}\nn={sz}", ha="center", fontsize=6)

    # Chu thich mau
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color='salmon', label='Hub'),
                            Patch(color='steelblue', label='Leaf')],
                   fontsize=8, loc='upper right')

    plt.tight_layout()
    _save(fig, save_path)


# ── Topology summary / Hub-Leaf classification ────────────────────────
# (Tách từ topology.py: logic báo cáo/topology-summary thuộc về
#  visualizer — module "Phân tích & Báo cáo"; topology.py giờ chỉ giữ
#  create_topology / get_neighbors / get_topology_info.)
def classify_nodes(graph):
    """
    Phân loại Hub node và Leaf node (theo yêu cầu co transcript 27:18 + 46:03).

    Định nghĩa:
      Hub node : nhiều kết nối (degree > avg_degree)
      Leaf node: ít kết nối    (degree <= avg_degree)

    Returns:
        dict : {node_id: {'degree': int, 'type': 'Hub' | 'Leaf'}}
        float: avg_degree
    """
    degrees = dict(graph.degree())
    n       = graph.number_of_nodes()
    avg_deg = sum(degrees.values()) / n if n > 0 else 0

    result = {}
    for node_id, deg in degrees.items():
        result[node_id] = {
            'degree': deg,
            'type'  : 'Hub' if deg > avg_deg else 'Leaf',
        }
    return result, avg_deg


def print_topology_summary(graph, name=""):
    """
    In tóm tắt topology ra màn hình (theo yêu cầu co transcript 27:18):
      - Phần trăm Hub node và Leaf node
      - Degree distribution
      - Spectral gap (đo tốc độ hội tụ của Gossip)
    """
    from topology import get_topology_info

    info       = get_topology_info(graph)
    nodes_info, avg_deg = classify_nodes(graph)
    hub_nodes  = [k for k, v in nodes_info.items() if v['type'] == 'Hub']
    leaf_nodes = [k for k, v in nodes_info.items() if v['type'] == 'Leaf']
    n          = info['num_nodes']

    # Spectral gap: tốc độ hội tụ của Gossip
    # (cần Mixing Matrix W -- tính xấp xỉ từ eigenvalue của W)
    try:
        from trainer import get_mixing_matrix
        W    = get_mixing_matrix(graph)
        eigs = np.sort(np.linalg.eigvalsh(W))[::-1]
        spectral_gap = float(1.0 - eigs[1]) if len(eigs) > 1 else 1.0
    except Exception:
        spectral_gap = None

    print(f"\n  [Topology: {name.upper() if name else 'UNKNOWN'}]")
    print(f"  Nodes={n}  Edges={info['num_edges']}  "
          f"Connected={info['is_connected']}")
    print(f"  Degree: avg={avg_deg:.2f}  "
          f"max={info['max_degree']}  min={info['min_degree']}")
    print(f"  Hub  nodes ({len(hub_nodes)}/{n} = {len(hub_nodes)/n*100:.0f}%): "
          f"{hub_nodes}")
    print(f"  Leaf nodes ({len(leaf_nodes)}/{n} = {len(leaf_nodes)/n*100:.0f}%): "
          f"{leaf_nodes}")
    if spectral_gap is not None:
        print(f"  Spectral gap = {spectral_gap:.4f}  "
              f"(lon = hoi tu nhanh, nho = hoi tu cham)")
    print(f"  Degree per node: "
          + "  ".join(f"N{k}={v['degree']}({v['type'][0]})"
                      for k, v in sorted(nodes_info.items())))

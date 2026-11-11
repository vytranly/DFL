"""
visualizer.py
Module Phan tich & Bao cao: ve cac bieu do phan tich ket qua.

Cac ham:
  1. plot_f1_curves        -- F1-score per round (learning curve)
  2. plot_comm_cost        -- Communication cost cumulative per round
  3. plot_variance         -- Variance (std F1) among clients per round
  4. plot_final_bar        -- Bar chart comparing final F1 across experiments
  5. plot_client_heatmap   -- Heatmap F1 per client per round
  6. plot_hub_leaf         -- Hub vs Leaf F1 curves
  7. plot_node_costs       -- Per-node comm vs training cost
  8. plot_full_report      -- All 10 standard plots for one sub-exp
  9. plot_combined_report  -- Overlay all sub-exps of a group
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


# ── Full Report (10 plots) for a single sub-exp ───────────────────────
def plot_full_report(tracker, nodes_info, client_sizes,
                     title_prefix, save_dir, prefix):
    """
    Generate all 10 standard plots for one sub-experiment.

    Plots produced (saved to save_dir/prefix_*.png):
      1. F1 curves per round with std shading
      2. Accuracy curves per round
      3. Variance (std F1) per round
      4. Communication cost cumulative per round
      5. Final F1 bar chart with convergence round annotation
      6. Client heatmap (F1 per client per round)
      7. Hub vs Leaf F1 curves (weighted by sample count)
      8. Per-node cost bar chart (comm KB vs train time)
      9. Round time per round (straggler effect)
      10. Convergence speed chart (rounds to reach F1 targets)

    Args:
        tracker      : MetricsTracker instance after a completed run
        nodes_info   : dict {node_id: {'degree': int, 'type': 'Hub'|'Leaf'}}
        client_sizes : dict {node_id: int} number of samples per client
        title_prefix : string prefix shown in every plot title
        save_dir     : directory to save PNG files
        prefix       : filename prefix (e.g. 'expa1_mnist')
    """
    os.makedirs(save_dir, exist_ok=True)
    df           = tracker.to_dataframe()
    s            = tracker.summary()
    num_clients  = tracker.num_clients
    node_comm_kb = tracker.node_comm_kb
    node_train_s = getattr(tracker, 'node_train_time_s', {})

    def p(name):
        return os.path.join(save_dir, f"{prefix}_{name}.png")

    # ── 1. F1 curves with std shading ───────────────────────────────
    plot_f1_curves(
        {title_prefix: df},
        f"{title_prefix} -- F1 per Round",
        p("f1_curves")
    )

    # ── 2. Accuracy curves ───────────────────────────────────────────
    _plot_acc_curves(df, title_prefix, p("acc_curves"))

    # ── 3. Variance (std F1) ─────────────────────────────────────────
    plot_variance(
        {title_prefix: df},
        f"{title_prefix} -- Variance (std F1) per Round",
        p("variance")
    )

    # ── 4. Communication cost cumulative ────────────────────────────
    plot_comm_cost(
        {title_prefix: df},
        f"{title_prefix} -- Cumulative Comm Cost (MB)",
        p("comm_cost")
    )

    # ── 5. Final F1 bar + convergence round ─────────────────────────
    summary_list = [dict(s, label=title_prefix)]
    _plot_final_bar_single(summary_list, title_prefix, p("final_bar"))

    # ── 6. Client heatmap ────────────────────────────────────────────
    plot_client_heatmap(df, num_clients,
                        f"{title_prefix} -- Client Heatmap",
                        p("heatmap"))

    # ── 7. Hub vs Leaf F1 curves ─────────────────────────────────────
    if nodes_info:
        plot_hub_leaf(df, nodes_info,
                      f"{title_prefix} -- Hub vs Leaf F1",
                      p("hub_leaf"),
                      client_sizes=client_sizes)

    # ── 8. Per-node cost bar chart ───────────────────────────────────
    if nodes_info and (node_comm_kb or node_train_s):
        plot_node_costs(nodes_info, node_comm_kb, node_train_s, client_sizes,
                        f"{title_prefix} -- Per-node Comm vs Train Cost",
                        p("node_costs"))

    # ── 9. Round time per round ──────────────────────────────────────
    _plot_round_time(df, title_prefix, p("round_time"))

    # ── 10. Convergence speed chart ──────────────────────────────────
    _plot_convergence_speed(df, title_prefix, p("conv_speed"))

    print(f"[plot_full_report] 10 plots saved to {save_dir}/{prefix}_*.png")


# ── Helper: Accuracy curves ────────────────────────────────────────────
def _plot_acc_curves(df, title_prefix, save_path):
    """
    Plot accuracy curves per round (mean_acc with std_acc shading).
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["round"], df["mean_acc"], marker='o', markersize=3,
            linewidth=1.8, color='steelblue', label="Mean Accuracy")
    if "std_acc" in df.columns:
        ax.fill_between(df["round"],
                        df["mean_acc"] - df["std_acc"],
                        df["mean_acc"] + df["std_acc"],
                        alpha=0.15, color='steelblue')
    ax.axhline(0.75, color='red', linestyle='--', linewidth=1,
               label='Target Acc=0.75')
    ax.set_xlabel("Round")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title_prefix} -- Accuracy per Round")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    _save(fig, save_path)


# ── Helper: Final F1 bar for a single exp ─────────────────────────────
def _plot_final_bar_single(summary_list, title_prefix, save_path):
    """
    Bar chart for a single experiment: final F1 and convergence round.
    summary_list: list with one dict containing 'final_f1', 'convergence_round'.
    """
    s    = summary_list[0]
    f1   = s.get("final_f1", 0.0)
    conv = s.get("convergence_round", -1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

    axes[0].bar(["Final F1"], [f1], color='steelblue',
                edgecolor='black', alpha=0.8)
    axes[0].axhline(0.75, color='red', linestyle='--', linewidth=1,
                    label='Target 0.75')
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("F1-macro")
    axes[0].set_title(f"{title_prefix} -- Final F1")
    axes[0].text(0, f1 + 0.02, f"{f1:.3f}", ha='center', fontsize=10)
    axes[0].legend(fontsize=8)

    conv_val   = conv if conv != -1 else 0
    conv_color = 'mediumseagreen' if conv != -1 else 'salmon'
    conv_label = f"Round {conv}" if conv != -1 else "Not reached"
    axes[1].bar(["Convergence"], [conv_val], color=conv_color,
                edgecolor='black', alpha=0.8)
    axes[1].set_ylabel("Round")
    axes[1].set_title(f"{title_prefix} -- Convergence (F1>=0.75)")
    axes[1].text(0, conv_val + 0.3, conv_label, ha='center', fontsize=9)

    plt.tight_layout()
    _save(fig, save_path)


# ── Helper: Round time per round ──────────────────────────────────────
def _plot_round_time(df, title_prefix, save_path):
    """
    Plot wall-clock time per round to highlight straggler effect.
    Uses 'round_time_s' column if present.
    """
    if "round_time_s" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(df["round"], df["round_time_s"], color='darkorange',
           edgecolor='black', alpha=0.75, width=0.6)
    mean_t = df["round_time_s"].mean()
    ax.axhline(mean_t, color='red', linestyle='--', linewidth=1.2,
               label=f"Mean={mean_t:.2f}s")
    ax.set_xlabel("Round")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title(f"{title_prefix} -- Round Time (Straggler Effect)")
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    _save(fig, save_path)


# ── Helper: Convergence speed (rounds to reach F1 targets) ────────────
def _plot_convergence_speed(df, title_prefix, save_path):
    """
    Bar chart: rounds needed to first reach F1 targets [0.5, 0.6, 0.75].
    Shows 'N/A' (bar=0, red) when target is never reached.
    """
    targets = [0.5, 0.6, 0.75]
    conv_rounds = []
    for tgt in targets:
        hit = df[df["mean_f1"] >= tgt]["round"]
        conv_rounds.append(int(hit.iloc[0]) if len(hit) > 0 else -1)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    colors = ['mediumseagreen' if c != -1 else 'salmon' for c in conv_rounds]
    vals   = [c if c != -1 else 0 for c in conv_rounds]
    labels = [f"F1>={t}" for t in targets]
    bars   = ax.bar(labels, vals, color=colors, edgecolor='black', alpha=0.85)
    for bar, c in zip(bars, conv_rounds):
        txt = str(c) if c != -1 else "N/A"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                txt, ha='center', fontsize=10)
    ax.set_ylabel("Round")
    ax.set_title(f"{title_prefix} -- Convergence Speed")
    ax.grid(True, axis='y', alpha=0.3)
    _save(fig, save_path)


# ── Combined comparison plots (after a full group) ────────────────────
def plot_combined_report(all_results, all_summaries, group_key,
                         title, save_dir, prefix):
    """
    Overlay all sub-experiments of a group on shared axes.

    Args:
        all_results   : {label: DataFrame}
        all_summaries : list of summary dicts (each has group_key field)
        group_key     : str key to use as bar label (e.g. 'alpha', 'size_label')
        title         : overall title string
        save_dir      : directory to save PNGs
        prefix        : filename prefix (e.g. 'expa_combined_mnist')
    """
    os.makedirs(save_dir, exist_ok=True)

    def p(name):
        return os.path.join(save_dir, f"{prefix}_{name}.png")

    # F1 curves overlay
    plot_f1_curves(all_results,
                   f"{title} -- F1 Curves (all sub-exps)",
                   p("f1_curves"))

    # Variance overlay
    plot_variance(all_results,
                  f"{title} -- Variance (std F1) (all sub-exps)",
                  p("variance"))

    # Comm cost overlay
    plot_comm_cost(all_results,
                   f"{title} -- Cumulative Comm Cost (all sub-exps)",
                   p("comm_cost"))

    # Final F1 bar
    plot_final_bar(all_summaries, group_key,
                   f"{title} -- Final F1 Comparison",
                   p("final_bar"))

    # Convergence speed trend line (line chart across ordered group values)
    # Shows: rounds-to-target vs group variable (alpha / topology / size)
    _plot_convergence_trend(all_summaries, group_key,
                            f"{title} -- Convergence Speed Trend",
                            p("conv_trend"))

    # Round time trend (avg round time vs group variable -> straggler effect)
    _plot_round_time_trend(all_summaries, group_key,
                           f"{title} -- Avg Round Time Trend",
                           p("round_time_trend"))

    # Hub vs Leaf final F1 comparison across sub-exps
    # (requires hub_f1 / leaf_f1 in summaries if available)
    _plot_hub_leaf_trend(all_summaries, group_key,
                         f"{title} -- Hub vs Leaf F1 Trend",
                         p("hub_leaf_trend"))

    print(f"[plot_combined_report] 7 combined plots saved to {save_dir}/{prefix}_*.png")


def plot_cross_dataset(mnist_results, cifar_results,
                       mnist_summaries, cifar_summaries,
                       group_key, title, save_dir, prefix):
    """
    So sanh tuong quan MNIST vs CIFAR-10 tren cung thi nghiem.
    Sinh 3 bieu do dat 2 dataset canh nhau:
      1. Final F1 bar: MNIST vs CIFAR grouped bar
      2. F1 curves overlay: MNIST (lien) vs CIFAR (dut) cung axes
      3. Convergence speed: MNIST vs CIFAR side-by-side
    """
    os.makedirs(save_dir, exist_ok=True)

    def p(name):
        return os.path.join(save_dir, f"{prefix}_cross_{name}.png")

    labels = [str(s[group_key]) for s in mnist_summaries]
    x      = np.arange(len(labels))
    width  = 0.35

    # ── 1. Final F1 grouped bar: MNIST vs CIFAR ──────────────────────
    mnist_f1s = [s.get("final_f1", 0.0) for s in mnist_summaries]
    cifar_f1s = [s.get("final_f1", 0.0) for s in cifar_summaries]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars1 = ax.bar(x - width/2, mnist_f1s, width,
                   label="MNIST",   color="steelblue", edgecolor="black", alpha=0.85)
    bars2 = ax.bar(x + width/2, cifar_f1s, width,
                   label="CIFAR-10", color="coral",    edgecolor="black", alpha=0.85)
    for bar, v in zip(bars1, mnist_f1s):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.01,
                f"{v:.3f}", ha="center", fontsize=7)
    for bar, v in zip(bars2, cifar_f1s):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.01,
                f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Final F1-macro"); ax.set_ylim(0, 1.1)
    ax.axhline(0.75, color="red", linestyle="--", linewidth=1, label="Target 0.75")
    ax.set_title(f"{title} -- Final F1: MNIST vs CIFAR-10")
    ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, p("final_f1_bar"))

    # ── 2. F1 curves: MNIST (solid) vs CIFAR (dashed) ────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = plt.cm.tab10.colors
    for idx, (lbl, s_m, s_c) in enumerate(
            zip(labels, mnist_summaries, cifar_summaries)):
        df_m = s_m.get("_df")
        df_c = s_c.get("_df")
        col  = colors[idx % len(colors)]
        if df_m is not None:
            ax.plot(df_m["round"], df_m["mean_f1"],
                    color=col, linestyle="-",  linewidth=1.8,
                    label=f"{lbl} MNIST")
        if df_c is not None:
            ax.plot(df_c["round"], df_c["mean_f1"],
                    color=col, linestyle="--", linewidth=1.8,
                    label=f"{lbl} CIFAR")
    ax.axhline(0.75, color="red", linestyle=":", linewidth=1, label="Target 0.75")
    ax.set_xlabel("Round"); ax.set_ylabel("F1-macro")
    ax.set_title(f"{title} -- F1 Curves: MNIST (solid) vs CIFAR (dashed)")
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    _save(fig, p("f1_curves"))

    # ── 3. Convergence speed: MNIST vs CIFAR side-by-side ────────────
    mnist_conv = [s.get("convergence_round", -1) for s in mnist_summaries]
    cifar_conv = [s.get("convergence_round", -1) for s in cifar_summaries]
    mnist_v = [c if c != -1 else 0 for c in mnist_conv]
    cifar_v = [c if c != -1 else 0 for c in cifar_conv]

    fig, ax = plt.subplots(figsize=(11, 5))
    b1 = ax.bar(x - width/2, mnist_v, width,
                label="MNIST",    color="steelblue", edgecolor="black", alpha=0.85)
    b2 = ax.bar(x + width/2, cifar_v, width,
                label="CIFAR-10", color="coral",     edgecolor="black", alpha=0.85)
    for bar, c in zip(b1, mnist_conv):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                str(c) if c != -1 else "N/A", ha="center", fontsize=8)
    for bar, c in zip(b2, cifar_conv):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                str(c) if c != -1 else "N/A", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Convergence Round (F1>=0.75)")
    ax.set_title(f"{title} -- Convergence Speed: MNIST vs CIFAR-10")
    ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, p("conv_speed"))

    print(f"[plot_cross_dataset] 3 cross-dataset plots -> {save_dir}/{prefix}_cross_*.png")


# ── Trend helpers for combined reports ───────────────────────────────

def _plot_convergence_trend(summaries, group_key, title, save_path):
    """
    Line chart: convergence round vs ordered group variable.
    Shows clearly how convergence speed degrades as Non-IID increases
    (or improves with better topology/algorithm).
    """
    labels  = [str(s[group_key]) for s in summaries]
    targets = {'F1>=0.50': 0.50, 'F1>=0.60': 0.60, 'F1>=0.75': 0.75}
    x       = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 5))
    for tgt_label, tgt_val in targets.items():
        conv_vals = []
        for s in summaries:
            df = s.get('_df')
            if df is not None:
                hit = df[df['mean_f1'] >= tgt_val]['round']
                conv_vals.append(int(hit.iloc[0]) if len(hit) > 0 else None)
            else:
                conv_vals.append(None)
        # Plot with None as gaps
        y_plot = [v if v is not None else float('nan') for v in conv_vals]
        ax.plot(x, y_plot, marker='o', linewidth=2, label=tgt_label)
        for xi, v in zip(x, conv_vals):
            lbl = str(v) if v is not None else 'N/A'
            ax.annotate(lbl, (xi, y_plot[xi] if not np.isnan(y_plot[xi]) else 0),
                        textcoords='offset points', xytext=(0, 6),
                        ha='center', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_xlabel(group_key)
    ax.set_ylabel("Convergence Round")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, save_path)


def _plot_round_time_trend(summaries, group_key, title, save_path):
    """
    Bar chart: avg round time vs group variable.
    Straggler effect visible when size imbalance increases.
    """
    labels    = [str(s[group_key]) for s in summaries]
    avg_times = [s.get('avg_round_time_s', 0.0) for s in summaries]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(labels, avg_times, color='coral', edgecolor='black', alpha=0.85)
    for bar, v in zip(bars, avg_times):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                f"{v:.1f}s", ha='center', fontsize=8)
    ax.set_xlabel(group_key)
    ax.set_ylabel("Avg Round Time (s)")
    ax.set_title(title)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    _save(fig, save_path)


def _plot_hub_leaf_trend(summaries, group_key, title, save_path):
    """
    Grouped bar chart: Hub final F1 vs Leaf final F1 per sub-exp.
    Shows whether Hub/Leaf gap changes with the group variable.
    Requires summaries to have 'hub_final_f1' and 'leaf_final_f1' keys.
    If missing, plots placeholder noting data not available.
    """
    labels    = [str(s[group_key]) for s in summaries]
    hub_f1s   = [s.get('hub_final_f1', None) for s in summaries]
    leaf_f1s  = [s.get('leaf_final_f1', None) for s in summaries]
    mean_f1s  = [s.get('final_f1', 0.0) for s in summaries]

    has_data = any(v is not None for v in hub_f1s)

    x    = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.bar(x - width, mean_f1s, width, label='Mean F1 (all)',
           color='steelblue', edgecolor='black', alpha=0.8)
    if has_data:
        hub_vals  = [v if v is not None else 0.0 for v in hub_f1s]
        leaf_vals = [v if v is not None else 0.0 for v in leaf_f1s]
        ax.bar(x,         hub_vals,  width, label='Hub F1',
               color='salmon',     edgecolor='black', alpha=0.8)
        ax.bar(x + width, leaf_vals, width, label='Leaf F1',
               color='lightblue',  edgecolor='black', alpha=0.8)
    else:
        ax.text(0.5, 0.5, 'Hub/Leaf data not available\n(scale_free topology needed)',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=12, color='gray')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_xlabel(group_key)
    ax.set_ylabel("F1-macro")
    ax.set_title(title)
    ax.set_ylim(0, 1.1)
    ax.axhline(0.75, color='red', linestyle='--', linewidth=1, label='Target 0.75')
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
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

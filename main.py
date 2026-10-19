"""
main.py — Điều phối toàn bộ 6 khối kiến trúc DFL.

Các thí nghiệm:
  --exp baseline   : IID + FC + Gossip (upper bound)
  --exp noniid     : sweep alpha [0.1, 0.5, 1.0, 5.0, 100.0]
  --exp topology   : Ring vs FC vs Random
  --exp algorithm  : Gossip vs FedAvg vs FedProx
  --exp all        : chạy tất cả
"""

import os
import copy
import argparse
import torch
import numpy as np

from config    import DEFAULT_CONFIG
from data_generator import build_dataloaders
from topology  import create_topology
from simulator import DFLSimulator
from visualizer import (plot_f1_curves, plot_comm_cost,
                        plot_variance, plot_final_bar,
                        plot_client_heatmap)

SAVE_DIR = "./results"
os.makedirs(SAVE_DIR, exist_ok=True)


# ── Helper ────────────────────────────────────────────────────────────
def make_cfg(**overrides):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(overrides)
    cfg["save_dir"] = SAVE_DIR
    return cfg


def run_single(cfg: dict):
    """Chạy 1 thí nghiệm, trả về MetricsTracker."""
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # Khối 1: Sinh dữ liệu Non-IID
    client_loaders, test_loader, client_indices, _ = build_dataloaders(
        dataset_name = cfg["dataset"],
        num_clients  = cfg["num_clients"],
        alpha        = cfg["alpha"],
        batch_size   = cfg["batch_size"],
        data_dir     = cfg["data_dir"],
    )

    # Khối 3: Tạo topology
    graph = create_topology(
        cfg["topology"],
        cfg["num_clients"],
        random_p = cfg.get("random_p", 0.5),
    )

    # Khối 2 + 4: Simulator (local train + aggregate)
    sim     = DFLSimulator(cfg, client_loaders, test_loader,
                           graph, client_indices)
    tracker = sim.run()

    return tracker


# ═══════════════════════════════════════════════════════════════════
# EXP 1 — Baseline: IID + Fully-Connected + Gossip
# ═══════════════════════════════════════════════════════════════════
def exp_baseline():
    print("\n" + "="*60)
    print("EXP 1 — Baseline: IID (alpha=100) + FC + Gossip")
    print("="*60)
    cfg     = make_cfg(alpha=100.0, topology="fully_connected",
                       algorithm="gossip", num_rounds=30)
    tracker = run_single(cfg)

    df = tracker.to_dataframe()
    tracker.save_csv(f"{SAVE_DIR}/exp1_baseline.csv")

    # Khối 6: Vẽ biểu đồ
    plot_f1_curves(
        {"Baseline (IID, FC, Gossip)": df},
        "Baseline — F1 theo Round",
        f"{SAVE_DIR}/exp1_f1_curve.png"
    )
    plot_comm_cost(
        {"Baseline": df},
        "Baseline — Communication Cost",
        f"{SAVE_DIR}/exp1_comm_cost.png"
    )
    return tracker


# ═══════════════════════════════════════════════════════════════════
# EXP 2 — Non-IID Sweep: vary alpha
# ═══════════════════════════════════════════════════════════════════
def exp_noniid():
    print("\n" + "="*60)
    print("EXP 2 — Non-IID Sweep: vary alpha (Ring + Gossip)")
    print("="*60)
    alphas    = [0.1, 0.5, 1.0, 5.0, 100.0]
    results   = {}
    summaries = []

    for alpha in alphas:
        print(f"\n  alpha = {alpha}")
        cfg     = make_cfg(alpha=alpha, topology="ring",
                           algorithm="gossip", num_rounds=50)
        tracker = run_single(cfg)
        label   = f"α={alpha}"
        results[label] = tracker.to_dataframe()
        s = tracker.summary(); s["alpha"] = alpha
        summaries.append(s)
        tracker.save_csv(f"{SAVE_DIR}/exp2_alpha{alpha}.csv")

    # Khối 6
    plot_f1_curves(results, "Non-IID Sweep — F1 (Ring + Gossip)",
                   f"{SAVE_DIR}/exp2_f1_curves.png")
    plot_comm_cost(results, "Non-IID Sweep — Comm Cost",
                   f"{SAVE_DIR}/exp2_comm_cost.png")
    plot_variance(results, "Non-IID Sweep — Variance giữa clients",
                  f"{SAVE_DIR}/exp2_variance.png")
    plot_final_bar(summaries, "alpha",
                   "Non-IID Sweep", f"{SAVE_DIR}/exp2_final_bar.png")
    return results, summaries


# ═══════════════════════════════════════════════════════════════════
# EXP 3 — Topology Sweep
# ═══════════════════════════════════════════════════════════════════
def exp_topology():
    print("\n" + "="*60)
    print("EXP 3 — Topology Sweep (alpha=0.5, Gossip)")
    print("="*60)
    topos     = ["ring", "fully_connected", "random"]
    results   = {}
    summaries = []

    for topo in topos:
        print(f"\n  topology = {topo}")
        cfg     = make_cfg(alpha=0.5, topology=topo,
                           algorithm="gossip", num_rounds=50)
        tracker = run_single(cfg)
        results[topo] = tracker.to_dataframe()
        s = tracker.summary(); s["topology"] = topo
        summaries.append(s)
        tracker.save_csv(f"{SAVE_DIR}/exp3_topo_{topo}.csv")

    # Khối 6
    plot_f1_curves(results, "Topology Sweep — F1 (alpha=0.5, Gossip)",
                   f"{SAVE_DIR}/exp3_f1_curves.png")
    plot_comm_cost(results, "Topology Sweep — Comm Cost",
                   f"{SAVE_DIR}/exp3_comm_cost.png")
    plot_final_bar(summaries, "topology",
                   "Topology Sweep", f"{SAVE_DIR}/exp3_final_bar.png")
    return results, summaries


# ═══════════════════════════════════════════════════════════════════
# EXP 4 — Algorithm Sweep: Gossip vs FedAvg vs FedProx
# ═══════════════════════════════════════════════════════════════════
def exp_algorithm():
    print("\n" + "="*60)
    print("EXP 4 — Algorithm Sweep (alpha=0.5, Ring)")
    print("="*60)
    # (algorithm, mu, label)
    algos = [
        ("gossip",  0.0,  "Gossip"),
        ("fedavg",  0.0,  "FedAvg"),
        ("fedprox", 0.01, "FedProx μ=0.01"),
        ("fedprox", 0.1,  "FedProx μ=0.1"),
    ]
    results   = {}
    summaries = []

    for algo, mu, label in algos:
        print(f"\n  {label}")
        cfg     = make_cfg(alpha=0.5, topology="ring",
                           algorithm=algo, mu=mu, num_rounds=50)
        tracker = run_single(cfg)
        results[label] = tracker.to_dataframe()
        s = tracker.summary(); s["algorithm"] = label
        summaries.append(s)
        tracker.save_csv(f"{SAVE_DIR}/exp4_{label.replace(' ','_')}.csv")

    # Khối 6
    plot_f1_curves(results, "Algorithm Sweep — F1 (alpha=0.5, Ring)",
                   f"{SAVE_DIR}/exp4_f1_curves.png")
    plot_variance(results, "Algorithm Sweep — Variance",
                  f"{SAVE_DIR}/exp4_variance.png")
    plot_final_bar(summaries, "algorithm",
                   "Algorithm Sweep", f"{SAVE_DIR}/exp4_final_bar.png")

    # Client heatmap cho Gossip
    num_clients = DEFAULT_CONFIG["num_clients"]
    plot_client_heatmap(results["Gossip"], num_clients,
                        "Gossip α=0.5 Ring",
                        f"{SAVE_DIR}/exp4_gossip_heatmap.png")
    return results, summaries


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DFL Experiments")
    parser.add_argument("--exp", default="all",
                        choices=["baseline", "noniid", "topology",
                                 "algorithm", "all"])
    args = parser.parse_args()

    if args.exp in ("baseline", "all"): exp_baseline()
    if args.exp in ("noniid",   "all"): exp_noniid()
    if args.exp in ("topology", "all"): exp_topology()
    if args.exp in ("algorithm","all"): exp_algorithm()

    print(f"\n[DONE] Kết quả lưu tại: {SAVE_DIR}/")

"""
main.py — Điều phối toàn bộ hệ thống DFL.

Thí nghiệm:
  --exp baseline   : IID + FC + Gossip (upper bound)
  --exp noniid     : Non-IID sweep alpha [0.1, 0.5, 1.0, 5.0, 100.0]
  --exp topology   : Ring vs FC vs Random
  --exp algorithm  : Gossip vs FedAvg vs FedProx
  --exp p2p        : P2P process thật (mỗi node = 1 process độc lập)
  --exp combined   : Dirichlet non-IID x Topology x FedProx (showcase thay đổi)
  --exp all        : chạy tất cả
"""

import os, copy, argparse
import torch, numpy as np

from config         import DEFAULT_CONFIG
from data_generator import build_dataloaders
from topology       import create_topology
from simulator      import DFLSimulator
from visualizer     import (plot_f1_curves, plot_comm_cost,
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
    """Chạy 1 thí nghiệm simulator, trả về MetricsTracker."""
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    client_loaders, test_loader, client_indices, _ = build_dataloaders(
        dataset_name   = cfg["dataset"],
        num_clients    = cfg["num_clients"],
        alpha          = cfg["alpha"],
        batch_size     = cfg["batch_size"],
        data_dir       = cfg["data_dir"],
        size_alpha     = cfg.get("size_alpha"),
        max_client_size= cfg.get("max_client_size"),
    )
    graph   = create_topology(cfg["topology"], cfg["num_clients"],
                               random_p=cfg.get("random_p", 0.5),
                               lattice_k=cfg.get("lattice_k", 4))
    sim     = DFLSimulator(cfg, client_loaders, test_loader,
                           graph, client_indices)
    return sim.run()


# ═══════════════════════════════════════════════════════════════════
# EXP 1 — Baseline: IID + FC + Gossip
# ═══════════════════════════════════════════════════════════════════
def exp_baseline():
    print("\n" + "="*60)
    print("EXP 1 — Baseline: IID (alpha=100) + FC + Gossip")
    print("="*60)
    cfg     = make_cfg(alpha=100.0, topology="fully_connected",
                       algorithm="gossip", num_rounds=20)
    tracker = run_single(cfg)
    df      = tracker.to_dataframe()
    tracker.save_csv(f"{SAVE_DIR}/exp1_baseline.csv")

    plot_f1_curves({"Baseline (IID, FC, Gossip)": df},
                   "Baseline — F1 theo Round",
                   f"{SAVE_DIR}/exp1_f1_curve.png")
    plot_comm_cost({"Baseline": df},
                   "Baseline — Communication Cost",
                   f"{SAVE_DIR}/exp1_comm_cost.png")
    return tracker


# ═══════════════════════════════════════════════════════════════════
# EXP 2 — Non-IID Sweep: 5 mức alpha
# ═══════════════════════════════════════════════════════════════════
def exp_noniid():
    print("\n" + "="*60)
    print("EXP 2 — Non-IID Sweep (Ring + Gossip, 5 alpha)")
    print("="*60)
    # Giảm xuống còn 5 alpha theo yêu cầu
    alphas    = [0.1, 0.5, 1.0, 5.0, 100.0]
    results   = {}
    summaries = []

    for alpha in alphas:
        print(f"\n  alpha = {alpha}")
        cfg     = make_cfg(alpha=alpha, topology="ring",
                           algorithm="gossip", num_rounds=30)
        tracker = run_single(cfg)
        label   = f"α={alpha}"
        results[label] = tracker.to_dataframe()
        s = tracker.summary(); s["alpha"] = alpha
        summaries.append(s)
        tracker.save_csv(f"{SAVE_DIR}/exp2_alpha{alpha}.csv")

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
    # Them lattice va scale_free (co yeu cau 00:57 + 27:18)
    topos = ["ring", "lattice", "fully_connected", "random", "scale_free"]
    results   = {}
    summaries = []

    for topo in topos:
        print(f"\n  topology = {topo}")
        extra = {"lattice_k": 4} if topo == "lattice" else {}
        cfg = make_cfg(alpha=0.5, topology=topo,
                       algorithm="gossip", num_rounds=30, **extra)
        tracker = run_single(cfg)
        results[topo] = tracker.to_dataframe()
        s = tracker.summary(); s["topology"] = topo
        summaries.append(s)
        tracker.save_csv(f"{SAVE_DIR}/exp3_topo_{topo}.csv")

    plot_f1_curves(results, "Topology Sweep — F1 (alpha=0.5)",
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
    # Theo yeu cau co (59:28): FedProx can alpha thap moi thay tac dung
    algos = [
        ("gossip",  0.0, 0.5, "Gossip a=0.5"),
        ("fedavg",  0.0, 0.5, "FedAvg a=0.5"),
        ("fedprox", 0.1, 0.5, "FedProx a=0.5"),
        ("fedprox", 0.1, 0.1, "FedProx a=0.1"),
    ]
    results   = {}
    summaries = []

    for algo, mu, alpha_val, label in algos:
        print(f"\n  {label}")
        cfg = make_cfg(alpha=alpha_val, topology="ring",
                       algorithm=algo, mu=mu, num_rounds=30)
        tracker = run_single(cfg)
        results[label] = tracker.to_dataframe()
        s = tracker.summary(); s["algorithm"] = label
        summaries.append(s)
        tracker.save_csv(f"{SAVE_DIR}/exp4_{label.replace(' ','_')}.csv")

    plot_f1_curves(results, "Algorithm Sweep — F1 (alpha=0.5, Ring)",
                   f"{SAVE_DIR}/exp4_f1_curves.png")
    plot_variance(results, "Algorithm Sweep — Variance",
                  f"{SAVE_DIR}/exp4_variance.png")
    plot_final_bar(summaries, "algorithm",
                   "Algorithm Sweep", f"{SAVE_DIR}/exp4_final_bar.png")
    plot_client_heatmap(results["Gossip"], DEFAULT_CONFIG["num_clients"],
                        "Gossip α=0.5 Ring",
                        f"{SAVE_DIR}/exp4_gossip_heatmap.png")
    return results, summaries


# ═══════════════════════════════════════════════════════════════════
# EXP 5 — P2P Process thật (mỗi node = 1 process độc lập, TCP socket)
# ═══════════════════════════════════════════════════════════════════
def exp_p2p():
    print("\n" + "="*60)
    print("EXP 5 — P2P Process thật (Ring + Gossip, alpha=0.5)")
    print("Mỗi node = 1 Python process riêng, giao tiếp TCP socket")
    print("="*60)

    from p2p.p2p_runner import run_p2p

    # Chạy Ring + Gossip
    run_p2p(
        topology     = "ring",
        algorithm    = "gossip",
        alpha        = 0.5,
        num_nodes    = 5,
        num_rounds   = 10,
        local_epochs = 2,
        lr           = 0.01,
        batch_size   = 32,
        eval_every   = 2,
        max_samples  = 800,   # giới hạn để chạy nhanh
        base_port    = 9100,
        results_dir  = SAVE_DIR,
        data_dir     = "./data",
    )

    # So sánh thêm FC + Gossip
    run_p2p(
        topology     = "fully_connected",
        algorithm    = "gossip",
        alpha        = 0.5,
        num_nodes    = 5,
        num_rounds   = 10,
        local_epochs = 2,
        lr           = 0.01,
        batch_size   = 32,
        eval_every   = 2,
        max_samples  = 800,
        base_port    = 9200,   # port khác tránh xung đột
        results_dir  = SAVE_DIR,
        data_dir     = "./data",
    )


# ═══════════════════════════════════════════════════════════════════
# EXP 6 — Combined Showcase: Dirichlet non-IID x Topology x FedProx
# ═══════════════════════════════════════════════════════════════════
def exp_combined():
    print("\n" + "="*60)
    print("COMBINED SHOWCASE — Dirichlet non-IID (alpha=0.1) x Topology x FedProx")
    print("4 topology (gồm MỚI: lattice, scale_free) | size imbalance (size_alpha=0.1)")
    print("="*60)
    # 4 topology (lattice/scale_free là mới thêm ở topology.py) dưới FedProx,
    # non-IID mạnh (alpha=0.1) + size imbalance (size_alpha=0.1) — chạy nhanh:
    # 10 client, 10 round, 1 local epoch.
    topos = [
        ("ring",            {}),                 # regular, degree 2
        ("fully_connected", {}),                 # dense, degree n-1
        ("lattice",         {"lattice_k": 4}),   # MỚI: Watts-Strogatz grid
        ("scale_free",      {}),                 # MỚI: Barabási-Albert hub/leaf
    ]
    results   = {}
    summaries = []
    trackers  = {}

    for topo, extra in topos:
        print(f"\n  >> topology = {topo}")
        cfg = make_cfg(alpha=0.1, topology=topo, algorithm="fedprox",
                       mu=0.1, size_alpha=0.1, max_client_size=3000,
                       num_clients=10, num_rounds=10,
                       local_epochs=1, batch_size=64, eval_every=2, **extra)
        tracker = run_single(cfg)
        trackers[topo] = tracker
        df      = tracker.to_dataframe()
        results[topo] = df
        s = tracker.summary(); s["topology"] = topo
        summaries.append(s)
        tracker.save_csv(f"{SAVE_DIR}/exp5_topo_{topo}.csv")

    # ── Biểu đồ so sánh ──
    plot_f1_curves(results,
                   "Combined Showcase — F1 (Dirichlet alpha=0.1, FedProx)",
                   f"{SAVE_DIR}/exp5_f1_curves.png")
    plot_variance(results,
                  "Combined Showcase — Variance giữa clients (alpha=0.1)",
                  f"{SAVE_DIR}/exp5_variance.png")
    plot_final_bar(summaries, "topology",
                   "Combined Showcase (alpha=0.1, FedProx)",
                   f"{SAVE_DIR}/exp5_final_bar.png")
    plot_comm_cost(results,
                   "Combined Showcase — Comm Cost (Dirichlet alpha=0.1, FedProx)",
                   f"{SAVE_DIR}/exp5_comm_cost.png")

    # ── Bảng tóm tắt so sánh (F1 + comm cost theo topology) ──
    print("\n" + "-"*70)
    print("  BẢNG SO SÁNH — Combined Showcase (Dirichlet alpha=0.1, FedProx)")
    print("-"*70)
    print(f"  {'Topology':<18}{'Edges':>7}{'Final F1':>10}{'Std F1':>9}"
          f"{'Conv.Rd':>9}{'Comm(MB)':>10}")
    print("  " + "-"*68)
    for topo, _ in topos:
        g = create_topology(topo, 10, lattice_k=4)
        s = next(x for x in summaries if x["topology"] == topo)
        conv = s.get("convergence_round", -1)
        conv_str = str(conv) if conv != -1 else "N/A"
        print(f"  {topo:<18}{g.number_of_edges():>7}{s['final_f1']:>10.4f}"
              f"{s['final_std_f1']:>9.4f}{conv_str:>9}{s['total_comm_mb']:>10.2f}")
    print("-"*70)

    # ── Size imbalance + comm cost từng client (Hub vs Leaf) ──
    # (cho thấy size imbalance và chênh lệch comm cost giữa client/topology)
    print("  SIZE IMBALANCE & COMM COST TỪNG CLIENT (Hub vs Leaf)")
    print("  " + "-"*68)
    print(f"  {'Topology':<18}{'Size min/max':>16}{'Comm min/max (MB)':>22}")
    print("  " + "-"*68)
    for topo, _ in topos:
        tr   = trackers[topo]
        sz   = tr.client_sizes
        smin, smax = min(sz.values()), max(sz.values())
        comm = tr.get_node_comm_summary()
        cmin = comm.get("min_kb", 0.0) / 1024.0
        cmax = comm.get("max_kb", 0.0) / 1024.0
        print(f"  {topo:<18}{smin:>6}/{smax:<9}{cmin:>8.1f}/{cmax:<9.1f}")
    print("-"*70)
    print("  (Size min/max : mẫu của client nhỏ/lớn nhất — cho thấy size imbalance)")
    print("  (Comm min/max : comm cost node thấp/cao nhất — Hub gửi nhiều hơn Leaf)")
    print("="*60)
    return results, summaries


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DFL Experiments")
    parser.add_argument("--exp", default="all",
                        choices=["baseline", "noniid", "topology",
                                 "algorithm", "p2p", "combined", "all"])
    args = parser.parse_args()

    if args.exp in ("baseline",  "all"): exp_baseline()
    if args.exp in ("noniid",    "all"): exp_noniid()
    if args.exp in ("topology",  "all"): exp_topology()
    if args.exp in ("algorithm", "all"): exp_algorithm()
    if args.exp in ("p2p",       "all"): exp_p2p()
    if args.exp in ("combined",  "all"): exp_combined()

    print(f"\n[DONE] Kết quả lưu tại: {SAVE_DIR}/")

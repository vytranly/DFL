"""
main.py — Điều phối toàn bộ hệ thống DFL.

Thí nghiệm:
  --exp baseline   : IID + FC + Gossip (upper bound)
  --exp noniid     : Non-IID sweep alpha [0.1, 0.5, 1.0, 5.0, 100.0]
  --exp topology   : Ring vs FC vs Random
  --exp algorithm  : Gossip vs FedAvg vs FedProx
  --exp p2p        : P2P process thật (mỗi node = 1 process độc lập)
  --exp combined   : Scale-Free, per-node comm+training cost, FedProx ưu việt
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
                            plot_client_heatmap, plot_hub_leaf,
                            plot_node_costs)

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
        size_mode      = cfg.get("size_mode", "default"),
        max_diff       = cfg.get("max_diff", 1500),
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
# EXP 5 — Combined Showcase: Scale-Free topology, size imbalance,
#          per-node comm+training cost, FedProx vs FedAvg vs Gossip
# ═══════════════════════════════════════════════════════════════════
def exp_combined():
    """
    Thí nghiệm đơn giản, nhanh — làm nổi bật 4 điểm thay đổi theo transcript:
      1. Hub/Leaf classification (transcript 27:18 + 46:03) — dùng scale_free
         vì Barabási-Albert tạo ra Hub thực sự (degree >> avg)
      2. Round time / dense-vs-sparse (transcript 43:45)
      3. Per-node comm cost (transcript 52:22) — Hub gửi nhiều hơn Leaf
      4. Per-node training cost (MỚI) — node có nhiều mẫu hơn tốn nhiều thời gian
         hơn (size imbalance qua dirichlet_imbalanced)
      5. FedProx ưu việt hơn FedAvg và Gossip dưới Non-IID mạnh (alpha=0.1)

    Cấu hình: scale_free, 10 clients, 10 rounds, 1 epoch, alpha=0.1,
              size_alpha=0.1, max_client_size=3000 -- chạy nhanh.
    """
    print("\n" + "="*65)
    print("EXP 5 — COMBINED SHOWCASE (Scale-Free | Non-IID alpha=0.1)")
    print("  Muc tieu: Hub/Leaf, per-node comm+train cost, FedProx uu viet")
    print("="*65)

    # ── Cấu hình chung ───────────────────────────────────────────────
    base_kw = dict(
        topology       = "scale_free",
        alpha          = 0.1,
        size_alpha     = 0.1,
        max_client_size= 3000,
        num_clients    = 10,
        num_rounds     = 10,
        local_epochs   = 1,
        batch_size     = 64,
        eval_every     = 2,
    )

    # ── Ba thuật toán cần so sánh ────────────────────────────────────
    algo_cfgs = [
        ("gossip",  0.0, "Gossip"),
        ("fedavg",  0.0, "FedAvg"),
        ("fedprox", 0.1, "FedProx (μ=0.1)"),
    ]

    results   = {}
    summaries = []
    trackers  = {}

    for algo, mu, label in algo_cfgs:
        print(f"\n  >> Algorithm = {label}")
        cfg = make_cfg(algorithm=algo, mu=mu, **base_kw)
        tracker = run_single(cfg)
        trackers[label] = tracker
        df = tracker.to_dataframe()
        results[label] = df
        s = tracker.summary(); s["algorithm"] = label
        summaries.append(s)
        tracker.save_csv(f"{SAVE_DIR}/exp5_{algo}.csv")

    # ── Lấy thông tin topology từ tracker đầu tiên ──────────────────
    first_tracker = trackers["Gossip"]
    nodes_info    = first_tracker.nodes_info   # dict từ classify_nodes()

    # ── Biểu đồ 1: F1 curves so sánh thuật toán ────────────────────
    plot_f1_curves(results,
                   "EXP5 — F1 curves (scale_free, alpha=0.1, size imbalance)",
                   f"{SAVE_DIR}/exp5_f1_curves.png")

    # ── Biểu đồ 2: Variance giữa clients ────────────────────────────
    plot_variance(results,
                  "EXP5 — Variance (std F1) giữa clients",
                  f"{SAVE_DIR}/exp5_variance.png")

    # ── Biểu đồ 3: Hub vs Leaf F1 (FedProx) ─────────────────────────
    prox_df = results["FedProx (μ=0.1)"]
    plot_hub_leaf(prox_df, nodes_info,
                  "EXP5 — Hub vs Leaf: F1 (FedProx, scale_free)",
                  f"{SAVE_DIR}/exp5_hub_leaf.png")

    # ── Biểu đồ 4: Per-node comm+train cost (FedProx run) ───────────
    prox_tracker = trackers["FedProx (μ=0.1)"]
    node_comm_kb  = prox_tracker.node_comm_kb
    node_train_s  = getattr(prox_tracker, 'node_train_time_s', {})
    client_sizes  = prox_tracker.client_sizes
    plot_node_costs(nodes_info, node_comm_kb, node_train_s, client_sizes,
                    "EXP5 — Per-node Comm vs Training cost (FedProx, scale_free)",
                    f"{SAVE_DIR}/exp5_node_costs.png")

    # ══════════════════════════════════════════════════════════════════
    # BẢNG 1 — Per-node cost (topology + size imbalance)
    # (Theo yêu cầu: node, type, degree, samples, comm_cost, train_time)
    # Sap xep theo degree giam dan (Hub truoc) -> Comm giam dan, Training KHONG
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─"*72)
    print("  BẢNG 1 — PER-NODE COST (FedProx run, scale_free, alpha=0.1)")
    print("  (sap xep theo degree giam dan -> Comm giam dan, Training KHONG)")
    print("─"*72)
    print(f"  {'Node':>4}  {'Type':>5}  {'Deg':>4}  {'Samples':>8}"
          f"  {'CommCost(KB)':>12}  {'TrainTime(s)':>12}")
    print("  " + "─"*68)

    # Sap xep theo degree giam dan, roi theo so mau giam dan (Hub truoc)
    node_ids = sorted(nodes_info.keys(),
                      key=lambda i: (-nodes_info[i]['degree'],
                                     -client_sizes.get(i, 0)))
    for nid in node_ids:
        ntype = nodes_info[nid]['type']
        deg   = nodes_info[nid]['degree']
        n_smp = client_sizes.get(nid, 0)
        ckb   = node_comm_kb.get(nid, 0.0)
        tts   = node_train_s.get(nid, 0.0)
        flag  = "<<HUB" if ntype == "Hub" else ""
        print(f"  {nid:>4}  {ntype:>5}  {deg:>4}  {n_smp:>8}"
              f"  {ckb:>12.1f}  {tts:>12.4f}  {flag}")

    comm_sum = prox_tracker.get_node_comm_summary()
    train_sum = prox_tracker.get_node_train_summary()
    print("  " + "─"*68)
    print(f"  Comm:  avg={comm_sum.get('avg_kb',0):.1f} KB  "
          f"max={comm_sum.get('max_kb',0):.1f} KB  "
          f"min={comm_sum.get('min_kb',0):.1f} KB")
    print(f"  Train: avg={train_sum.get('avg_s',0):.4f} s  "
          f"max={train_sum.get('max_s',0):.4f} s  "
          f"min={train_sum.get('min_s',0):.4f} s")

    # ── 2 ví dụ làm nổi bật 2 TRỤC ĐỘC LẬP ──────────────────────────
    # Hub có degree cao nhất (comm cao) -> xem training của nó
    hub_ex = max(nodes_info, key=lambda i: nodes_info[i]['degree']
                 if nodes_info[i]['type'] == 'Hub' else -1)
    # Leaf có training cao nhất (nhiều mẫu nhất) -> xem comm của nó
    big_leaf = max(
        (i for i in nodes_info if nodes_info[i]['type'] == 'Leaf'),
        key=lambda i: node_train_s.get(i, 0.0))

    print("  " + "─"*68)
    print(f"  >> Node {hub_ex} (Hub, deg={nodes_info[hub_ex]['degree']}, "
          f"{client_sizes.get(hub_ex,0)} mau): "
          f"Comm={node_comm_kb.get(hub_ex,0):.0f} KB (CAO) + "
          f"Train={node_train_s.get(hub_ex,0):.2f}s (THAP)")
    print(f"  >> Node {big_leaf} (Leaf, deg={nodes_info[big_leaf]['degree']}, "
          f"{client_sizes.get(big_leaf,0)} mau): "
          f"Comm={node_comm_kb.get(big_leaf,0):.0f} KB (THAP) + "
          f"Train={node_train_s.get(big_leaf,0):.2f}s (CAO)")
    print(f"  => Comm theo DEGREE (topology) | Training theo SO MAU (data size)")
    print(f"  => 2 truc DOC LAP: hub it mau = comm cao + training thap;")
    print(f"     leaf nhieu mau = comm thap + training cao")
    print("─"*72)

    # ══════════════════════════════════════════════════════════════════
    # BẢNG 2 — So sánh thuật toán (FedAvg vs Gossip vs FedProx)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─"*72)
    print("  BẢNG 2 — SO SÁNH THUẬT TOÁN (scale_free, alpha=0.1, size imbalance)")
    print("─"*72)
    print(f"  {'Algorithm':<20}  {'Final F1':>9}  {'Std F1':>8}"
          f"  {'Conv.Rd':>8}  {'AvgTime(s)':>11}")
    print("  " + "─"*68)

    for s in summaries:
        conv = s.get("convergence_round", -1)
        conv_str = str(conv) if conv != -1 else "N/A"
        avgt = s.get("avg_round_time_s", 0.0)
        print(f"  {s['algorithm']:<20}  {s['final_f1']:>9.4f}  "
              f"{s['final_std_f1']:>8.4f}  {conv_str:>8}  {avgt:>11.2f}")

    print("  " + "─"*68)
    # Best F1 và lowest variance -- lay tu data thuc te (khong hardcode)
    best_f1  = max(summaries, key=lambda x: x["final_f1"])
    low_var  = min(summaries, key=lambda x: x["final_std_f1"])
    print(f"  => Final F1 cao nhat : {best_f1['algorithm']} "
          f"(F1={best_f1['final_f1']:.4f})")
    print(f"  => Variance thap nhat: {low_var['algorithm']} "
          f"(std={low_var['final_std_f1']:.4f})")
    print(f"  => FedProx: tong hop proximal (w_avg - mu*(w_own - w_avg))")
    print(f"     keo weight tung node ve trung binh -> han che client drift")
    print(f"     under Non-IID manh (alpha=0.1) + size imbalance.")
    print(f"  (Luu y: 1 epoch + 10 round chua hoi tu -> F1 con thap,")
    print(f"   bieu do F1/variance curve cho thay su so sanh som.)")
    print("─"*72)

    print("\n  [EXP5] Ket qua luu: exp5_*.csv + exp5_*.png")
    print("="*65)
    return results, summaries


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════
# EXP 6 — Topology mới: Lattice & Scale-Free hội tụ thế nào?
# ═══════════════════════════════════════════════════════════════════
def exp_topology_long():
    """
    EXP 6 — Chứng minh topology.py: lattice + scale_free mới.
    Giữ cố định FedProx + alpha=0.1 để topology là biến duy nhất.
    10 clients, 50 rounds, 3 local epochs.
    """
    print("\n" + "="*60)
    print("EXP 6 — Topology mới (10 clients, 50 rounds, FedProx, alpha=0.1)")
    print("="*60)

    topos = ["ring", "lattice", "fully_connected", "random", "scale_free"]
    results   = {}
    summaries = []

    trackers6 = {}
    for topo in topos:
        print(f"\n  topology = {topo}")
        extra = {"lattice_k": 4} if topo == "lattice" else {}
        cfg = make_cfg(
            topology     = topo,
            algorithm    = "fedprox",
            mu           = 0.1,
            alpha        = 0.1,
            num_clients  = 10,
            num_rounds   = 50,
            local_epochs = 3,
            eval_every   = 5,
            **extra
        )
        tracker = run_single(cfg)
        trackers6[topo] = tracker
        results[topo] = tracker.to_dataframe()
        s = tracker.summary(); s["topology"] = topo
        summaries.append(s)
        tracker.save_csv(f"{SAVE_DIR}/exp6_topo_{topo}.csv")

    plot_f1_curves(results,
                   "EXP6 — Topology Sweep (FedProx, alpha=0.1, 10 clients)",
                   f"{SAVE_DIR}/exp6_f1_curves.png")
    plot_comm_cost(results,
                   "EXP6 — Comm Cost theo Topology",
                   f"{SAVE_DIR}/exp6_comm_cost.png")
    plot_final_bar(summaries, "topology",
                   "EXP6 — Final F1 theo Topology",
                   f"{SAVE_DIR}/exp6_final_bar.png")
    plot_variance(results,
                  "EXP6 — Variance (std F1) theo Topology",
                  f"{SAVE_DIR}/exp6_variance.png")

    # ── Per-node cost: chỉ scale_free mới có Hub/Leaf thật ──────────
    sf_tracker = trackers6.get("scale_free")
    if sf_tracker and hasattr(sf_tracker, "nodes_info"):
        nodes_info   = sf_tracker.nodes_info
        node_comm_kb = sf_tracker.node_comm_kb
        node_train_s = getattr(sf_tracker, "node_train_time_s", {})
        client_sizes = sf_tracker.client_sizes

        plot_node_costs(nodes_info, node_comm_kb, node_train_s, client_sizes,
                        "EXP6 — Per-node Comm vs Training cost (scale_free, FedProx)",
                        f"{SAVE_DIR}/exp6_node_costs_scalefree.png")

        print("\n  --- Per-node cost scale_free (EXP6) ---")
        print(f"  {'Node':>4}  {'Type':>5}  {'Deg':>3}  {'Samples':>7}"
              f"  {'Comm(KB)':>9}  {'Train(s)':>9}")
        print("  " + "-"*52)
        for nid in sorted(nodes_info, key=lambda i: -nodes_info[i]["degree"]):
            print(f"  {nid:>4}  {nodes_info[nid]['type']:>5}"
                  f"  {nodes_info[nid]['degree']:>3}"
                  f"  {client_sizes.get(nid,0):>7}"
                  f"  {node_comm_kb.get(nid,0):>9.1f}"
                  f"  {node_train_s.get(nid,0):>9.4f}")

    print("\n  [EXP6] Ket qua luu: exp6_*.csv + exp6_*.png")
    print("="*60)
    return results, summaries


# ═══════════════════════════════════════════════════════════════════
# EXP 7 — FedProx ưu việt dưới Non-IID mạnh + size imbalance
# ═══════════════════════════════════════════════════════════════════
def exp_fedprox_long():
    """
    EXP 7 — Chứng minh: trainer.py (FedProx) + data_generator.py (dirichlet_imbalanced).
    So sánh Gossip / FedAvg / FedProx(mu=0.1) trên scale_free, alpha=0.1.
    2 mức size_alpha: None (bằng nhau) vs 0.1 (lệch mạnh).
    10 clients, 50 rounds, 3 local epochs.
    """
    print("\n" + "="*60)
    print("EXP 7 — FedProx ưu việt (10 clients, 50 rounds, scale_free, alpha=0.1)")
    print("="*60)

    base_kw = dict(
        topology     = "scale_free",
        alpha        = 0.1,
        num_clients  = 10,
        num_rounds   = 50,
        local_epochs = 3,
        eval_every   = 5,
        batch_size   = 64,
    )

    # 2 mức size_alpha × 3 thuật toán = 6 run
    size_cfgs = [
        (None,  None,  "equal"),          # size bằng nhau
        (0.1,   3000,  "imbalanced"),     # lệch mạnh: ~20 vs 3000
    ]
    algos = [
        ("gossip",  0.0, "Gossip"),
        ("fedavg",  0.0, "FedAvg"),
        ("fedprox", 0.1, "FedProx"),
    ]

    all_results   = {}
    all_summaries = []
    trackers7     = {}   # lưu tracker để in per-node cost sau

    for size_alpha, max_cs, size_label in size_cfgs:
        print(f"\n  === size = {size_label} ===")
        for algo, mu, algo_label in algos:
            label = f"{algo_label} ({size_label})"
            print(f"    {label}")
            cfg = make_cfg(
                algorithm      = algo,
                mu             = mu,
                size_alpha     = size_alpha,
                max_client_size= max_cs,
                **base_kw
            )
            tracker = run_single(cfg)
            trackers7[label] = tracker
            all_results[label] = tracker.to_dataframe()
            s = tracker.summary()
            s["algorithm"]  = algo_label
            s["size_config"] = size_label
            all_results_label = label
            all_summaries.append(s)
            tracker.save_csv(
                f"{SAVE_DIR}/exp7_{algo}_{size_label}.csv")

    # Plot riêng theo từng size config để so sánh rõ
    for size_label in ["equal", "imbalanced"]:
        sub = {k: v for k, v in all_results.items() if size_label in k}
        plot_f1_curves(sub,
                       f"EXP7 — F1 curves ({size_label}, scale_free, alpha=0.1)",
                       f"{SAVE_DIR}/exp7_f1_{size_label}.png")
        plot_variance(sub,
                      f"EXP7 — Variance ({size_label})",
                      f"{SAVE_DIR}/exp7_variance_{size_label}.png")

    plot_final_bar(all_summaries, "algorithm",
                   "EXP7 — Final F1: Gossip vs FedAvg vs FedProx",
                   f"{SAVE_DIR}/exp7_final_bar.png")

    # ── Per-node cost: FedProx imbalanced — thú vị nhất ────────────
    # Thấy rõ: Hub comm cao/train rẻ vs Leaf comm thấp/train đắt
    # đồng thời thấy FedProx giảm gap Hub/Leaf F1 so với Gossip/FedAvg
    for size_label in ["equal", "imbalanced"]:
        key = f"FedProx ({size_label})"
        trk = trackers7.get(key)
        if trk and hasattr(trk, "nodes_info"):
            nodes_info   = trk.nodes_info
            node_comm_kb = trk.node_comm_kb
            node_train_s = getattr(trk, "node_train_time_s", {})
            client_sizes = trk.client_sizes

            plot_node_costs(nodes_info, node_comm_kb, node_train_s, client_sizes,
                            f"EXP7 — Per-node Cost FedProx ({size_label})",
                            f"{SAVE_DIR}/exp7_node_costs_{size_label}.png")

            print(f"\n  --- Per-node cost FedProx ({size_label}) ---")
            print(f"  {'Node':>4}  {'Type':>5}  {'Deg':>3}  {'Samples':>7}"
                  f"  {'Comm(KB)':>9}  {'Train(s)':>9}")
            print("  " + "-"*52)
            for nid in sorted(nodes_info, key=lambda i: -nodes_info[i]["degree"]):
                print(f"  {nid:>4}  {nodes_info[nid]['type']:>5}"
                      f"  {nodes_info[nid]['degree']:>3}"
                      f"  {client_sizes.get(nid,0):>7}"
                      f"  {node_comm_kb.get(nid,0):>9.1f}"
                      f"  {node_train_s.get(nid,0):>9.4f}")

    print("\n  [EXP7] Ket qua luu: exp7_*.csv + exp7_*.png")
    print("="*60)
    return all_results, all_summaries


# ═══════════════════════════════════════════════════════════════════
# EXP 8 — Straggler & Communication Cost dưới size imbalance
# ═══════════════════════════════════════════════════════════════════
def exp_straggler():
    """
    EXP 8 — Chứng minh: metrics.py (round_time 43:45, per-node comm 52:22,
    per-node train time) + data_generator.py (dirichlet_imbalanced).
    2 mức size_alpha: None (bằng nhau) vs 0.1 (lệch mạnh).
    Giữ cố định scale_free + FedProx để size là biến duy nhất.
    10 clients, 30 rounds, 3 local epochs.
    """
    print("\n" + "="*60)
    print("EXP 8 — Straggler & Per-node Cost (10 clients, 30 rounds)")
    print("="*60)

    base_kw = dict(
        topology     = "scale_free",
        algorithm    = "fedprox",
        mu           = 0.1,
        alpha        = 0.1,
        num_clients  = 10,
        num_rounds   = 30,
        local_epochs = 3,
        eval_every   = 5,
        batch_size   = 64,
    )

    size_cfgs = [
        (None,  None,  "equal"),      # baseline: size bằng nhau
        (0.1,   3000,  "imbalanced"), # lệch mạnh: ~20 vs 3000
    ]

    results   = {}
    summaries = []
    trackers  = {}

    for size_alpha, max_cs, size_label in size_cfgs:
        print(f"\n  === size_alpha = {size_alpha} ({size_label}) ===")
        cfg = make_cfg(
            size_alpha      = size_alpha,
            max_client_size = max_cs,
            **base_kw
        )
        tracker = run_single(cfg)
        trackers[size_label] = tracker
        results[size_label]  = tracker.to_dataframe()
        s = tracker.summary(); s["size_config"] = size_label
        summaries.append(s)
        tracker.save_csv(f"{SAVE_DIR}/exp8_size_{size_label}.csv")

    # F1 curves: so sánh equal vs imbalanced
    plot_f1_curves(results,
                   "EXP8 — F1: equal vs imbalanced size (scale_free, FedProx)",
                   f"{SAVE_DIR}/exp8_f1_curves.png")

    # Per-node cost cho từng size config
    for size_label, tracker in trackers.items():
        nodes_info   = tracker.nodes_info
        node_comm_kb = tracker.node_comm_kb
        node_train_s = getattr(tracker, "node_train_time_s", {})
        client_sizes = tracker.client_sizes

        plot_node_costs(
            nodes_info, node_comm_kb, node_train_s, client_sizes,
            f"EXP8 — Per-node Cost ({size_label}, scale_free, FedProx)",
            f"{SAVE_DIR}/exp8_node_costs_{size_label}.png"
        )

        # Bảng per-node in ra console
        print(f"\n  --- Per-node cost ({size_label}) ---")
        print(f"  {'Node':>4}  {'Type':>5}  {'Deg':>3}  {'Samples':>7}"
              f"  {'Comm(KB)':>9}  {'Train(s)':>9}")
        print("  " + "-"*52)
        node_ids = sorted(nodes_info.keys(),
                          key=lambda i: -nodes_info[i]["degree"])
        for nid in node_ids:
            print(f"  {nid:>4}  {nodes_info[nid]['type']:>5}"
                  f"  {nodes_info[nid]['degree']:>3}"
                  f"  {client_sizes.get(nid,0):>7}"
                  f"  {node_comm_kb.get(nid,0):>9.1f}"
                  f"  {node_train_s.get(nid,0):>9.4f}")

        comm_s  = tracker.get_node_comm_summary()
        train_s = tracker.get_node_train_summary()
        print(f"  Comm : avg={comm_s.get('avg_kb',0):.0f} KB"
              f"  max={comm_s.get('max_kb',0):.0f}"
              f"  min={comm_s.get('min_kb',0):.0f}")
        print(f"  Train: avg={train_s.get('avg_s',0):.3f}s"
              f"  max={train_s.get('max_s',0):.3f}"
              f"  min={train_s.get('min_s',0):.3f}")

    print("\n  [EXP8] Ket qua luu: exp8_*.csv + exp8_*.png")
    print("="*60)
    return results, summaries


# ═══════════════════════════════════════════════════════════════════
# EXP TEST — Quick FedProx fix verification
# ═══════════════════════════════════════════════════════════════════
def exp_test_fedprox():
    """
    Test nhanh sau khi fix fedprox_aggregate.
    scale_free, 5 clients, 20 rounds, 5 local epochs, alpha=0.05,
    max_client_size=300 (ít sample, chạy nhanh).
    So sánh Gossip / FedAvg / FedProx(mu=0.1).
    FedProx phải có F1 >= FedAvg ở cuối (hoặc converge ổn hơn).
    """
    print("\n" + "="*60)
    print("EXP TEST — FedProx fix check")
    print("  scale_free | 5 clients | 10 rounds | 3 epochs")
    print("  alpha=0.1  | max_samples=1000/client")
    print("="*60)

    base_kw = dict(
        topology        = "scale_free",
        alpha           = 0.1,
        size_alpha      = None,
        max_client_size = 1000,
        num_clients     = 5,
        num_rounds      = 6,
        local_epochs    = 3,
        eval_every      = 2,
        batch_size      = 32,
    )
    algos = [
        # ("gossip",  0.0, "Gossip"),
        # ("fedavg",  0.0, "FedAvg"),
        ("fedprox", 0.0, "FedProx μ=0.1"),
    ]

    results   = {}
    summaries = []
    for algo, mu, label in algos:
        print(f"\n  >> {label}")
        cfg     = make_cfg(algorithm=algo, mu=mu, **base_kw)
        tracker = run_single(cfg)
        df      = tracker.to_dataframe()
        results[label] = df
        s = tracker.summary(); s["algorithm"] = label
        summaries.append(s)
        tracker.save_csv(f"{SAVE_DIR}/exptest_{algo}.csv")

    print("\n" + "─"*55)
    print(f"  {'Algorithm':<18}  {'Final F1':>8}  {'Std F1':>7}  {'Conv.Rd':>7}")
    print("  " + "─"*51)
    for s in summaries:
        conv = s.get("convergence_round", -1)
        print(f"  {s['algorithm']:<18}  {s['final_f1']:>8.4f}"
              f"  {s['final_std_f1']:>7.4f}  {str(conv):>7}")
    best = max(summaries, key=lambda x: x["final_f1"])
    print(f"\n  => Best: {best['algorithm']} (F1={best['final_f1']:.4f})")
    print("─"*55)

    plot_f1_curves(results,
                   "EXP TEST — FedProx fix (scale_free, alpha=0.05)",
                   f"{SAVE_DIR}/exptest_f1.png")
    return results, summaries


# ═══════════════════════════════════════════════════════════════════
# EXP 9A/9B/9C — Benchmark 3 thuật toán, chạy song song
# scale_free, 8 clients, 5 epochs, 50 rounds, alpha=0.05
# size imbalance nhẹ: 300–1500 mẫu/client (biên độ ≤ 1500)
# FedProx chỉ dùng proximal ở aggregate (mu_local=0)
# Thể hiện: benchmark F1/round, training cost Hub vs Leaf
# ═══════════════════════════════════════════════════════════════════

# Cấu hình chung cho cả 3 exp
_EXP9_BASE = dict(
    topology        = "random",    # random: degree phân bố tự nhiên → hub/leaf rõ,
                                   # spectral gap cao hơn scale_free nhiều
    random_p        = 0.4,         # p=0.4 với 8 node → ~11 edges, gap ~0.3-0.5
    alpha           = 0.05,
    size_mode       = "bounded",
    max_diff        = 2500,
    num_clients     = 8,
    num_rounds      = 60,          # tăng lên 60 round
    local_epochs    = 5,
    eval_every      = 1,
    batch_size      = 32,
)


def _run_exp9(algo: str, mu: float, mu_local: float, mu_agg: float,
              label: str, prefix: str):
    """Helper chung cho exp9a/9b/9c/9d."""
    print("\n" + "="*62)
    print(f"  {prefix} — {label}")
    print(f"  scale_free | 8 clients | 50 rounds | 5 epochs")
    print(f"  alpha=0.05 | full MNIST (~57k) | diff<=2500")
    if algo == "fedprox":
        print(f"  mu_local={mu_local}  mu_agg={mu_agg}")
    print("="*62)

    cfg = make_cfg(algorithm=algo, mu=mu,
                   mu_local=mu_local, mu_agg=mu_agg,
                   **_EXP9_BASE)
    tracker = run_single(cfg)
    df      = tracker.to_dataframe()
    tracker.save_csv(f"{SAVE_DIR}/{prefix}_result.csv")

    plot_f1_curves({label: df},
                   f"{prefix} — F1 Benchmark mỗi round ({label})",
                   f"{SAVE_DIR}/{prefix}_f1_benchmark.png")
    plot_variance({label: df},
                  f"{prefix} — Variance F1 ({label})",
                  f"{SAVE_DIR}/{prefix}_variance.png")
    plot_comm_cost({label: df},
                   f"{prefix} — Communication Cost ({label})",
                   f"{SAVE_DIR}/{prefix}_comm_cost.png")

    nodes_info   = tracker.nodes_info
    node_comm_kb = tracker.node_comm_kb
    node_train_s = getattr(tracker, "node_train_time_s", {})
    client_sizes = tracker.client_sizes

    plot_node_costs(nodes_info, node_comm_kb, node_train_s, client_sizes,
                    f"{prefix} — Per-node Comm vs Training Cost ({label})",
                    f"{SAVE_DIR}/{prefix}_node_costs.png")
    plot_hub_leaf(df, nodes_info,
                  f"{prefix} — Hub vs Leaf F1 ({label})",
                  f"{SAVE_DIR}/{prefix}_hub_leaf.png",
                  client_sizes=client_sizes)

    s = tracker.summary()
    print(f"\n  Final F1={s['final_f1']:.4f}  Std={s['final_std_f1']:.4f}"
          f"  Conv.Round={s['convergence_round']}")
    print(f"\n  {'Node':>4}  {'Type':>5}  {'Deg':>3}  {'Samples':>7}"
          f"  {'Comm(KB)':>9}  {'Train(s)':>9}")
    print("  " + "-"*50)
    for nid in sorted(nodes_info, key=lambda i: -nodes_info[i]["degree"]):
        print(f"  {nid:>4}  {nodes_info[nid]['type']:>5}"
              f"  {nodes_info[nid]['degree']:>3}"
              f"  {client_sizes.get(nid,0):>7}"
              f"  {node_comm_kb.get(nid,0):>9.1f}"
              f"  {node_train_s.get(nid,0):>9.2f}")
    print(f"\n  [DONE] {prefix} -> {prefix}_*.csv + {prefix}_*.png")
    return tracker, df, s


def exp9a_gossip():
    """EXP 9A — Gossip (không có proximal)"""
    return _run_exp9("gossip", 0.0, 0.0, 0.0,
                     "Gossip", "exp9a_gossip")


def exp9b_fedavg():
    """EXP 9B — FedAvg (weighted mean, không có proximal)"""
    return _run_exp9("fedavg", 0.0, 0.0, 0.0,
                     "FedAvg", "exp9b_fedavg")


def exp9c_fedprox_local():
    """
    EXP 9C — FedProx proximal CHỈ trong local train.
    Mỗi gradient step kéo model về anchor (weight đầu round).
    Aggregate = FedAvg thuần (mu_agg=0).
    """
    return _run_exp9("fedprox", 0.1, 0.1, 0.0,
                     "FedProx local-train only (μ=0.1)",
                     "exp9c_fedprox_local")


def exp9d_fedprox_agg():
    """
    EXP 9D — Proximal Interpolation tại aggregate (custom DFL adaptation).
    KHÔNG phải FedProx gốc — đây là biến thể tự thiết kế cho DFL:
    Local train = SGD thuần (mu_local=0).
    Sau aggregate: w = (1-μ)*w_own + μ*w_avg  (kéo về consensus).
    Dùng để so sánh với exp9c (FedProx gốc).
    """
    return _run_exp9("fedprox", 0.1, 0.0, 0.1,
                     "Proximal Interpolation aggregate (μ=0.1)",
                     "exp9d_proximal_agg")


def exp9e_fedprox_fedavg():
    """
    EXP 9E — FedProx-FedAvg: công thức mới nghiệm dạng đóng.
    Local train = SGD thuần (mu_local=0).
    Aggregate: w_new = (1/(1+μ))×w_nb + (μ/(1+μ))×w_anchor
               w_nb = FedAvg(n_j/N) — base tốt hơn Gossip.
    So sánh với exp9b (FedAvg thuần) và exp9d (fedprox aggregate cũ).
    """
    return _run_exp9("fedprox_fedavg", 0.1, 0.0, 0.1,
                     "FedProx-FedAvg closed-form (μ=0.1)",
                     "exp9e_fedprox_fedavg")


def exp9f_fedprox_gossip():
    """
    EXP 9F — FedProx-Gossip: Gossip base + proximal closed-form.
    Local train = SGD thuần (mu_local=0).
    Aggregate: w_new = (1/(1+μ))×w_nb_gossip + (μ/(1+μ))×w_anchor
    Topology: random (vừa có hub vừa có leaf, spectral gap tốt hơn scale_free).
    60 rounds.
    """
    return _run_exp9("fedprox_gossip", 0.1, 0.0, 0.1,
                     "FedProx-Gossip closed-form (μ=0.1)",
                     "exp9f_fedprox_gossip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DFL Experiments")
    parser.add_argument("--exp", default="all",
                        choices=["baseline", "noniid", "topology",
                                 "algorithm", "p2p", "combined",
                                 "exp6", "exp7", "exp8", "test",
                                 "exp9a", "exp9b", "exp9c", "exp9d", "exp9e", "exp9f",
                                 "all_long", "all"])
    args = parser.parse_args()

    if args.exp in ("baseline",  "all"): exp_baseline()
    if args.exp in ("noniid",    "all"): exp_noniid()
    if args.exp in ("topology",  "all"): exp_topology()
    if args.exp in ("algorithm", "all"): exp_algorithm()
    if args.exp in ("p2p",       "all"): exp_p2p()
    if args.exp in ("combined",  "all"): exp_combined()
    if args.exp in ("exp6", "all_long"): exp_topology_long()
    if args.exp in ("exp7", "all_long"): exp_fedprox_long()
    if args.exp in ("exp8", "all_long"): exp_straggler()
    if args.exp in ("test",):            exp_test_fedprox()
    if args.exp in ("exp9a",):           exp9a_gossip()
    if args.exp in ("exp9b",):           exp9b_fedavg()
    if args.exp in ("exp9c",):           exp9c_fedprox_local()
    if args.exp in ("exp9d",):           exp9d_fedprox_agg()
    if args.exp in ("exp9e",):           exp9e_fedprox_fedavg()
    if args.exp in ("exp9f",):           exp9f_fedprox_gossip()

    print(f"\n[DONE] Kết quả lưu tại: {SAVE_DIR}/")

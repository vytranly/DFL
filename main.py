"""
main.py — Dieu phoi toan bo he thong DFL.

Thi nghiem:
  --exp exp_a1..exp_a5     : EXP A -- Non-IID alpha sweep (label skew)
  --exp exp_a_combined     : EXP A -- Tong hop sau khi a1-a5 xong
  --exp all_a              : Chay tuan tu a1->a2->a3->a4->a5->combined

  --exp exp_b1..exp_b4     : EXP B -- Quantity skew (size imbalance)
  --exp exp_b_combined     : EXP B -- Tong hop sau khi b1-b4 xong
  --exp all_b              : Chay tuan tu b1->b2->b3->b4->combined

  --exp exp_c1..exp_c4     : EXP C -- Label + Quantity skew ket hop
  --exp exp_c_combined     : EXP C -- Tong hop sau khi c1-c4 xong
  --exp all_c              : Chay tuan tu c1->c2->c3->c4->combined

  --exp exp_d1..exp_d5     : EXP D -- Topology sweep
  --exp exp_d_combined     : EXP D -- Tong hop sau khi d1-d5 xong
  --exp all_d              : Chay tuan tu d1->d2->d3->d4->d5->combined

  --exp exp_e1..exp_e3     : EXP E -- Algorithm sweep (Gossip vs FedAvg vs FedProx)
  --exp exp_e_combined     : EXP E -- Tong hop sau khi e1-e3 xong
  --exp all_e              : Chay tuan tu e1->e2->e3->combined

Vi du:
  python main.py --exp all_a --dataset mnist
  python main.py --exp all_a --dataset cifar10
  python main.py --exp all_a                   # ca 2 dataset
"""

import os, copy, argparse
import pandas as pd
import torch, numpy as np

from config         import DEFAULT_CONFIG
from data_generator import build_dataloaders
from topology       import create_topology
from simulator      import DFLSimulator
from visualizer     import (plot_f1_curves, plot_comm_cost,
                            plot_variance, plot_final_bar,
                            plot_client_heatmap, plot_hub_leaf,
                            plot_node_costs,
                            plot_full_report, plot_combined_report,
                            plot_cross_dataset)

SAVE_DIR = "./results"
os.makedirs(SAVE_DIR, exist_ok=True)


# ── Helper ────────────────────────────────────────────────────────────
def make_cfg(**overrides):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(overrides)
    cfg["save_dir"] = SAVE_DIR
    return cfg


def run_single(cfg: dict):
    """Chay 1 thi nghiem simulator, tra ve MetricsTracker."""
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


# ===================================================================
# SHARED SETTINGS for EXP A/B/C/D/E
# Run command for each sub-exp: python main.py --exp exp_a1
# ===================================================================

# Common kwargs for all new experiments
_NEW_EXP_BASE = dict(
    num_clients  = 10,
    num_rounds   = 50,
    local_epochs = 5,
    eval_every   = 1,
    batch_size   = 32,
    lr           = 0.01,
)

# Both datasets to compare
_DATASETS = [("mnist", "mnist"), ("cifar10", "cifar")]


def _run_new_exp(label, file_prefix, cfg_kwargs):
    """
    Helper: run one sub-exp for both mnist and cifar10.

    Saves CSV + 10 plots per dataset.
    Returns dict {dataset_key: (tracker, df, summary)}.
    """
    out = {}
    for ds_name, ds_tag in _DATASETS:
        pfx = f"{file_prefix}_{ds_tag}"
        print(f"\n  >> {label} | dataset={ds_name}")
        cfg     = make_cfg(dataset=ds_name, **cfg_kwargs)
        tracker = run_single(cfg)
        df      = tracker.to_dataframe()
        tracker.save_csv(f"{SAVE_DIR}/{pfx}_result.csv")

        nodes_info   = tracker.nodes_info
        client_sizes = tracker.client_sizes

        plot_full_report(
            tracker      = tracker,
            nodes_info   = nodes_info,
            client_sizes = client_sizes,
            title_prefix = f"{label} ({ds_name})",
            save_dir     = SAVE_DIR,
            prefix       = pfx,
        )
        s = tracker.summary()
        print(f"  F1={s['final_f1']:.4f}  Std={s['final_std_f1']:.4f}"
              f"  Conv={s['convergence_round']}")
        out[ds_name] = (tracker, df, s)
    return out


# ===================================================================
# EXP A -- Non-IID alpha sweep (label skew)
# Fixed: scale_free, fedprox, equal size (default), 10 clients
# Variable: alpha in [0.05, 0.1, 1.0, 50.0, 100.0]
# Run: python main.py --exp exp_a1  (through exp_a5)
# ===================================================================

_EXP_A_BASE = dict(
    topology  = "scale_free",
    algorithm = "fedprox",
    mu        = 0.1,
    mu_local  = 0.0,
    mu_agg    = 0.1,
    size_mode = "default",
    **_NEW_EXP_BASE,
)

_EXP_A_CONFIGS = [
    ("exp_a1", 0.05,  "alpha=0.05"),
    ("exp_a2", 0.1,   "alpha=0.1"),
    ("exp_a3", 1.0,   "alpha=1.0"),
    ("exp_a4", 50.0,  "alpha=50.0"),
    ("exp_a5", 100.0, "alpha=100.0"),
]


def _run_expa(idx):
    """
    Run one sub-exp from EXP A (Non-IID alpha sweep).
    idx: 0-based index into _EXP_A_CONFIGS (0=exp-a1, 4=exp-a5).
    """
    file_prefix, alpha_val, alpha_label = _EXP_A_CONFIGS[idx]
    exp_name = file_prefix.replace("_", "-")
    print("\n" + "="*62)
    print(f"  EXP A -- Non-IID alpha sweep | {exp_name} | {alpha_label}")
    print(f"  scale_free | fedprox | 10 clients | 50 rounds")
    print(f"  BOTH mnist and cifar10")
    print("="*62)
    cfg_kw = dict(alpha=alpha_val, **_EXP_A_BASE)
    return _run_new_exp(
        label       = f"EXP-A ({alpha_label})",
        file_prefix = file_prefix,
        cfg_kwargs  = cfg_kw,
    )


def exp_a1():
    """
    EXP A1 -- Non-IID cuc manh (alpha=0.05).
    Moi client chi co 1-2 class, gradient keo nguoc nhau hoan toan.

    Lenh chay:
      python main.py --exp exp_a1                  # ca 2 dataset
      python main.py --exp exp_a1 --dataset mnist   # chi MNIST (~1.5h)
      python main.py --exp exp_a1 --dataset cifar10 # chi CIFAR-10 (~2.5h)

    Output: results/exp_a1_mnist_*.png + results/exp_a1_cifar_*.png (10 bieu do moi)
    """
    return _run_expa(0)


def exp_a2():
    """
    EXP A2 -- Non-IID manh (alpha=0.1).
    Moi client tap trung 2-3 class chinh.

    Lenh chay:
      python main.py --exp exp_a2 --dataset mnist
      python main.py --exp exp_a2 --dataset cifar10
    """
    return _run_expa(1)


def exp_a3():
    """
    EXP A3 -- Non-IID trung binh (alpha=1.0).
    Diem be gay: tu day tro di F1 tang ro ret.

    Lenh chay:
      python main.py --exp exp_a3 --dataset mnist
      python main.py --exp exp_a3 --dataset cifar10
    """
    return _run_expa(2)


def exp_a4():
    """
    EXP A4 -- Gan IID (alpha=50.0).
    Phan phoi class kha dong deu, convergence tot.

    Lenh chay:
      python main.py --exp exp_a4 --dataset mnist
      python main.py --exp exp_a4 --dataset cifar10
    """
    return _run_expa(3)


def exp_a5():
    """
    EXP A5 -- IID hoan toan (alpha=100.0).
    Upper bound cua he thong. Dung lam baseline so sanh.

    Lenh chay:
      python main.py --exp exp_a5 --dataset mnist
      python main.py --exp exp_a5 --dataset cifar10
    """
    return _run_expa(4)


def _exp_a_combined(dataset_name, ds_tag):
    """
    Combined comparison plot for EXP A: overlay all 5 alpha values.
    Loads CSVs saved by exp_a1..exp_a5 and plots on shared axes.
    """
    all_results   = {}
    all_summaries = []
    for file_prefix, alpha_val, alpha_label in _EXP_A_CONFIGS:
        pfx = f"{file_prefix}_{ds_tag}"
        csv_path = f"{SAVE_DIR}/{pfx}_result.csv"
        if not os.path.exists(csv_path):
            print(f"  [skip] {csv_path} not found -- run sub-exps first")
            continue
        df = pd.read_csv(csv_path)
        all_results[alpha_label] = df
        last = df.iloc[-1]
        hit  = df[df["mean_f1"] >= 0.75]["round"]
        all_summaries.append({
            "alpha"             : str(alpha_val),
            "final_f1"          : float(last["mean_f1"]),
            "final_std_f1"      : float(last["std_f1"]),
            "convergence_round" : int(hit.iloc[0]) if len(hit) > 0 else -1,
            "avg_round_time_s"  : float(df["round_time_s"].mean()) if "round_time_s" in df.columns else 0.0,
            "_df"               : df,
        })
    if not all_results:
        return [], {}
    plot_combined_report(
        all_results   = all_results,
        all_summaries = all_summaries,
        group_key     = "alpha",
        title         = f"EXP A -- Non-IID alpha sweep ({dataset_name})",
        save_dir      = SAVE_DIR,
        prefix        = f"expa_combined_{ds_tag}",
    )
    return all_summaries, all_results


def exp_a_combined():
    """
    Tong hop EXP A: overlay 5 muc alpha tren cung bieu do.
    PHAI chay sau khi da co it nhat 1 sub-exp (exp_a1..exp_a5).
    Sub-exp nao chua chay se bi bo qua tu dong.

    Lenh chay (sau khi cac sub-exp xong):
      python main.py --exp exp_a_combined             # ca 2 dataset + cross-dataset
      python main.py --exp exp_a_combined --dataset mnist   # chi MNIST
      python main.py --exp exp_a_combined --dataset cifar10 # chi CIFAR

    Output (7 bieu do per dataset + 3 cross-dataset):
      results/expa_combined_mnist_*.png
      results/expa_combined_cifar_*.png
      results/expa_cross_*.png  (so sanh MNIST vs CIFAR-10)
    """
    print("\n" + "="*62)
    print("  EXP A -- Combined: overlay all 5 alpha values")
    print("="*62)
    summaries_per_ds = {}
    for ds_name, ds_tag in _DATASETS:
        s = _exp_a_combined(ds_name, ds_tag)
        summaries_per_ds[ds_tag] = s
    # Cross-dataset comparison
    if "mnist" in summaries_per_ds and "cifar" in summaries_per_ds:
        mnist_s, mnist_r = summaries_per_ds["mnist"]
        cifar_s, cifar_r = summaries_per_ds["cifar"]
        plot_cross_dataset(
            mnist_results=mnist_r, cifar_results=cifar_r,
            mnist_summaries=mnist_s, cifar_summaries=cifar_s,
            group_key="alpha",
            title="EXP A -- Non-IID alpha sweep",
            save_dir=SAVE_DIR, prefix="expa",
        )


def run_all_a():
    """
    Chay toan bo EXP A tuan tu: a1->a2->a3->a4->a5->combined.

    Lenh chay:
      python main.py --exp all_a --dataset mnist   # chay tuan tu a1->a5->combined
      python main.py --exp all_a --dataset cifar10
      python main.py --exp all_a                   # ca 2 dataset
    """
    print("\n" + "="*62)
    print("  RUN ALL A -- Chay toan bo EXP A tuan tu")
    print("  a1 -> a2 -> a3 -> a4 -> a5 -> combined")
    print("="*62)
    for fn in [exp_a1, exp_a2, exp_a3, exp_a4, exp_a5]:
        fn()
    exp_a_combined()


# ===================================================================
# EXP B -- Quantity skew (size imbalance)
# Fixed: scale_free, fedprox, alpha=100.0 (IID), 10 clients
# Variable: size imbalance level
# Run: python main.py --exp exp_b1  (through exp_b4)
# ===================================================================

_EXP_B_BASE = dict(
    topology  = "scale_free",
    algorithm = "fedprox",
    mu        = 0.1,
    mu_local  = 0.0,
    mu_agg    = 0.1,
    alpha     = 100.0,
    **_NEW_EXP_BASE,
)

_EXP_B_CONFIGS = [
    # (file_prefix, size_mode, max_diff, size_alpha, max_client_size, label)
    ("exp_b1", "default",    None,  None, None, "equal (default)"),
    ("exp_b2", "bounded",    1500,  None, None, "bounded max_diff=1500"),
    ("exp_b3", "bounded",    3000,  None, None, "bounded max_diff=3000"),
    ("exp_b4", "imbalanced", None,  0.1,  8000, "imbalanced alpha=0.1 max=8000"),
]


def _run_expb(idx):
    """
    Run one sub-exp from EXP B (quantity skew / size imbalance sweep).
    idx: 0-based index into _EXP_B_CONFIGS (0=exp-b1, 3=exp-b4).
    """
    file_prefix, size_mode, max_diff, size_alpha, max_cs, size_label = _EXP_B_CONFIGS[idx]
    exp_name = file_prefix.replace("_", "-")
    print("\n" + "="*62)
    print(f"  EXP B -- Quantity skew | {exp_name} | {size_label}")
    print(f"  scale_free | fedprox | alpha=100 (IID) | 10 clients")
    print(f"  BOTH mnist and cifar10")
    print("="*62)
    extra = {}
    if size_mode != "default":
        extra["size_mode"] = size_mode
    if max_diff is not None:
        extra["max_diff"] = max_diff
    if size_alpha is not None:
        extra["size_alpha"] = size_alpha
    if max_cs is not None:
        extra["max_client_size"] = max_cs
    cfg_kw = dict(**_EXP_B_BASE, **extra)
    return _run_new_exp(
        label       = f"EXP-B ({size_label})",
        file_prefix = file_prefix,
        cfg_kwargs  = cfg_kw,
    )


def exp_b1():
    """
    EXP B1 -- Size bang nhau (~6000 mau/client), label IID.
    Baseline khong co straggler. Training cost deu nhau.

    Lenh chay:
      python main.py --exp exp_b1 --dataset mnist
      python main.py --exp exp_b1 --dataset cifar10
    """
    return _run_expb(0)


def exp_b2():
    """
    EXP B2 -- Size lech nhe (max_diff=1500, ~5250-7500 mau/client).
    Straggler effect bat dau xuat hien nhe.

    Lenh chay:
      python main.py --exp exp_b2 --dataset mnist
      python main.py --exp exp_b2 --dataset cifar10
    """
    return _run_expb(1)


def exp_b3():
    """
    EXP B3 -- Size lech trung binh (max_diff=3000, ~4500-7500 mau/client).
    Round time tang ro vi straggler.

    Lenh chay:
      python main.py --exp exp_b3 --dataset mnist
      python main.py --exp exp_b3 --dataset cifar10
    """
    return _run_expb(2)


def exp_b4():
    """
    EXP B4 -- Size lech manh (size_alpha=0.1, ~200-8000 mau/client).
    Worst case: gap training cost >10x giua client lon nhat va nho nhat.

    Lenh chay:
      python main.py --exp exp_b4 --dataset mnist
      python main.py --exp exp_b4 --dataset cifar10
    """
    return _run_expb(3)


def _exp_b_combined(dataset_name, ds_tag):
    """
    Combined comparison plot for EXP B: overlay all 4 size configs.
    """
    all_results   = {}
    all_summaries = []
    for file_prefix, _sm, _md, _sa, _mc, size_label in _EXP_B_CONFIGS:
        pfx = f"{file_prefix}_{ds_tag}"
        csv_path = f"{SAVE_DIR}/{pfx}_result.csv"
        if not os.path.exists(csv_path):
            print(f"  [skip] {csv_path} not found -- run sub-exps first")
            continue
        df = pd.read_csv(csv_path)
        all_results[size_label] = df
        last = df.iloc[-1]
        hit  = df[df["mean_f1"] >= 0.75]["round"]
        all_summaries.append({
            "size_label"        : size_label,
            "final_f1"          : float(last["mean_f1"]),
            "final_std_f1"      : float(last["std_f1"]),
            "convergence_round" : int(hit.iloc[0]) if len(hit) > 0 else -1,
            "avg_round_time_s"  : float(df["round_time_s"].mean()) if "round_time_s" in df.columns else 0.0,
            "_df"               : df,
        })
    if not all_results:
        return [], {}
    plot_combined_report(
        all_results   = all_results,
        all_summaries = all_summaries,
        group_key     = "size_label",
        title         = f"EXP B -- Quantity skew ({dataset_name})",
        save_dir      = SAVE_DIR,
        prefix        = f"expb_combined_{ds_tag}",
    )
    return all_summaries, all_results


def exp_b_combined():
    """
    Tong hop EXP B: overlay 4 muc size imbalance.
    PHAI chay sau khi da co it nhat 1 sub-exp (exp_b1..exp_b4).

    Lenh chay:
      python main.py --exp exp_b_combined
      python main.py --exp exp_b_combined --dataset mnist

    Output: results/expb_combined_*.png + results/expb_cross_*.png
    """
    print("\n" + "="*62)
    print("  EXP B -- Combined: overlay all 4 size configs")
    print("="*62)
    summaries_per_ds = {}
    for ds_name, ds_tag in _DATASETS:
        s = _exp_b_combined(ds_name, ds_tag)
        summaries_per_ds[ds_tag] = s
    if "mnist" in summaries_per_ds and "cifar" in summaries_per_ds:
        mnist_s, mnist_r = summaries_per_ds["mnist"]
        cifar_s, cifar_r = summaries_per_ds["cifar"]
        if mnist_s and cifar_s:
            plot_cross_dataset(
                mnist_results=mnist_r, cifar_results=cifar_r,
                mnist_summaries=mnist_s, cifar_summaries=cifar_s,
                group_key="size_label",
                title="EXP B -- Quantity skew",
                save_dir=SAVE_DIR, prefix="expb",
            )


def run_all_b():
    """
    Chay toan bo EXP B tuan tu: b1->b2->b3->b4->combined.

    Lenh chay:
      python main.py --exp all_b --dataset mnist   # chay tuan tu b1->b4->combined
      python main.py --exp all_b --dataset cifar10
      python main.py --exp all_b                   # ca 2 dataset
    """
    print("\n" + "="*62)
    print("  RUN ALL B -- Chay toan bo EXP B tuan tu")
    print("  b1 -> b2 -> b3 -> b4 -> combined")
    print("="*62)
    for fn in [exp_b1, exp_b2, exp_b3, exp_b4]:
        fn()
    exp_b_combined()


# ===================================================================
# EXP C -- Label skew + Quantity skew combined
# Fixed: scale_free, fedprox, 10 clients
# Variable: (alpha, size) combinations
# Run: python main.py --exp exp_c1  (through exp_c4)
# ===================================================================

_EXP_C_BASE = dict(
    topology  = "scale_free",
    algorithm = "fedprox",
    mu        = 0.1,
    mu_local  = 0.0,
    mu_agg    = 0.1,
    **_NEW_EXP_BASE,
)

_EXP_C_CONFIGS = [
    # (file_prefix, alpha, size_mode, max_diff, size_alpha, max_cs, label)
    ("exp_c1", 0.05,  "default",    None, None, None,
     "label-skew only (alpha=0.05, equal)"),
    ("exp_c2", 100.0, "imbalanced", None, 0.1,  8000,
     "qty-skew only (alpha=100, imbalanced)"),
    ("exp_c3", 0.5,   "bounded",    2500, None, None,
     "both medium (alpha=0.5, max_diff=2500)"),
    ("exp_c4", 0.05,  "imbalanced", None, 0.1,  8000,
     "both worst (alpha=0.05, imbalanced)"),
]


def _run_expc(idx):
    """
    Run one sub-exp from EXP C (combined label + quantity skew).
    idx: 0-based index into _EXP_C_CONFIGS (0=exp-c1, 3=exp-c4).
    """
    (file_prefix, alpha_val, size_mode,
     max_diff, size_alpha, max_cs, comb_label) = _EXP_C_CONFIGS[idx]
    exp_name = file_prefix.replace("_", "-")
    print("\n" + "="*62)
    print(f"  EXP C -- Label+Qty skew | {exp_name} | {comb_label}")
    print(f"  scale_free | fedprox | 10 clients | 50 rounds")
    print(f"  BOTH mnist and cifar10")
    print("="*62)
    extra = {"alpha": alpha_val}
    if size_mode != "default":
        extra["size_mode"] = size_mode
    if max_diff is not None:
        extra["max_diff"] = max_diff
    if size_alpha is not None:
        extra["size_alpha"] = size_alpha
    if max_cs is not None:
        extra["max_client_size"] = max_cs
    cfg_kw = dict(**_EXP_C_BASE, **extra)
    return _run_new_exp(
        label       = f"EXP-C ({comb_label})",
        file_prefix = file_prefix,
        cfg_kwargs  = cfg_kw,
    )


def exp_c1():
    """
    EXP C1 -- Chi label skew (alpha=0.05, size equal).
    Baseline: chi co Non-IID label, khong co size imbalance.

    Lenh chay:
      python main.py --exp exp_c1 --dataset mnist
      python main.py --exp exp_c1 --dataset cifar10
    """
    return _run_expc(0)


def exp_c2():
    """
    EXP C2 -- Chi quantity skew (alpha=100 IID, size imbalanced).
    Baseline: chi co size lech, label IID.

    Lenh chay:
      python main.py --exp exp_c2 --dataset mnist
      python main.py --exp exp_c2 --dataset cifar10
    """
    return _run_expc(1)


def exp_c3():
    """
    EXP C3 -- Ca 2 muc trung binh (alpha=0.5, bounded max_diff=2500).
    Realistic: giong du lieu thuc te nhat.

    Lenh chay:
      python main.py --exp exp_c3 --dataset mnist
      python main.py --exp exp_c3 --dataset cifar10
    """
    return _run_expc(2)


def exp_c4():
    """
    EXP C4 -- Ca 2 muc nang nhat (alpha=0.05, size imbalanced).
    Worst case: label cuc Non-IID + size lech manh cung luc.

    Lenh chay:
      python main.py --exp exp_c4 --dataset mnist
      python main.py --exp exp_c4 --dataset cifar10
    """
    return _run_expc(3)


def _exp_c_combined(dataset_name, ds_tag):
    """
    Combined comparison plot for EXP C: overlay all 4 combinations.
    """
    all_results   = {}
    all_summaries = []
    for (file_prefix, _a, _sm, _md, _sa, _mc, comb_label) in _EXP_C_CONFIGS:
        pfx = f"{file_prefix}_{ds_tag}"
        csv_path = f"{SAVE_DIR}/{pfx}_result.csv"
        if not os.path.exists(csv_path):
            print(f"  [skip] {csv_path} not found -- run sub-exps first")
            continue
        df = pd.read_csv(csv_path)
        all_results[comb_label] = df
        last = df.iloc[-1]
        hit  = df[df["mean_f1"] >= 0.75]["round"]
        all_summaries.append({
            "comb_label"        : comb_label,
            "final_f1"          : float(last["mean_f1"]),
            "final_std_f1"      : float(last["std_f1"]),
            "convergence_round" : int(hit.iloc[0]) if len(hit) > 0 else -1,
            "avg_round_time_s"  : float(df["round_time_s"].mean()) if "round_time_s" in df.columns else 0.0,
            "_df"               : df,
        })
    if not all_results:
        return [], {}
    plot_combined_report(
        all_results   = all_results,
        all_summaries = all_summaries,
        group_key     = "comb_label",
        title         = f"EXP C -- Label+Qty skew ({dataset_name})",
        save_dir      = SAVE_DIR,
        prefix        = f"expc_combined_{ds_tag}",
    )
    return all_summaries, all_results


def exp_c_combined():
    """
    Tong hop EXP C: overlay 4 to hop label+size skew.
    PHAI chay sau khi da co it nhat 1 sub-exp (exp_c1..exp_c4).

    Lenh chay:
      python main.py --exp exp_c_combined
      python main.py --exp exp_c_combined --dataset mnist

    Output: results/expc_combined_*.png + results/expc_cross_*.png
    """
    print("\n" + "="*62)
    print("  EXP C -- Combined: label + quantity skew combinations")
    print("="*62)
    summaries_per_ds = {}
    for ds_name, ds_tag in _DATASETS:
        s = _exp_c_combined(ds_name, ds_tag)
        summaries_per_ds[ds_tag] = s
    if "mnist" in summaries_per_ds and "cifar" in summaries_per_ds:
        mnist_s, mnist_r = summaries_per_ds["mnist"]
        cifar_s, cifar_r = summaries_per_ds["cifar"]
        if mnist_s and cifar_s:
            plot_cross_dataset(
                mnist_results=mnist_r, cifar_results=cifar_r,
                mnist_summaries=mnist_s, cifar_summaries=cifar_s,
                group_key="comb_label",
                title="EXP C -- Label+Qty skew",
                save_dir=SAVE_DIR, prefix="expc",
            )


def run_all_c():
    """
    Chay toan bo EXP C tuan tu: c1->c2->c3->c4->combined.

    Lenh chay:
      python main.py --exp all_c --dataset mnist   # chay tuan tu c1->c4->combined
      python main.py --exp all_c --dataset cifar10
      python main.py --exp all_c                   # ca 2 dataset
    """
    print("\n" + "="*62)
    print("  RUN ALL C -- Chay toan bo EXP C tuan tu")
    print("  c1 -> c2 -> c3 -> c4 -> combined")
    print("="*62)
    for fn in [exp_c1, exp_c2, exp_c3, exp_c4]:
        fn()
    exp_c_combined()


# ===================================================================
# EXP D -- Topology sweep
# Fixed: fedprox, alpha=0.1, equal size, 10 clients
# Variable: topology
# Run: python main.py --exp exp_d1  (through exp_d5)
# ===================================================================

_EXP_D_BASE = dict(
    algorithm = "fedprox",
    mu        = 0.1,
    mu_local  = 0.0,
    mu_agg    = 0.1,
    alpha     = 0.1,
    size_mode = "default",
    **_NEW_EXP_BASE,
)

_EXP_D_CONFIGS = [
    # (file_prefix, topology, extra_kw, label)
    ("exp_d1", "ring",             {},               "ring"),
    ("exp_d2", "fully_connected",  {},               "fully_connected"),
    ("exp_d3", "random",           {"random_p": 0.4},"random (p=0.4)"),
    ("exp_d4", "lattice",          {"lattice_k": 4}, "lattice (k=4)"),
    ("exp_d5", "scale_free",       {},               "scale_free"),
]


def _run_expd(idx):
    """
    Run one sub-exp from EXP D (topology sweep).
    idx: 0-based index into _EXP_D_CONFIGS (0=exp-d1, 4=exp-d5).
    """
    file_prefix, topo, extra_kw, topo_label = _EXP_D_CONFIGS[idx]
    exp_name = file_prefix.replace("_", "-")
    print("\n" + "="*62)
    print(f"  EXP D -- Topology sweep | {exp_name} | {topo_label}")
    print(f"  fedprox | alpha=0.1 | equal size | 10 clients | 50 rounds")
    print(f"  BOTH mnist and cifar10")
    print("="*62)
    cfg_kw = dict(topology=topo, **_EXP_D_BASE, **extra_kw)
    return _run_new_exp(
        label       = f"EXP-D ({topo_label})",
        file_prefix = file_prefix,
        cfg_kwargs  = cfg_kw,
    )


def exp_d1():
    """
    EXP D1 -- Topology: ring.
    2 neighbor/node, spectral gap thap nhat -> hoi tu cham nhat.
    Comm cost thap nhat (2N edges).

    Lenh chay:
      python main.py --exp exp_d1 --dataset mnist
      python main.py --exp exp_d1 --dataset cifar10
    """
    return _run_expd(0)


def exp_d2():
    """
    EXP D2 -- Topology: fully_connected.
    Moi node noi voi tat ca -> spectral gap = 1.0 -> hoi tu nhanh nhat.
    Comm cost cao nhat (N*(N-1)/2 edges).

    Lenh chay:
      python main.py --exp exp_d2 --dataset mnist
      python main.py --exp exp_d2 --dataset cifar10
    """
    return _run_expd(1)


def exp_d3():
    """
    EXP D3 -- Topology: random (Erdos-Renyi p=0.4).
    Degree phan bo ngau nhien, co hub/leaf nhe. Trung binh.

    Lenh chay:
      python main.py --exp exp_d3 --dataset mnist
      python main.py --exp exp_d3 --dataset cifar10
    """
    return _run_expd(2)


def exp_d4():
    """
    EXP D4 -- Topology: lattice (Watts-Strogatz k=4).
    Mang luoi deu, moi node degree=4, khong co hub/leaf.

    Lenh chay:
      python main.py --exp exp_d4 --dataset mnist
      python main.py --exp exp_d4 --dataset cifar10
    """
    return _run_expd(3)


def exp_d5():
    """
    EXP D5 -- Topology: scale_free (Barabasi-Albert).
    Hub/Leaf ro rang nhat. Spectral gap thap -> hoi tu cham.
    Thich hop nhat de quan sat biet biet Hub vs Leaf F1.

    Lenh chay:
      python main.py --exp exp_d5 --dataset mnist
      python main.py --exp exp_d5 --dataset cifar10
    """
    return _run_expd(4)


def _exp_d_combined(dataset_name, ds_tag):
    """
    Combined comparison plot for EXP D: overlay all 5 topologies.
    """
    all_results   = {}
    all_summaries = []
    for file_prefix, _t, _e, topo_label in _EXP_D_CONFIGS:
        pfx = f"{file_prefix}_{ds_tag}"
        csv_path = f"{SAVE_DIR}/{pfx}_result.csv"
        if not os.path.exists(csv_path):
            print(f"  [skip] {csv_path} not found -- run sub-exps first")
            continue
        df = pd.read_csv(csv_path)
        all_results[topo_label] = df
        last = df.iloc[-1]
        hit  = df[df["mean_f1"] >= 0.75]["round"]
        all_summaries.append({
            "topology"          : topo_label,
            "final_f1"          : float(last["mean_f1"]),
            "final_std_f1"      : float(last["std_f1"]),
            "convergence_round" : int(hit.iloc[0]) if len(hit) > 0 else -1,
            "avg_round_time_s"  : float(df["round_time_s"].mean()) if "round_time_s" in df.columns else 0.0,
            "_df"               : df,
        })
    if not all_results:
        return [], {}
    plot_combined_report(
        all_results   = all_results,
        all_summaries = all_summaries,
        group_key     = "topology",
        title         = f"EXP D -- Topology sweep ({dataset_name})",
        save_dir      = SAVE_DIR,
        prefix        = f"expd_combined_{ds_tag}",
    )
    return all_summaries, all_results


def exp_d_combined():
    """
    Tong hop EXP D: overlay 5 topology.
    PHAI chay sau khi da co it nhat 1 sub-exp (exp_d1..exp_d5).

    Lenh chay:
      python main.py --exp exp_d_combined
      python main.py --exp exp_d_combined --dataset mnist

    Output: results/expd_combined_*.png + results/expd_cross_*.png
    """
    print("\n" + "="*62)
    print("  EXP D -- Combined: overlay all 5 topologies")
    print("="*62)
    summaries_per_ds = {}
    for ds_name, ds_tag in _DATASETS:
        s = _exp_d_combined(ds_name, ds_tag)
        summaries_per_ds[ds_tag] = s
    if "mnist" in summaries_per_ds and "cifar" in summaries_per_ds:
        mnist_s, mnist_r = summaries_per_ds["mnist"]
        cifar_s, cifar_r = summaries_per_ds["cifar"]
        if mnist_s and cifar_s:
            plot_cross_dataset(
                mnist_results=mnist_r, cifar_results=cifar_r,
                mnist_summaries=mnist_s, cifar_summaries=cifar_s,
                group_key="topology",
                title="EXP D -- Topology sweep",
                save_dir=SAVE_DIR, prefix="expd",
            )


def run_all_d():
    """
    Chay toan bo EXP D tuan tu: d1->d2->d3->d4->d5->combined.

    Lenh chay:
      python main.py --exp all_d --dataset mnist   # chay tuan tu d1->d5->combined
      python main.py --exp all_d --dataset cifar10
      python main.py --exp all_d                   # ca 2 dataset
    """
    print("\n" + "="*62)
    print("  RUN ALL D -- Chay toan bo EXP D tuan tu")
    print("  d1 -> d2 -> d3 -> d4 -> d5 -> combined")
    print("="*62)
    for fn in [exp_d1, exp_d2, exp_d3, exp_d4, exp_d5]:
        fn()
    exp_d_combined()


# -------------------------------------------------------------------
# EXP E -- Algorithm sweep: Gossip vs FedAvg vs FedProx
# Co dinh: scale_free, alpha=0.1, equal size, 10 clients, 50 rounds
# Thay doi: algorithm (3 thuat toan)
# -------------------------------------------------------------------

_EXP_E_BASE = dict(
    topology     = "scale_free",
    alpha        = 0.1,
    size_mode    = "default",
    num_clients  = 10,
    num_rounds   = 50,
    local_epochs = 5,
    eval_every   = 1,
    batch_size   = 32,
)

_EXP_E_CONFIGS = [
    # (file_prefix, algorithm, mu, algo_label)
    ("exp_e1", "gossip",  0.0, "Gossip"),
    ("exp_e2", "fedavg",  0.0, "FedAvg"),
    ("exp_e3", "fedprox", 0.1, "FedProx"),
]


def _run_expe(idx):
    """Run one sub-exp from EXP E (algorithm sweep)."""
    file_prefix, algo, mu, algo_label = _EXP_E_CONFIGS[idx]
    exp_name = file_prefix.replace("_", "-")
    print("\n" + "="*62)
    print(f"  EXP E -- Algorithm sweep | {exp_name} | {algo_label}")
    print(f"  scale_free | alpha=0.1 | 10 clients | 50 rounds")
    print(f"  BOTH mnist and cifar10")
    print("="*62)
    cfg_kw = dict(algorithm=algo, mu=mu, **_EXP_E_BASE)
    return _run_new_exp(
        label       = f"EXP-E ({algo_label})",
        file_prefix = file_prefix,
        cfg_kwargs  = cfg_kw,
    )


def exp_e1():
    """
    EXP E1 -- Gossip aggregation.
    Tong hop theo Mixing Matrix W (theo degree).
    Khong co co che chong client drift.

    Lenh chay:
      python main.py --exp exp_e1 --dataset mnist
      python main.py --exp exp_e1 --dataset cifar10
    """
    return _run_expe(0)


def exp_e2():
    """
    EXP E2 -- FedAvg aggregation.
    Tong hop theo so mau (weighted mean).
    Tot hon Gossip vi khong phu thuoc spectral gap.

    Lenh chay:
      python main.py --exp exp_e2 --dataset mnist
      python main.py --exp exp_e2 --dataset cifar10
    """
    return _run_expe(1)


def exp_e3():
    """
    EXP E3 -- FedProx aggregation (closed-form, mu=0.1).
    FedAvg + proximal term keo ve anchor -> han che client drift.
    Du kien tot nhat ve F1 va variance thap nhat.

    Lenh chay:
      python main.py --exp exp_e3 --dataset mnist
      python main.py --exp exp_e3 --dataset cifar10
    """
    return _run_expe(2)


def _exp_e_combined(dataset_name, ds_tag):
    """Load CSVs from exp_e1-e3 and generate combined plots."""
    all_results   = {}
    all_summaries = []
    for file_prefix, _a, _m, algo_label in _EXP_E_CONFIGS:
        pfx      = f"{file_prefix}_{ds_tag}"
        csv_path = f"{SAVE_DIR}/{pfx}_result.csv"
        if not os.path.exists(csv_path):
            print(f"  [skip] {csv_path} not found -- run sub-exps first")
            continue
        df   = pd.read_csv(csv_path)
        last = df.iloc[-1]
        hit  = df[df["mean_f1"] >= 0.75]["round"]
        all_results[algo_label] = df
        all_summaries.append({
            "algorithm"         : algo_label,
            "final_f1"          : float(last["mean_f1"]),
            "final_std_f1"      : float(last["std_f1"]),
            "convergence_round" : int(hit.iloc[0]) if len(hit) > 0 else -1,
            "avg_round_time_s"  : float(df["round_time_s"].mean()) if "round_time_s" in df.columns else 0.0,
            "_df"               : df,
        })
    if not all_results:
        return [], {}
    plot_combined_report(
        all_results   = all_results,
        all_summaries = all_summaries,
        group_key     = "algorithm",
        title         = f"EXP E -- Algorithm sweep ({dataset_name})",
        save_dir      = SAVE_DIR,
        prefix        = f"expe_combined_{ds_tag}",
    )
    return all_summaries, all_results


def exp_e_combined():
    """
    Tong hop EXP E: overlay 3 thuat toan tren cung bieu do.
    PHAI chay sau khi da co it nhat 1 sub-exp (exp_e1..exp_e3).

    Lenh chay:
      python main.py --exp exp_e_combined
      python main.py --exp exp_e_combined --dataset mnist

    Output: results/expe_combined_*.png + results/expe_cross_*.png
    """
    print("\n" + "="*62)
    print("  EXP E -- Combined: overlay 3 algorithms")
    print("="*62)
    summaries_per_ds = {}
    for ds_name, ds_tag in _DATASETS:
        s = _exp_e_combined(ds_name, ds_tag)
        summaries_per_ds[ds_tag] = s
    if "mnist" in summaries_per_ds and "cifar" in summaries_per_ds:
        mnist_s, mnist_r = summaries_per_ds["mnist"]
        cifar_s, cifar_r = summaries_per_ds["cifar"]
        if mnist_s and cifar_s:
            plot_cross_dataset(
                mnist_results=mnist_r, cifar_results=cifar_r,
                mnist_summaries=mnist_s, cifar_summaries=cifar_s,
                group_key="algorithm",
                title="EXP E -- Algorithm sweep",
                save_dir=SAVE_DIR, prefix="expe",
            )


def run_all_e():
    """
    Chay toan bo EXP E tuan tu: e1->e2->e3->combined.

    Lenh chay:
      python main.py --exp all_e --dataset mnist   # chay tuan tu e1->e3->combined
      python main.py --exp all_e --dataset cifar10
      python main.py --exp all_e                   # ca 2 dataset
    """
    print("\n" + "="*62)
    print("  RUN ALL E -- Chay toan bo EXP E tuan tu")
    print("  e1 -> e2 -> e3 -> combined")
    print("="*62)
    for fn in [exp_e1, exp_e2, exp_e3]:
        fn()
    exp_e_combined()


# ===================================================================
# Entry point
# ===================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DFL Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Vi du:
  python main.py --exp exp_a1 --dataset mnist
  python main.py --exp all_a --dataset mnist
  python main.py --exp all_a                  # ca 2 dataset
  python main.py --exp exp_e_combined
        """
    )
    parser.add_argument(
        "--exp", default="exp_a1",
        choices=[
            # EXP A
            "exp_a1", "exp_a2", "exp_a3", "exp_a4", "exp_a5",
            "exp_a_combined", "all_a",
            # EXP B
            "exp_b1", "exp_b2", "exp_b3", "exp_b4",
            "exp_b_combined", "all_b",
            # EXP C
            "exp_c1", "exp_c2", "exp_c3", "exp_c4",
            "exp_c_combined", "all_c",
            # EXP D
            "exp_d1", "exp_d2", "exp_d3", "exp_d4", "exp_d5",
            "exp_d_combined", "all_d",
            # EXP E
            "exp_e1", "exp_e2", "exp_e3",
            "exp_e_combined", "all_e",
        ],
        help="Ten thi nghiem can chay"
    )
    parser.add_argument(
        "--dataset", default="both",
        choices=["both", "mnist", "cifar10"],
        help="Dataset: both (default), mnist only, hoac cifar10 only"
    )
    args = parser.parse_args()

    # Filter _DATASETS based on --dataset flag
    import main as _self
    if args.dataset == "mnist":
        _self._DATASETS = [("mnist", "mnist")]
    elif args.dataset == "cifar10":
        _self._DATASETS = [("cifar10", "cifar")]
    # else: both (default, no change)

    # EXP A -- Non-IID alpha sweep
    if args.exp == "exp_a1":          exp_a1()
    elif args.exp == "exp_a2":        exp_a2()
    elif args.exp == "exp_a3":        exp_a3()
    elif args.exp == "exp_a4":        exp_a4()
    elif args.exp == "exp_a5":        exp_a5()
    elif args.exp == "exp_a_combined": exp_a_combined()
    elif args.exp == "all_a":         run_all_a()

    # EXP B -- Quantity skew
    elif args.exp == "exp_b1":        exp_b1()
    elif args.exp == "exp_b2":        exp_b2()
    elif args.exp == "exp_b3":        exp_b3()
    elif args.exp == "exp_b4":        exp_b4()
    elif args.exp == "exp_b_combined": exp_b_combined()
    elif args.exp == "all_b":         run_all_b()

    # EXP C -- Label + Quantity skew combined
    elif args.exp == "exp_c1":        exp_c1()
    elif args.exp == "exp_c2":        exp_c2()
    elif args.exp == "exp_c3":        exp_c3()
    elif args.exp == "exp_c4":        exp_c4()
    elif args.exp == "exp_c_combined": exp_c_combined()
    elif args.exp == "all_c":         run_all_c()

    # EXP D -- Topology sweep
    elif args.exp == "exp_d1":        exp_d1()
    elif args.exp == "exp_d2":        exp_d2()
    elif args.exp == "exp_d3":        exp_d3()
    elif args.exp == "exp_d4":        exp_d4()
    elif args.exp == "exp_d5":        exp_d5()
    elif args.exp == "exp_d_combined": exp_d_combined()
    elif args.exp == "all_d":         run_all_d()

    # EXP E -- Algorithm sweep
    elif args.exp == "exp_e1":        exp_e1()
    elif args.exp == "exp_e2":        exp_e2()
    elif args.exp == "exp_e3":        exp_e3()
    elif args.exp == "exp_e_combined": exp_e_combined()
    elif args.exp == "all_e":         run_all_e()

    print(f"\n[DONE] Ket qua luu tai: {SAVE_DIR}/")

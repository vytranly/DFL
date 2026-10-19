"""
config.py — Cấu hình mặc định cho toàn bộ hệ thống DFL.
Có thể override từng key khi gọi run_experiment().
"""
import torch

DEFAULT_CONFIG = {
    # ── Khối 1: Dữ liệu ──────────────────────────────────────────────
    "dataset"           : "mnist",       # "mnist" | "cifar10"
    "num_clients"       : 5,
    "alpha"             : 0.5,           # Dirichlet alpha (nhỏ = Non-IID mạnh)
    "batch_size"        : 32,
    "data_dir"          : "./data",

    # ── Khối 3: Topology ─────────────────────────────────────────────
    "topology"          : "ring",        # "ring" | "fully_connected" | "random"
    "random_p"          : 0.5,           # p cho Erdos-Renyi random graph

    # ── Khối 4: Thuật toán ───────────────────────────────────────────
    "algorithm"         : "gossip",      # "gossip" | "fedavg" | "fedprox"
    "mu"                : 0.1,           # FedProx proximal coefficient

    # ── Khối 2: Training ─────────────────────────────────────────────
    "num_rounds"        : 50,
    "local_epochs"      : 5,
    "lr"                : 0.01,

    # ── Khối 5: Đánh giá ────────────────────────────────────────────
    "eval_every"        : 5,
    "target_f1"         : 0.75,

    # ── Hệ thống ─────────────────────────────────────────────────────
    "seed"              : 42,
    "device"            : "cuda" if torch.cuda.is_available() else "cpu",
    "save_dir"          : "./results",
}

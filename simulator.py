"""
Khối 2 — simulator.py
Hệ thống mô phỏng DFL: điều phối toàn bộ vòng lặp training.

Mỗi round:
  1. (FedProx) Lưu anchor weight
  2. Tất cả client local_train song song (SGD / FedProx)
  3. Gossip Exchange: mỗi client lấy weight hàng xóm
  4. Gossip/FedAvg Aggregate: trộn weight
  5. Evaluate mỗi eval_every round
  6. Ghi metrics vào MetricsTracker
"""

import copy
import os
import time
import threading
import numpy as np
import torch
from tqdm import tqdm

from trainer import (get_model, get_model_size_kb,
                     local_train, gossip_aggregate, fedavg_aggregate,
                     fedprox_aggregate, fedprox_aggregate,
                     fedprox_gossip_aggregate,
                     evaluate, get_mixing_matrix)
from topology import get_neighbors
from metrics import MetricsTracker


class DFLSimulator:
    """
    Mô phỏng DFL với N client trên 1 topology cho trước.

    Args:
        config        : dict cấu hình (xem config.py)
        client_loaders: {client_id: DataLoader}
        test_loader   : DataLoader test chung (Global Test Set)
        graph         : NetworkX Graph (topology)
        client_indices: {client_id: list of indices} — để biết num_samples
    """

    def __init__(self, config: dict, client_loaders: dict,
                 test_loader, graph, client_indices: dict):
        self.cfg            = config
        self.client_loaders = client_loaders
        self.test_loader    = test_loader
        self.graph          = graph
        self.device         = config.get("device", "cpu")
        self.num_clients    = config["num_clients"]
        self.algorithm      = config.get("algorithm", "gossip")

        # GPU optimization: auto-tune cuDNN for faster conv
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True

        # ── Hybrid CPU+GPU: chia client giua 2 device ──────────────────
        # PyTorch nha GIL khi chay CUDA kernel -> 2 thread chay song song
        self.client_devices = {}
        if self.device == "cuda" and config.get("hybrid", False):
            n_gpu = max(1, int(np.ceil(0.6 * self.num_clients)))
            for cid in range(self.num_clients):
                self.client_devices[cid] = "cuda" if cid < n_gpu else "cpu"
        else:
            for cid in range(self.num_clients):
                self.client_devices[cid] = self.device

        # So mau tung client
        self.num_samples = {
            cid: len(client_indices[cid])
            for cid in range(self.num_clients)
        }

        # Khoi tao CNN cho tung client -- moi model tren device rieng
        self.models = {
            cid: get_model(config["dataset"]).to(self.client_devices[cid])
            for cid in range(self.num_clients)
        }

        # Mixing Matrix W từ topology
        self.W         = get_mixing_matrix(graph)
        self.neighbors = {cid: get_neighbors(graph, cid)
                          for cid in range(self.num_clients)}

        # Tính communication cost mỗi round
        sample_model      = get_model(config["dataset"])
        model_kb          = get_model_size_kb(sample_model)
        num_edges         = graph.number_of_edges()
        self.comm_bytes_per_round = int(num_edges * model_kb * 1024 * 2)

        # Per-node degree -- de tinh comm cost rieng tung node
        # (Theo yeu cau co 52:22: tach training cost va communication cost)
        self.node_degree = {cid: graph.degree(cid) for cid in range(self.num_clients)}
        self.model_kb    = model_kb

        # Phan loai Hub va Leaf node (Theo yeu cau co 27:18 + 46:03)
        # (classify_nodes / print_topology_summary da chuyen sang visualizer.py)
        from visualizer import classify_nodes, print_topology_summary
        self.nodes_info, self.avg_degree = classify_nodes(graph)
        print_topology_summary(graph, config.get('topology','unknown'))

        # Metrics tracker
        self.tracker = MetricsTracker(
            num_clients=self.num_clients,
            target_f1=config.get("target_f1", 0.75)
        )
        # Lưu kích thước mỗi client để exp5 báo cáo size imbalance
        self.tracker.client_sizes = dict(self.num_samples)
        # Lưu nodes_info (Hub/Leaf) lên tracker để main.py truy cập
        self.tracker.nodes_info = self.nodes_info

        print(f"\n{'='*55}")
        print(f"  DFL Simulator — {config.get('topology','ring').upper()}"
              f" | {self.algorithm.upper()}"
              f" | alpha={config.get('alpha', 0.5)}"
              f" | {self.num_clients} clients")
        print(f"  Model size : {model_kb:.1f} KB")
        print(f"  Comm/round : {self.comm_bytes_per_round/1024:.1f} KB")
        print(f"  Num edges  : {num_edges}")
        if self.device == "cuda" and config.get("hybrid", False):
            gpu_count = sum(1 for d in self.client_devices.values() if d == "cuda")
            cpu_count = self.num_clients - gpu_count
            print(f"  Hybrid    : {gpu_count} GPU + {cpu_count} CPU (parallel)")
        print(f"{'='*55}\n")

    # ── Một round ────────────────────────────────────────────────────
    def run_round(self, round_num: int):
        cfg = self.cfg
        lr      = cfg.get("lr", 0.01)
        epochs  = cfg.get("local_epochs", 5)
        mu      = cfg.get("mu", 0.0)
        # mu_local: hệ số proximal trong local_train
        # mu_agg  : hệ số proximal trong fedprox_aggregate
        # Tách riêng để exp9c (local only) và exp9d (aggregate only) khác nhau rõ
        mu_local = cfg.get("mu_local", mu)   # mặc định = mu
        mu_agg   = cfg.get("mu_agg",   mu)   # mặc định = mu

        # Bước 1: Lưu anchor cho FedProx (weight TRƯỚC local train)
        anchors = {}
        if self.algorithm == "fedprox":
            anchors = {
                cid: copy.deepcopy(self.models[cid].state_dict())
                for cid in range(self.num_clients)
            }

        # Bước 2: Local training (song song CPU+GPU neu hybrid)
        new_weights = {}
        def _train_client(cid):
            w_anchor = anchors.get(cid) if self.algorithm == "fedprox" else None
            state, _, _tt = local_train(
                model      = self.models[cid],
                dataloader = self.client_loaders[cid],
                epochs     = epochs,
                lr         = lr,
                device     = self.client_devices.get(cid, self.device),
                w_anchor   = w_anchor,
                mu         = mu_local if self.algorithm == "fedprox" else 0.0,
            )
            new_weights[cid] = (state, _tt)

        # Chia 2 nhom: GPU clients + CPU clients, chay song song
        gpu_cids = [c for c in range(self.num_clients)
                     if self.client_devices.get(c) == "cuda"]
        cpu_cids = [c for c in range(self.num_clients)
                     if self.client_devices.get(c) != "cuda"]

        if gpu_cids and cpu_cids:
            # Hybrid: 2 thread song song (GIL nha khi CUDA chay)
            t_gpu = threading.Thread(
                target=lambda: [_train_client(c) for c in gpu_cids])
            t_cpu = threading.Thread(
                target=lambda: [_train_client(c) for c in cpu_cids])
            t_gpu.start(); t_cpu.start()
            t_gpu.join(); t_cpu.join()
        else:
            # Single device: tuan tu
            for cid in range(self.num_clients):
                _train_client(cid)

        # Cap nhat weight sau local train
        for cid in range(self.num_clients):
            state, _tt = new_weights[cid]
            self.models[cid].load_state_dict(state)
            self.tracker.add_node_train_time(cid, _tt)

        # Bước 3 + 4: Exchange & Aggregate (tất cả trên CPU để trộn weight)
        aggregated = {}
        for cid in range(self.num_clients):
            nbs      = self.neighbors[cid]
            nb_w     = [{k: v.cpu() for k, v in self.models[j].state_dict().items()}
                        for j in nbs]
            own_w    = {k: v.cpu() for k, v in self.models[cid].state_dict().items()}

            if self.algorithm == "gossip":
                mixing = [self.W[cid][cid]] + [self.W[cid][j] for j in nbs]
                aggregated[cid] = gossip_aggregate(own_w, nb_w, mixing)

            elif self.algorithm == "fedprox_gossip":
                # FedProx-Gossip: Gossip base + proximal closed-form
                # w_new = (1/(1+μ))×w_nb_gossip + (μ/(1+μ))×w_anchor
                mixing = [self.W[cid][cid]] + [self.W[cid][j] for j in nbs]
                anchor_cpu = ({k: v.cpu() for k, v in anchors[cid].items()}
                              if mu_agg > 0 and cid in anchors else None)
                aggregated[cid] = fedprox_gossip_aggregate(
                    own_w, nb_w, mixing,
                    mu       = mu_agg,
                    w_anchor = anchor_cpu,
                )

            elif self.algorithm == "fedprox":
                nb_sizes = [self.num_samples[j] for j in nbs]
                anchor_cpu = ({k: v.cpu() for k, v in anchors[cid].items()}
                              if mu_agg > 0 and cid in anchors else None)
                # Truyền w_anchor (weight trước local train) vào aggregate
                # để áp dụng đúng proximal term: w - mu*(w_avg - w_anchor)
                aggregated[cid] = fedprox_aggregate(
                    own_w, nb_w,
                    [self.num_samples[cid]] + nb_sizes,
                    mu     = mu_agg,
                    w_anchor = anchor_cpu,
                )

            elif self.algorithm == "fedavg":
                # FedAvg: trọng số theo số mẫu
                nb_sizes  = [self.num_samples[j] for j in nbs]
                all_w     = [own_w] + nb_w
                all_sizes = [self.num_samples[cid]] + nb_sizes
                aggregated[cid] = fedavg_aggregate(all_w, all_sizes)

        # Nạp weight sau aggregate (chuyển về device của client)
        for cid in range(self.num_clients):
            agg_dev = self.client_devices.get(cid, self.device)
            if agg_dev != "cpu":
                aggregated[cid] = {k: v.to(agg_dev) for k, v in aggregated[cid].items()}
            self.models[cid].load_state_dict(aggregated[cid])

        # Comm cost tong mang
        self.tracker.add_comm_bytes(self.comm_bytes_per_round)

        # Per-node comm cost: Hub node nhieu ket noi gui nhieu hon Leaf
        # (Theo yeu cau co 52:22 + 46:03)
        for cid in range(self.num_clients):
            sent_kb = self.node_degree[cid] * self.model_kb
            self.tracker.add_node_comm(cid, sent_kb)

    # ── Đánh giá tất cả client ────────────────────────────────────────
    def eval_all(self):
        results = []
        for cid in range(self.num_clients):
            acc, f1, loss = evaluate(
                self.models[cid], self.test_loader,
                self.client_devices.get(cid, self.device)
            )
            results.append((acc, f1, loss))
        return results

    # ── Vòng lặp chính ───────────────────────────────────────────────
    def run(self) -> MetricsTracker:
        cfg        = self.cfg
        num_rounds = cfg.get("num_rounds", 50)
        eval_every = cfg.get("eval_every", 5)
        ckpt_path  = cfg.get("ckpt_path")

        # ── Resume from checkpoint ───────────────────────────────────
        start_round = 1
        if ckpt_path and os.path.exists(ckpt_path):
            if cfg.get("fresh"):
                os.remove(ckpt_path)
                print(f"  [checkpoint] Fresh start -- old checkpoint removed")
            else:
                print(f"  [checkpoint] Resuming from {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location="cpu",
                                  weights_only=False)
                start_round = ckpt["round"] + 1
                for cid in range(self.num_clients):
                    self.models[cid].load_state_dict(ckpt["models"][cid])
                self.tracker.history        = ckpt["tracker_history"]
                self.tracker.total_bytes     = ckpt["tracker_total_bytes"]
                self.tracker.total_time_s    = ckpt["tracker_total_time_s"]
                self.tracker.round_times     = ckpt["tracker_round_times"]
                self.tracker._conv_round     = ckpt["tracker_conv_round"]
                self.tracker.node_comm_kb    = ckpt.get("node_comm_kb", {})
                if "node_train_time_s" in ckpt:
                    self.tracker.node_train_time_s = ckpt["node_train_time_s"]
                print(f"  [checkpoint] Resumed at round {start_round}/{num_rounds}")

        if start_round > num_rounds:
            print(f"  [checkpoint] Already complete ({num_rounds} rounds) -- use --fresh to re-run")
            self.tracker.print_summary()
            return self.tracker

        for r in tqdm(range(start_round, num_rounds + 1), desc="Rounds"):
            t_round = time.time()
            self.run_round(r)

            if r % eval_every == 0 or r == num_rounds:
                per_client = self.eval_all()
                round_time = time.time() - t_round
                self.tracker.add_round_time(round_time)
                self.tracker.log_round(r, per_client, round_time_s=round_time)

                last = self.tracker.history[-1]
                tqdm.write(
                    f"  Round {r:4d} | "
                    f"F1={last['mean_f1']:.4f} +/- {last['std_f1']:.4f} | "
                    f"Acc={last['mean_acc']:.4f} | "
                    f"Comm={last['comm_cost_mb']:.1f} MB | "
                    f"Time={last.get('round_time_s',0):.1f}s"
                )
                # In per-node F1 de thay Hub vs Leaf
                node_f1s = {cid: last.get(f'f1_client_{cid}', 0)
                            for cid in range(self.num_clients)}
                hub_f1  = [node_f1s[c] for c,v in self.nodes_info.items() if v['type']=='Hub']
                leaf_f1 = [node_f1s[c] for c,v in self.nodes_info.items() if v['type']=='Leaf']
                import numpy as np
                if hub_f1:
                    tqdm.write(f"         Hub  F1={np.mean(hub_f1):.4f} | "
                               f"Leaf F1={np.mean(leaf_f1):.4f}")

                # ── Save checkpoint ───────────────────────────────────
                if ckpt_path:
                    torch.save({
                        "round": r,
                        "models": {cid: self.models[cid].state_dict()
                                   for cid in range(self.num_clients)},
                        "tracker_history":     self.tracker.history,
                        "tracker_total_bytes":  self.tracker.total_bytes,
                        "tracker_total_time_s": self.tracker.total_time_s,
                        "tracker_round_times":  self.tracker.round_times,
                        "tracker_conv_round":   self.tracker._conv_round,
                        "node_comm_kb":         self.tracker.node_comm_kb,
                        "node_train_time_s":    getattr(self.tracker, 'node_train_time_s', {}),
                    }, ckpt_path)

        self.tracker.print_summary()
        return self.tracker

"""
p2p_runner.py — Coordinator cho chế độ P2P.
Khởi động N process con, mỗi process là 1 node độc lập giao tiếp TCP.
KHÔNG phải FL server — chỉ điều phối khởi động và thu thập metrics.
"""
import os, sys, time, json, pickle, socket, struct, threading, subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from topology import create_topology, get_topology_info
from trainer  import get_model, get_model_size_kb, get_mixing_matrix


# ── Partition data vào file .pkl cho từng node ────────────────────────
def prepare_partitions(num_nodes, alpha, data_dir, out_dir, max_samples=None):
    import numpy as np
    from torchvision import datasets, transforms
    import pickle, os

    os.makedirs(out_dir, exist_ok=True)
    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=T)
    labels   = np.array(train_ds.targets)

    np.random.seed(42)
    num_classes  = len(np.unique(labels))
    node_indices = [[] for _ in range(num_nodes)]

    for c in range(num_classes):
        idx   = np.where(labels == c)[0]
        np.random.shuffle(idx)
        props = np.random.dirichlet([alpha] * num_nodes)
        props = np.maximum(props, 10 / len(idx))
        props = props / props.sum()
        splits = (props * len(idx)).astype(int)
        splits[-1] = len(idx) - splits[:-1].sum()
        cur = 0
        for k in range(num_nodes):
            node_indices[k].extend(idx[cur: cur + splits[k]].tolist())
            cur += splits[k]

    if max_samples:
        rng = np.random.RandomState(42)
        node_indices = [
            rng.choice(idx, min(len(idx), max_samples), replace=False).tolist()
            for idx in node_indices
        ]

    meta = {"num_nodes": num_nodes, "alpha": alpha,
            "sizes": [len(x) for x in node_indices]}
    with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    for i, idx in enumerate(node_indices):
        with open(os.path.join(out_dir, f"node_{i}.pkl"), "wb") as f:
            pickle.dump(idx, f)

    print(f"[P2P] Partitioned: {[len(x) for x in node_indices]}")
    return node_indices


# ── Coordinator metric server ─────────────────────────────────────────
_metrics  = []
_m_lock   = threading.Lock()

def _coord_server(port):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(50)
    srv.settimeout(1.0)
    while True:
        try:
            conn, _ = srv.accept()
            conn.settimeout(5.0)
            raw_len = b""
            while len(raw_len) < 4:
                raw_len += conn.recv(4 - len(raw_len))
            n = struct.unpack(">I", raw_len)[0]
            data = b""
            while len(data) < n:
                data += conn.recv(min(4096, n - len(data)))
            conn.close()
            with _m_lock:
                _metrics.append(pickle.loads(data))
        except socket.timeout:
            continue
        except Exception:
            continue


# ── Vẽ kết quả P2P ───────────────────────────────────────────────────
def plot_p2p(metrics, results_dir, label):
    if not metrics:
        return
    from collections import defaultdict
    by_round = defaultdict(list)
    for m in metrics:
        by_round[m["round"]].append(m)

    rounds    = sorted(by_round.keys())
    mean_acc  = [np.mean([m["accuracy"] for m in by_round[r]]) for r in rounds]
    std_acc   = [np.std ([m["accuracy"] for m in by_round[r]]) for r in rounds]
    mean_f1   = [np.mean([m["f1"]       for m in by_round[r]]) for r in rounds]
    comm_mb   = [max(m["bytes_sent"] for m in by_round[r]) / 1e6 for r in rounds]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(rounds, mean_acc, 'b-o', lw=2, label="Mean Acc")
    axes[0].fill_between(rounds,
                         [a-s for a,s in zip(mean_acc, std_acc)],
                         [a+s for a,s in zip(mean_acc, std_acc)],
                         alpha=0.2)
    axes[0].set_title(f"P2P Learning Curve\n{label}")
    axes[0].set_xlabel("Round"); axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1); axes[0].grid(True, alpha=0.3); axes[0].legend()

    last_r   = max(rounds)
    node_acc = {m["node_id"]: m["accuracy"] for m in by_round[last_r]}
    nids     = sorted(node_acc)
    axes[1].bar(nids, [node_acc[n] for n in nids],
                color=plt.cm.tab10.colors[:len(nids)], alpha=0.85)
    axes[1].axhline(np.mean(list(node_acc.values())),
                    color='red', ls='--', label='Mean')
    axes[1].set_title(f"Per-Node Acc (Round {last_r})")
    axes[1].set_xlabel("Node ID"); axes[1].set_ylabel("Accuracy")
    axes[1].legend(); axes[1].grid(True, axis='y', alpha=0.3)

    axes[2].plot(rounds, comm_mb, 'g-s', lw=2)
    axes[2].set_title("Comm Cost (per node)")
    axes[2].set_xlabel("Round"); axes[2].set_ylabel("MB sent (cumulative)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(results_dir, f"p2p_{label}.png")
    plt.savefig(path, dpi=130); plt.close()
    print(f"[P2P] Plot saved → {path}")


# ── Entry point ───────────────────────────────────────────────────────
def run_p2p(topology="ring", algorithm="gossip", alpha=0.5,
            num_nodes=5, num_rounds=10, local_epochs=2,
            lr=0.01, batch_size=32, eval_every=2,
            max_samples=800, base_port=9100, seed=42,
            results_dir="./results", data_dir="./data"):

    os.makedirs(results_dir, exist_ok=True)
    part_dir   = os.path.join(results_dir, "p2p_partitions")
    coord_port = base_port + num_nodes + 1
    label      = f"{topology}_{algorithm}_a{alpha}"

    print(f"\n{'='*55}")
    print(f" P2P DFL -- topology={topology} algo={algorithm} alpha={alpha}")
    print(f" nodes={num_nodes}  rounds={num_rounds}  port_base={base_port}")
    print(f"{'='*55}\n")

    # Step 1: Partition data
    prepare_partitions(num_nodes, alpha, data_dir, part_dir, max_samples)

    # Step 2: Topology + Mixing Matrix
    G  = create_topology(topology, num_nodes)
    W  = get_mixing_matrix(G)
    nb = [sorted(G.neighbors(i)) for i in range(num_nodes)]
    info = get_topology_info(G)
    m  = get_model("mnist")
    kb = get_model_size_kb(m)
    print(f"[P2P] {info['num_edges']} edges | model={kb:.0f}KB | "
          f"comm/round≈{info['num_edges']*kb*2:.0f}KB")

    # Step 3: Shared config JSON
    config = {
        "num_nodes"    : num_nodes,
        "base_port"    : base_port,
        "coord_port"   : coord_port,
        "dataset"      : "mnist",
        "data_dir"     : data_dir,
        "partition_dir": part_dir,
        "batch_size"   : batch_size,
        "num_rounds"   : num_rounds,
        "local_epochs" : local_epochs,
        "lr"           : lr,
        "algorithm"    : algorithm,
        "eval_every"   : eval_every,
        "seed"         : seed,
        "device"       : "cpu",
        "results_dir"  : results_dir,
        "neighbors"    : nb,
        "mixing_row"   : [
            [W[i][i]] + [W[i][j] for j in nb[i]]
            for i in range(num_nodes)
        ],
    }
    cfg_path = os.path.join(results_dir, f"p2p_config_{label}.json")
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)

    # Step 4: Khởi động coordinator metric server
    t = threading.Thread(target=_coord_server, args=(coord_port,), daemon=True)
    t.start()
    print(f"[P2P] Coord metric server → port {coord_port}")

    # Step 5: Khởi động N process con
    node_script = os.path.join(os.path.dirname(__file__),
                               "node", "node_process.py")
    procs = []
    for i in range(num_nodes):
        p = subprocess.Popen(
            [sys.executable, node_script, str(i), cfg_path],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        procs.append(p)
        print(f"[P2P] Node {i} launched (pid={p.pid})")
        time.sleep(0.4)

    # Step 6: Stream stdout từ tất cả node
    def stream(proc, nid):
        for line in iter(proc.stdout.readline, ""):
            print(f"  [N{nid}] {line.rstrip()}")
        proc.stdout.close()

    for i, p in enumerate(procs):
        threading.Thread(target=stream, args=(p, i), daemon=True).start()

    # Step 7: Chờ tất cả xong
    for i, p in enumerate(procs):
        p.wait()
        print(f"[P2P] Node {i} finished (exit={p.returncode})")

    time.sleep(2.0)

    # Step 8: Vẽ biểu đồ + summary
    with _m_lock:
        ms = list(_metrics)

    plot_p2p(ms, results_dir, label)

    if ms:
        from collections import defaultdict
        by_round = defaultdict(list)
        for m in ms:
            by_round[m["round"]].append(m)
        last = max(by_round.keys())
        accs = [m["accuracy"] for m in by_round[last]]
        comm = max(m["bytes_sent"] for m in by_round[last]) / 1e6

        print(f"\n{'='*55}")
        print(f" P2P SUMMARY — {label}")
        print(f"{'='*55}")
        print(f"  Final Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"  Comm cost/node : {comm:.2f} MB")
        print(f"  Nodes/Rounds   : {num_nodes} / {num_rounds}")
        print(f"  Topology       : {topology} ({info['num_edges']} edges)")
        print(f"  Algorithm      : {algorithm}")
        print(f"{'='*55}\n")

    return ms

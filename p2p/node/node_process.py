"""
DFL Node Process — chạy như 1 process độc lập.
Mỗi round:
  1. Local training (SGD)
  2. Gửi weight cho hàng xóm qua TCP
  3. Nhận weight từ hàng xóm qua TCP
  4. Aggregate (Gossip hoặc FedAvg)
  5. Evaluate + báo cáo về coordinator
"""
import sys, os, time, json, pickle, socket, struct, threading, logging
# Thêm thư mục gốc vào path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch
import numpy as np
from collections import OrderedDict

from trainer import get_model, local_train, gossip_aggregate, fedavg_aggregate, evaluate


# ── Logging ──────────────────────────────────────────────────────────
def setup_logging(node_id, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    fmt = f"[Node {node_id}] %(asctime)s %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=fmt, datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f"node_{node_id}.log"), mode="w"),
            logging.StreamHandler(sys.stdout),
        ]
    )
    return logging.getLogger(f"node_{node_id}")


# ── Load data cho 1 node từ partition file ───────────────────────────
def load_node_data(node_id, batch_size, partition_dir, data_dir):
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset

    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST(data_dir, train=True,  download=True,  transform=T)
    test_ds  = datasets.MNIST(data_dir, train=False, download=False, transform=T)

    with open(os.path.join(partition_dir, f"node_{node_id}.pkl"), "rb") as f:
        indices = pickle.load(f)

    train_loader = DataLoader(Subset(train_ds, indices),
                              batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False)
    return train_loader, test_loader, len(indices)


# ── Báo cáo metric về coordinator ────────────────────────────────────
def report_metric(coord_port, metric):
    try:
        data = pickle.dumps(metric)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5.0)
        s.connect(("127.0.0.1", coord_port))
        s.sendall(struct.pack(">I", len(data)))
        s.sendall(data)
        s.close()
    except Exception:
        pass


# ── Main node loop ────────────────────────────────────────────────────
def run_node(node_id: int, config: dict):
    log = setup_logging(node_id, config["results_dir"])
    log.info(f"PID={os.getpid()} starting")

    torch.manual_seed(config["seed"] + node_id)
    np.random.seed(config["seed"] + node_id)

    device       = config["device"]
    neighbors    = config["neighbors"][node_id]

    # Phan loai Hub/Leaf theo yeu cau co (transcript 27:18 + 46:03)
    degree    = len(neighbors)
    avg_degree = config.get("avg_degree", 2.0)
    node_type  = "Hub" if degree > avg_degree else "Leaf"
    log.info(f"Topology: degree={degree}, type={node_type}, neighbors={neighbors}")
    mixing_row   = config["mixing_row"][node_id]
    base_port    = config["base_port"]
    coord_port   = config["coord_port"]
    algorithm    = config["algorithm"]
    num_rounds   = config["num_rounds"]
    local_epochs = config["local_epochs"]
    lr           = config["lr"]
    eval_every   = config["eval_every"]

    # Load data
    train_loader, test_loader, num_samples = load_node_data(
        node_id, config["batch_size"],
        config["partition_dir"], config["data_dir"]
    )
    log.info(f"Loaded {num_samples} samples")

    # Init CNN
    model = get_model(config["dataset"]).to(device)

    # Server lắng nghe trong thread riêng
    from p2p.node.comms import NodeServer, NodeClient
    server   = NodeServer(node_id, base_port)
    received = {}
    recv_lock = threading.Lock()
    stop_evt  = threading.Event()

    def server_loop():
        while not stop_evt.is_set():
            try:
                sid, w, n, nb = server.receive_one()
                actual = sid // 1000
                rnd    = sid % 1000
                with recv_lock:
                    received.setdefault(rnd, []).append((actual, w, n, nb))
            except OSError:
                break
            except Exception:
                continue

    threading.Thread(target=server_loop, daemon=True).start()

    # Đợi tất cả node khởi động
    time.sleep(3.0)

    total_bytes_sent = 0

    for r in range(1, num_rounds + 1):
        log.info(f"=== Round {r}/{num_rounds} ===")
        t0 = time.time()

        # STEP 1: Local train
        weights, loss = local_train(
            model, train_loader, local_epochs, lr, device, algorithm="gossip"
        )
        model.load_state_dict(weights)
        log.info(f"  Train loss={loss:.4f}")

        # STEP 2: Gửi cho hàng xóm
        for nb_id in neighbors:
            encoded = node_id * 1000 + r
            try:
                sent = NodeClient.send(
                    target_port = base_port + nb_id,
                    sender_id   = encoded,
                    weights     = model.state_dict(),
                    num_samples = num_samples,
                )
                total_bytes_sent += sent
                log.info(f"  Sent {sent/1024:.1f}KB to Node {nb_id}")
            except Exception as e:
                log.error(f"  Send to {nb_id} failed: {e}")

        # STEP 3: Chờ nhận từ hàng xóm
        deadline = time.time() + 300.0
        while time.time() < deadline:
            with recv_lock:
                got = len(received.get(r, []))
            if got >= len(neighbors):
                break
            time.sleep(0.1)

        with recv_lock:
            round_recv = received.get(r, [])

        log.info(f"  Got {len(round_recv)}/{len(neighbors)} neighbors")

        # STEP 4: Aggregate
        if round_recv:
            nb_w   = [x[1] for x in round_recv]
            nb_n   = [x[2] for x in round_recv]
            own_w  = model.state_dict()

            if algorithm == "gossip":
                row = [mixing_row[0]] + [
                    mixing_row[neighbors.index(x[0]) + 1]
                    for x in round_recv if x[0] in neighbors
                ]
                new_w = gossip_aggregate(own_w, nb_w, row)
            else:  # fedavg
                new_w = fedavg_aggregate(
                    [own_w] + nb_w,
                    [num_samples] + nb_n
                )
            model.load_state_dict(new_w)

        # STEP 5: Evaluate
        if r % eval_every == 0 or r == num_rounds:
            acc, f1, loss_v = evaluate(model, test_loader, device)
            rt = time.time() - t0
            log.info(f"  EVAL: acc={acc:.4f} f1={f1:.4f} "
                     f"comm={total_bytes_sent/1024/1024:.2f}MB "
                     f"time={rt:.1f}s")
            report_metric(coord_port, {
                "round"      : r,
                "node_id"    : node_id,
                "accuracy"   : acc,
                "f1"         : f1,
                "loss"       : loss_v,
                "bytes_sent" : total_bytes_sent,
            })

    log.info(f"Done. Total sent={total_bytes_sent/1024/1024:.2f}MB")
    stop_evt.set()
    server.close()

    # Lưu metrics
    out = os.path.join(config["results_dir"], f"p2p_node_{node_id}.pkl")
    with open(out, "wb") as f:
        pickle.dump({"node_id": node_id, "total_bytes": total_bytes_sent}, f)


if __name__ == "__main__":
    node_id     = int(sys.argv[1])
    config_path = sys.argv[2]
    with open(config_path) as f:
        cfg = json.load(f)
    run_node(node_id, cfg)

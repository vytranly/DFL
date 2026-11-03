"""
Module 3 — trainer.py
Chứa:
  1. SimpleCNN     — mạng CNN dùng cho MNIST
  2. local_train   — huấn luyện cục bộ (SGD thông thường hoặc FedProx)
  3. gossip_aggregate — tổng hợp weight theo Mixing Matrix W (Gossip)
  4. fedavg_aggregate — tổng hợp weight theo số mẫu (FedAvg)
  5. evaluate         — đánh giá model trên test set
  6. get_mixing_matrix — tính Mixing Matrix W từ topology (Metropolis-Hastings)
"""

import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict
from sklearn.metrics import f1_score


# ─────────────────────────────────────────────
# 1. MODEL — SimpleCNN
# ─────────────────────────────────────────────

class SimpleCNN(nn.Module):
    """
    CNN đơn giản cho MNIST / FashionMNIST.
    Input:  (B, 1, 28, 28)
    Output: (B, 10) — logits 10 class
    Params: ~422K
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # (B,32,28,28)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # (B,64,14,14)
        self.pool  = nn.MaxPool2d(2)                               # giảm 1/2
        self.drop  = nn.Dropout(0.25)                              # chống overfit
        self.fc1   = nn.Linear(64 * 7 * 7, 128)                   # 3136 → 128
        self.fc2   = nn.Linear(128, num_classes)                   # 128 → 10

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # (B,32,28,28) → (B,32,14,14)
        x = self.pool(F.relu(self.conv2(x)))   # (B,64,14,14) → (B,64,7,7)
        x = x.view(x.size(0), -1)             # flatten → (B,3136)
        x = F.relu(self.fc1(self.drop(x)))     # (B,3136) → (B,128)
        return self.fc2(x)                     # (B,128)  → (B,10)


def get_model(dataset_name='mnist', num_classes=10):
    """Tạo model phù hợp với dataset."""
    if dataset_name.lower() in ('mnist', 'fashionmnist'):
        return SimpleCNN(num_classes=num_classes)
    raise ValueError(f"Chua ho tro dataset: {dataset_name}")


def get_model_size_kb(model):
    """Tính kích thước model tính bằng KB."""
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_bytes / 1024


# ─────────────────────────────────────────────
# 2. LOCAL TRAINING
# ─────────────────────────────────────────────

def local_train(model, dataloader, epochs, lr, device):
    """
    Huan luyen cuc bo SGD thuan tuy.
    KHONG co proximal term o day -- FedProx xu ly o fedprox_aggregate().
    (Theo feedback co 31:24: FedProx la co che tong hop, khong phai local train)

    Returns:
        state_dict   : weight sau train
        avg_loss     : cross-entropy loss trung binh
        train_time_s : thoi gian train (giay) -- de tach rieng training cost
    """
    t0 = time.time()
    model.train()
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    steps = 0

    for _ in range(epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1

    avg_loss = total_loss / max(steps, 1)
    return copy.deepcopy(model.state_dict()), avg_loss, time.time() - t0


# ─────────────────────────────────────────────
# 3. MIXING MATRIX W
# ─────────────────────────────────────────────

def get_mixing_matrix(graph):
    """
    Tính Mixing Matrix W theo Metropolis-Hastings.
    W[i][j] = 1 / (1 + max(deg_i, deg_j))  nếu (i,j) là cạnh
    W[i][i] = 1 - tổng hàng i
    Kết quả là doubly-stochastic (tổng hàng = tổng cột = 1).
    """
    n = graph.number_of_nodes()
    W = np.zeros((n, n))
    degrees = dict(graph.degree())

    for i, j in graph.edges():
        w_ij = 1.0 / (1 + max(degrees[i], degrees[j]))
        W[i][j] = w_ij
        W[j][i] = w_ij

    for i in range(n):
        W[i][i] = 1.0 - W[i].sum()

    return W


# ─────────────────────────────────────────────
# 4. AGGREGATION
# ─────────────────────────────────────────────

def gossip_aggregate(own_weights, neighbor_weights, mixing_row):
    """
    Gossip Aggregation — trọng số theo Mixing Matrix W.

    Args:
        own_weights      : state_dict của node i
        neighbor_weights : list state_dict của các hàng xóm
        mixing_row       : [W[i][i], W[i][j1], W[i][j2], ...]
                           (self-weight đầu tiên, rồi neighbor weights)

    Returns:
        OrderedDict — state_dict mới sau aggregate
    """
    all_weights = [own_weights] + neighbor_weights
    result = OrderedDict()

    for key in own_weights.keys():
        # Nhân từng weight với hệ số tương ứng rồi cộng lại
        stacked = torch.stack([
            w[key].float() * coeff
            for w, coeff in zip(all_weights, mixing_row)
        ])
        result[key] = stacked.sum(dim=0)

    return result


def fedavg_aggregate(weights_list, sizes_list):
    """
    FedAvg Aggregation — trọng số theo số mẫu.

    Args:
        weights_list : list state_dict (self + neighbors)
        sizes_list   : list số mẫu tương ứng

    Returns:
        OrderedDict — state_dict trung bình có trọng số
    """
    total = sum(sizes_list)
    result = OrderedDict()

    for key in weights_list[0].keys():
        stacked = torch.stack([
            w[key].float() * (n / total)
            for w, n in zip(weights_list, sizes_list)
        ])
        result[key] = stacked.sum(dim=0)

    return result


def fedprox_aggregate(own_weights, neighbor_weights,
                      sizes_list, mu=0.1):
    """
    FedProx Aggregation -- buoc TONG HOP, xu ly client drift Non-IID.
    (Theo feedback co 37:28: FedProx la co che tong hop, khong phai local train)

    Buoc 1: Tinh w_avg theo FedAvg (trong so theo so mau)
    Buoc 2: Proximal correction:
            w_result = w_avg - mu*(w_own - w_avg)
            => keo w_own ve phia w_avg, han che drift
    mu=0 => FedProx = FedAvg.
    """
    all_w = [own_weights] + neighbor_weights
    total = sum(sizes_list)
    w_avg = OrderedDict()
    for key in own_weights.keys():
        w_avg[key] = torch.stack([
            w[key].float() * (n / total)
            for w, n in zip(all_w, sizes_list)
        ]).sum(dim=0)
    if mu <= 0:
        return w_avg
    result = OrderedDict()
    for key in own_weights.keys():
        w_o = own_weights[key].float()
        w_a = w_avg[key].float()
        result[key] = w_a - mu * (w_o - w_a)
    return result


# ─────────────────────────────────────────────
# 5. EVALUATE
# ─────────────────────────────────────────────

def evaluate(model, test_loader, device):
    """
    Đánh giá model trên test set.

    Returns:
        accuracy  : float
        f1_macro  : float (F1 trung bình macro trên 10 class)
        avg_loss  : float
    """
    model.eval()
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    all_preds  = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits, y)

            total_loss  += loss.item()
            preds        = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())

    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    avg_loss = total_loss / max(len(test_loader), 1)

    return accuracy, f1_macro, avg_loss

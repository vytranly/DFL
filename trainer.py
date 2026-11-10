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
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2)
        self.drop  = nn.Dropout(0.25)
        self.fc1   = nn.Linear(64 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(self.drop(x)))
        return self.fc2(x)


class CifarCNN(nn.Module):
    """
    CNN cho CIFAR-10.
    Input:  (B, 3, 32, 32)  — ảnh màu RGB
    Output: (B, 10)         — logits 10 class
    Params: ~1.2M
    3 conv blocks + BN + Dropout để handle CIFAR phức tạp hơn MNIST.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        # Block 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2)   # 32→16
        # Block 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2)   # 16→8
        # Block 3
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5   = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2)   # 8→4
        # FC
        self.drop  = nn.Dropout(0.4)
        self.fc1   = nn.Linear(256 * 4 * 4, 512)
        self.fc2   = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(F.relu(self.bn4(self.conv4(x))))
        x = self.pool3(F.relu(self.bn5(self.conv5(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(self.drop(x)))
        return self.fc2(x)


def get_model(dataset_name='mnist', num_classes=10):
    """Tạo model phù hợp với dataset."""
    if dataset_name.lower() in ('mnist', 'fashionmnist'):
        return SimpleCNN(num_classes=num_classes)
    if dataset_name.lower() == 'cifar10':
        return CifarCNN(num_classes=num_classes)
    raise ValueError(f"Chua ho tro dataset: {dataset_name}")


def get_model_size_kb(model):
    """Tính kích thước model tính bằng KB."""
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_bytes / 1024


# ─────────────────────────────────────────────
# 2. LOCAL TRAINING
# ─────────────────────────────────────────────

def local_train(model, dataloader, epochs, lr, device,
                w_anchor=None, mu=0.0):
    """
    Huan luyen cuc bo SGD.
    Neu w_anchor != None va mu > 0: them proximal term vao loss (FedProx goc).
      loss_total = loss_CE + (mu/2) * ||w - w_anchor||^2
      => keo tung buoc gradient ve phia w_anchor (weight truoc round nay),
         ngan drift trong qua trinh train chu khong chi sua sau aggregate.

    Args:
        w_anchor : state_dict lam neo (weight truoc local train cua round nay)
                   None => SGD thuan (Gossip / FedAvg)
        mu       : he so proximal (0 = tat, 0.1 = nhe, 1.0 = manh)

    Returns:
        state_dict   : weight sau train
        avg_loss     : cross-entropy loss trung binh (khong tinh prox term)
        train_time_s : thoi gian train (giay)
    """
    t0 = time.time()
    model.train()
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Chuan bi anchor tensor tren dung device (chi tinh 1 lan)
    anchor_tensors = None
    if w_anchor is not None and mu > 0:
        anchor_tensors = {
            k: v.clone().detach().to(device).float()
            for k, v in w_anchor.items()
        }

    total_loss = 0.0
    steps = 0

    for _ in range(epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            loss = criterion(model(x), y)

            # Proximal term: (mu/2) * sum ||w_l - w_anchor_l||^2
            if anchor_tensors is not None:
                prox = sum(
                    torch.norm(p.float() - anchor_tensors[n]) ** 2
                    for n, p in model.named_parameters()
                    if n in anchor_tensors
                )
                loss = loss + (mu / 2.0) * prox

            loss.backward()
            optimizer.step()
            total_loss += loss.item()  # log toan bo loss (CE + prox)
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
# 4. AGGREGATION
# ─────────────────────────────────────────────

def gossip_aggregate(own_weights, neighbor_weights, mixing_row):
    """
    Gossip Aggregation — trọng số theo Mixing Matrix W.

    Args:
        own_weights      : state_dict của node i
        neighbor_weights : list state_dict của các hàng xóm
        mixing_row       : [W[i][i], W[i][j1], W[i][j2], ...]

    Returns:
        OrderedDict — state_dict mới sau aggregate
    """
    all_weights = [own_weights] + neighbor_weights
    result = OrderedDict()
    for key in own_weights.keys():
        stacked = torch.stack([
            w[key].float() * coeff
            for w, coeff in zip(all_weights, mixing_row)
        ])
        result[key] = stacked.sum(dim=0)
    return result


def fedprox_gossip_aggregate(own_weights, neighbor_weights,
                             mixing_row, mu=0.1, w_anchor=None):
    """
    FedProx-Gossip Aggregate — công thức closed-form dùng Gossip base.

    Bài toán tối ưu:
      min_w { Σ W_ij ‖w − w̃_j‖² + μ‖w − w_anchor‖² }

    Nghiệm dạng đóng:
      w_nb  = Σ W_ij × w̃_j              ← Gossip weighted mean
      w_new = (1/(1+μ)) × w_nb
            + (μ/(1+μ)) × w_anchor

    mixing_row: [W[i][i], W[i][j1], ...] — hàng i của Mixing Matrix
    w_anchor  : weight TRƯỚC local train (neo)
    """
    # Bước 1: Gossip weighted mean
    all_weights = [own_weights] + neighbor_weights
    w_nb = OrderedDict()
    for key in own_weights.keys():
        stacked = torch.stack([
            w[key].float() * coeff
            for w, coeff in zip(all_weights, mixing_row)
        ])
        w_nb[key] = stacked.sum(dim=0)

    # Bước 2: nghiệm dạng đóng
    if w_anchor is None or mu <= 0:
        return w_nb

    coeff_nb  = 1.0 / (1.0 + mu)
    coeff_anc = mu  / (1.0 + mu)

    result = OrderedDict()
    for key in w_nb.keys():
        w_n = w_nb[key].float()
        w_a = w_anchor[key].float().to(w_n.device)
        result[key] = coeff_nb * w_n + coeff_anc * w_a

    return result


def fedprox_aggregate(own_weights, neighbor_weights,
                             sizes_list, mu=0.1, w_anchor=None):
    """
    FedProx-FedAvg Aggregate — công thức mới cho DFL (exp9e).

    Bài toán tối ưu:
      min_w { Σ (n_j/N) ‖w − w̃_j‖² + μ‖w − w_anchor‖² }

    Nghiệm dạng đóng:
      w_nb  = Σ (n_j / N) × w̃_j          ← FedAvg weighted mean
      w_new = (1/(1+μ)) × w_nb
            + (μ/(1+μ)) × w_anchor

    Ưu điểm so với exp9d:
      - Hệ số 1/(1+μ) và μ/(1+μ) luôn dương dù μ bất kỳ
      - Có nghiệm lý thuyết chặt chẽ, không phải heuristic
      - Dùng n_j/N (FedAvg) thay vì W_ij (Gossip) → F1 cao hơn trên scale_free

    sizes_list: [own_size, nb1_size, ...] — bao gồm own
    w_anchor  : weight TRƯỚC local train của round này
    """
    all_w = [own_weights] + neighbor_weights
    total = sum(sizes_list)

    # Bước 1: FedAvg weighted mean
    w_nb = OrderedDict()
    for key in own_weights.keys():
        w_nb[key] = torch.stack([
            w[key].float() * (n / total)
            for w, n in zip(all_w, sizes_list)
        ]).sum(dim=0)

    # Bước 2: nghiệm dạng đóng
    if w_anchor is None or mu <= 0:
        return w_nb

    coeff_nb  = 1.0 / (1.0 + mu)
    coeff_anc = mu  / (1.0 + mu)

    result = OrderedDict()
    for key in w_nb.keys():
        w_n = w_nb[key].float()
        w_a = w_anchor[key].float().to(w_n.device)
        result[key] = coeff_nb * w_n + coeff_anc * w_a

    return result
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
                      sizes_list, mu=0.1, w_anchor=None):
    """
    FedProx Aggregation cho DFL — đúng công thức proximal term.

    Bước 1: Tính w_avg = FedAvg(self + neighbors) theo số mẫu.

    Bước 2 (nếu có w_anchor và mu > 0):
      Áp dụng 1 bước gradient descent trên proximal term:
        min_w { (μ/2) ‖w − w_anchor‖² }
        ∇_w = μ × (w − w_anchor)
        w_new = w_avg − μ × (w_avg − w_anchor)
              = (1 − μ) × w_avg + μ × w_anchor

      w_anchor = weight TRƯỚC local train của round này
                 (đóng vai trò w_global trong FedProx gốc, adapt cho DFL)

    Nếu w_anchor=None: trả về w_avg thuần (= FedAvg).

    Args:
        own_weights      : state_dict sau local train của node i
        neighbor_weights : list state_dict của các neighbor
        sizes_list       : [own_size, nb1_size, ...] — số mẫu
        mu               : hệ số proximal (μ)
        w_anchor         : state_dict trước local train (neo)
    """
    all_w = [own_weights] + neighbor_weights
    total = sum(sizes_list)

    # Bước 1: FedAvg weighted mean
    w_avg = OrderedDict()
    for key in own_weights.keys():
        w_avg[key] = torch.stack([
            w[key].float() * (n / total)
            for w, n in zip(all_w, sizes_list)
        ]).sum(dim=0)

    # Bước 2: proximal correction về w_anchor
    if w_anchor is None or mu <= 0:
        return w_avg

    result = OrderedDict()
    for key in w_avg.keys():
        w_a   = w_avg[key].float()
        w_anc = w_anchor[key].float().to(w_a.device)
        # w_new = w_avg - mu * (w_avg - w_anchor)
        #       = (1-mu)*w_avg + mu*w_anchor
        result[key] = w_a - mu * (w_a - w_anc)

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

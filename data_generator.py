import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

def load_raw_dataset(dataset_name='mnist', data_dir='./data'):
    os.makedirs(data_dir, exist_ok=True)
    
    if dataset_name.lower()== 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=True, download=True, transform=transform
        )
        global_test_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=False, download=True, transform=transform
        )
        
    elif dataset_name.lower() == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), 
                (0.2470, 0.2435, 0.2616)
            )
        ])
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform
        )
        global_test_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform
        )
    else:
        raise ValueError(f"Error dataset_name")
        
    return train_dataset, global_test_dataset


def dirichlet(dataset, num_clients, alpha=100, min_require_size=1):
    if torch.is_tensor(dataset.targets):
        targets = dataset.targets.cpu().numpy()
    else:
        targets = np.array(dataset.targets)
    num_classes = len(np.unique(targets))
    
    min_size = 0
    while min_size < min_require_size:
        client_dict = {i: [] for i in range(num_clients)}
        for k in range(num_classes):
            idx_k = np.where(targets == k)[0]
            np.random.shuffle(idx_k)

            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            client_splits = np.split(idx_k, proportions)
            
            for i in range(num_clients):
                client_dict[i].extend(client_splits[i].tolist())
                
        min_size = min(len(client_dict[i]) for i in range(num_clients))
            
    for i in range(num_clients):
        np.random.shuffle(client_dict[i])
        
    return client_dict


def dirichlet_imbalanced(dataset, num_clients, label_alpha=0.1, size_alpha=0.1,
                         min_require_size=20, max_client_size=None):
    """
    Phân vùng 2 TRỤC tách biệt (theo yêu cầu mới của exp5):
      - Label non-IID (label_alpha): dùng dirichlet() gốc — mỗi client tập trung
        vài lớp (label skew), full assignment, không client rỗng.
      - Size imbalance (size_alpha): Dirichlet over TỔNG số mẫu mỗi client, rồi
        subsample mỗi client xuống kích thước mục tiêu -> một số client giữ nhiều
        mẫu hơn hẳn các client khác. Trục size độc lập với trục label.

    max_client_size: cap kích thước tối đa mỗi client — giữ imbalance (client lớn
      vẫn >> client nhỏ) nhưng giới hạn client lớn để exp5 chạy nhanh.

    (Trục size chỉ làm client nhỏ đi để giữ nguyên label skew của trục label.)
    """
    # Trục 1: label non-IID (full assignment, label-skewed, không client rỗng)
    client_dict = dirichlet(dataset, num_clients, label_alpha,
                           min_require_size=min_require_size)
    N = sum(len(v) for v in client_dict.values())

    # Trục 2: size imbalance — mục tiêu kích thước mỗi client (Dirichlet over sizes)
    size_props = np.random.dirichlet(np.repeat(size_alpha, num_clients))
    target = np.maximum((size_props * N).astype(int), min_require_size)

    for i in range(num_clients):
        if len(client_dict[i]) > target[i]:
            sel = np.random.choice(client_dict[i], target[i], replace=False)
            client_dict[i] = sel.tolist()

    # Cap kích thước tối đa (giữ imbalance, chạy nhanh cho exp5)
    if max_client_size is not None:
        for i in range(num_clients):
            if len(client_dict[i]) > max_client_size:
                sel = np.random.choice(client_dict[i], max_client_size, replace=False)
                client_dict[i] = sel.tolist()
    return client_dict


def build_dataloaders(dataset_name='mnist',
                      num_clients=10, alpha=0.5, num_classes_per_client=2,
                      batch_size=32, data_dir='./data', size_alpha=None,
                      max_client_size=None):
    
    np.random.seed(42)
    torch.manual_seed(42)
    train_dataset, global_test_dataset = load_raw_dataset(
        dataset_name=dataset_name, 
        data_dir=data_dir, 
    )
    
    # size_alpha=None: giữ hành vi cũ (Dirichlet label non-IID, kích thước client
    #   chênh lệch như hệ quả phụ). size_alpha!=None: bật trục SIZE IMBALANCE tách
    #   biệt (Dirichlet over tổng số mẫu) + label non-IID (alpha) — cho exp5.
    if size_alpha is None:
        client_indices = dirichlet(train_dataset, num_clients, alpha)
    else:
        client_indices = dirichlet_imbalanced(train_dataset, num_clients,
                                              label_alpha=alpha,
                                              size_alpha=size_alpha,
                                              max_client_size=max_client_size)
    client_loaders = {}
    for cid, indices in client_indices.items():
        subset = Subset(train_dataset, indices)
        client_loaders[cid] = DataLoader(subset, batch_size=batch_size, shuffle=True)

    global_test_loader = DataLoader(global_test_dataset, batch_size=batch_size, shuffle=False)
    return client_loaders, global_test_loader, client_indices, train_dataset

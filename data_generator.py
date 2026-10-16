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


def build_dataloaders(dataset_name='mnist',
                      num_clients=10, alpha=0.5, num_classes_per_client=2,
                       batch_size=32, data_dir='./data'):
    
    np.random.seed(42)
    torch.manual_seed(42)
    train_dataset, global_test_dataset = load_raw_dataset(
        dataset_name=dataset_name, 
        data_dir=data_dir, 
    )
    

    client_indices = dirichlet(train_dataset, num_clients, alpha)        
    client_loaders = {}
    for cid, indices in client_indices.items():
        subset = Subset(train_dataset, indices)
        client_loaders[cid] = DataLoader(subset, batch_size=batch_size, shuffle=True)

    global_test_loader = DataLoader(global_test_dataset, batch_size=batch_size, shuffle=False)
    return client_loaders, global_test_loader, client_indices, train_dataset

# ===================================================================
# ĐOẠN CODE TEST ĐỘC LẬP (Chỉ chạy khi gọi trực tiếp file này)
# ===================================================================
if __name__ == "__main__":
    print("Đang tải dữ liệu MNIST và phân mảnh theo Dirichlet (alpha=0.1)...")
    
    # Gọi hàm chia dữ liệu cho 5 Client để dễ quan sát
    client_loaders, global_test_loader, client_indices, train_dataset = build_dataloaders(
        dataset_name='mnist',
        num_clients=5,   # Test với 5 máy
        alpha=0.1,       # Mức Non-IID khốc liệt
        batch_size=32
    )

    print(f"\n[+] Số lượng ảnh trong Global Test Set: {len(global_test_loader.dataset)} ảnh")

    print("\n[+] SỐ LƯỢNG ẢNH ĐƯỢC CẤP PHÁT CHO TỪNG CLIENT:")
    for cid, loader in client_loaders.items():
        print(f"    - Client {cid}: {len(loader.dataset)} ảnh (chia thành {len(loader)} mini-batches)")

    print("\n[+] KIỂM TRA ĐỘ LỆCH NHÃN (NON-IID) CỦA CLIENT 0 VÀ CLIENT 1:")
    # Hàm phụ trợ để đếm xem trong túi của Client có bao nhiêu ảnh của từng chữ số (0-9)
    def count_labels(dataset, indices):
        targets = np.array(dataset.targets)
        client_labels = targets[indices]
        unique_labels, counts = np.unique(client_labels, return_counts=True)
        return dict(zip(unique_labels, counts))

    # In kết quả của Client 0 và 1
    print(f"    - Nhãn của Client 0: {count_labels(train_dataset, client_indices[0])}")
    print(f"    - Nhãn của Client 1: {count_labels(train_dataset, client_indices[1])}")
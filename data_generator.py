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


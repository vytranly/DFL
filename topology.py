import networkx as nx 
import math

def create_topology(name, num_clients, random_p=0.5, lattice_k=4, seed=42):
    name = name.lower()
    if name == 'ring':
        G = nx.cycle_graph(num_clients)
    elif name == 'fully_connected':
        G = nx.complete_graph(num_clients)
    elif name == 'lattice':
        # Yêu cầu k phải là số chẵn để đảm bảo tính đối xứng của mạng lưới
        if lattice_k % 2 != 0:
            raise ValueError(f"[Topology] Warning: lattice_k must be even.")
        # p=0 đảm bảo không có kết nối ngẫu nhiên (rewiring), giữ mạng ở dạng lưới chuẩn
        G = nx.watts_strogatz_graph(num_clients, k=lattice_k, p=0, seed=seed)
    elif name == 'scale_free':
        # Thuật toán Barabási-Albert tạo mạng phi tỷ lệ (có Hub và Leaf rõ rệt)
        # m là số lượng cạnh mà một node mới sẽ kết nối với các node đã có trong mạng
        m = max(1, int(num_clients * 0.15)) 
        G = nx.barabasi_albert_graph(num_clients, m, seed=seed)
        
    elif name == 'random':
        safe_p = (math.log(num_clients) / num_clients) + 0.15
        actual_p = max(random_p, safe_p)    
        
        if actual_p > random_p:
            print(f"p={random_p} is too low (high disconnect risk).")
            print(f"-> Auto-adjusted to safe level: {actual_p:.3f}")
        
        G = nx.erdos_renyi_graph(num_clients, actual_p, seed=seed)
        
        if not nx.is_connected(G):
            print(f"Graph is disconnect")            
    else:
        raise ValueError(f"Error Topology name!")
    return G

def get_neighbors(graph, client_id):
    return list(graph.neighbors(client_id))

def get_topology_info(graph):
    degrees = [d for _, d in graph.degree()]
    info = {
        'num_nodes': graph.number_of_nodes(),
        'num_edges': graph.number_of_edges(),
        'avg_degree': sum(degrees) / len(degrees) if degrees else 0,
        'max_degree': max(degrees) if degrees else 0, # Số kết nối của Hub node lớn nhất
        'min_degree': min(degrees) if degrees else 0, # Số kết nối của Leaf node nhỏ nhất
        'is_connected': nx.is_connected(graph)
    }
    return info


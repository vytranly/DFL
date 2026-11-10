import networkx as nx 
import math

def create_topology(name, num_clients, random_p=0.5, seed=42, lattice_k=4):
    name = name.lower()
    if name == 'ring':
        G = nx.cycle_graph(num_clients)
    elif name == 'fully_connected':
        G = nx.complete_graph(num_clients)
    elif name == 'random':
        safe_p = (math.log(num_clients) / num_clients) + 0.15
        actual_p = max(random_p, safe_p)    
        
        if actual_p > random_p:
            print(f"p={random_p} is too low (high disconnect risk).")
            print(f"-> Auto-adjusted to safe level: {actual_p:.3f}")
        
        for attempt in range(10):
            G = nx.erdos_renyi_graph(num_clients, actual_p, seed=seed + attempt)
            if nx.is_connected(G):
                break
        else:
            print(f"[topology] Warning: could not generate connected random graph after 10 attempts, falling back to ring.")
            G = nx.cycle_graph(num_clients)
    elif name == 'scale_free':
        # Barabasi-Albert preferential attachment: m = initial edges per new node
        # m=2 gives moderate Hub/Leaf separation for small num_clients
        m = max(1, min(2, num_clients - 1))
        G = nx.barabasi_albert_graph(num_clients, m, seed=seed)
        # Ensure connectivity (BA graph is always connected for n >= m+1)
        if not nx.is_connected(G):
            G = nx.cycle_graph(num_clients)
    elif name == 'lattice':
        # 2D torus-like lattice with wrap-around, then augmented with extra
        # ring edges to ensure connectivity; lattice_k controls ring degree.
        # Build as a k-regular ring lattice (Watts-Strogatz with p=0).
        k_even = lattice_k if lattice_k % 2 == 0 else lattice_k - 1
        k_even = max(2, min(k_even, num_clients - 1))
        G = nx.watts_strogatz_graph(num_clients, k_even, 0, seed=seed)
        if not nx.is_connected(G):
            G = nx.cycle_graph(num_clients)
    else:
        raise ValueError(f"Error Topology name: {name}. "
                         f"Supported: ring, fully_connected, random, "
                         f"scale_free, lattice")
    return G

def get_neighbors(graph, client_id):
    return list(graph.neighbors(client_id))

def get_topology_info(graph):
    degrees = [d for _, d in graph.degree()]
    info = {
        'num_nodes'    : graph.number_of_nodes(),
        'num_edges'    : graph.number_of_edges(),
        'avg_degree'   : sum(degrees) / len(degrees) if degrees else 0,
        'max_degree'   : max(degrees) if degrees else 0,
        'min_degree'   : min(degrees) if degrees else 0,
        'is_connected' : nx.is_connected(graph)
    }
    return info


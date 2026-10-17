import networkx as nx 
import math

def create_topology(name, num_clients, random_p=0.5, seed=42):
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
        'avg_degree': sum(degrees) / len(degrees),
        'is_connected': nx.is_connected(graph)
    }
    return info


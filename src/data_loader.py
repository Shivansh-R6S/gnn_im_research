

import networkx as nx
import numpy as np
from typing import Tuple, Dict
import pickle
import os


def load_graph(edge_list_path: str) -> nx.Graph:
    """
    Load a cybercrime co-occurrence network from edge list file.
    
    File format:
        Crime1 Crime2 weight
        Crime1 Crime3 weight
        ...
    
    Args:
        edge_list_path: Path to edge list file
        
    Returns:
        NetworkX undirected weighted graph
    """
    G = nx.Graph()
    
    with open(edge_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) != 3:
                print(f"Warning: Skipping malformed line: {line}")
                continue
            
            node1, node2, weight = parts[0], parts[1], float(parts[2])
            
            # Add nodes with attributes
            if node1 not in G:
                G.add_node(node1, crime_type=node1)
            if node2 not in G:
                G.add_node(node2, crime_type=node2)
            
            # Add weighted edge
            G.add_edge(node1, node2, weight=weight)
    
    return G


def save_graph(G: nx.Graph, output_path: str) -> None:
    """Save graph to pickle file for faster loading."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(G, f)


def load_saved_graph(pickle_path: str) -> nx.Graph:
    """Load pre-processed graph from pickle."""
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)


def get_graph_statistics(G: nx.Graph) -> Dict:
    """Compute and return graph statistics."""
    edges = G.edges(data=True)
    weights = [d['weight'] for _, _, d in edges]
    
    stats = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'density': nx.density(G),
        'avg_degree': 2 * G.number_of_edges() / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
        'avg_weight': np.mean(weights) if weights else 0,
        'min_weight': np.min(weights) if weights else 0,
        'max_weight': np.max(weights) if weights else 0,
        'std_weight': np.std(weights) if weights else 0,
        'is_connected': nx.is_connected(G),
        'num_components': nx.number_connected_components(G),
    }
    
    # Compute weighted degree statistics
    weighted_degrees = [sum(d['weight'] for _, _, d in G.edges(node, data=True)) for node in G.nodes()]
    stats['avg_weighted_degree'] = np.mean(weighted_degrees) if weighted_degrees else 0
    stats['min_weighted_degree'] = np.min(weighted_degrees) if weighted_degrees else 0
    stats['max_weighted_degree'] = np.max(weighted_degrees) if weighted_degrees else 0
    
    return stats


def print_graph_info(G: nx.Graph) -> None:
    """Print human-readable graph information."""
    stats = get_graph_statistics(G)
    
    print("\n" + "="*60)
    print("GRAPH STATISTICS")
    print("="*60)
    print(f"Nodes:                 {stats['num_nodes']}")
    print(f"Edges:                 {stats['num_edges']}")
    print(f"Density:               {stats['density']:.4f}")
    print(f"Connected:             {stats['is_connected']}")
    print(f"Num Components:        {stats['num_components']}")
    print(f"\nDegree Statistics:")
    print(f"  Average Degree:      {stats['avg_degree']:.2f}")
    print(f"  Weighted Avg:        {stats['avg_weighted_degree']:.4f}")
    print(f"  Min Weighted:        {stats['min_weighted_degree']:.4f}")
    print(f"  Max Weighted:        {stats['max_weighted_degree']:.4f}")
    print(f"\nEdge Weight Statistics:")
    print(f"  Mean:                {stats['avg_weight']:.4f}")
    print(f"  Std Dev:             {stats['std_weight']:.4f}")
    print(f"  Min:                 {stats['min_weight']:.4f}")
    print(f"  Max:                 {stats['max_weight']:.4f}")
    print("="*60 + "\n")
    
    # Print node list
    print(f"NODES ({len(G.nodes())} total):")
    for node in sorted(G.nodes()):
        weighted_deg = sum(d['weight'] for _, _, d in G.edges(node, data=True))
        print(f"  {node:30s} - degree: {G.degree(node):2d}, weighted degree: {weighted_deg:.4f}")
    print()


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        edge_list_path = sys.argv[1]
    else:
        edge_list_path = "data/raw/cybercrime_edge_list.txt"
    
    print(f"Loading graph from: {edge_list_path}")
    G = load_graph(edge_list_path)
    print_graph_info(G)

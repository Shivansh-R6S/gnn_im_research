"""
Baseline Influence Maximization Algorithms

Implementations of standard IM methods:
1. Greedy (gold standard, slow but proven)
2. Degree Heuristic (fast, simple)
3. PageRank (centrality-based)
4. Random (lower bound)
"""

import networkx as nx
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
import time

from .diffusion import simulate_ic_diffusion, get_marginal_gain


class GreedyIM:
    """
    Standard Greedy Influence Maximization.
    
    Algorithm:
    1. Start with empty seed set S
    2. For i = 1 to k:
       - For each node v not in S:
         - Compute marginal gain: gain(v) = σ(S ∪ {v}) - σ(S)
       - Select node with max gain
       - Add to S
    
    Guarantees: (1 - 1/e) ≈ 63.2% approximation
    """
    
    def __init__(self, G: nx.Graph, mc_iterations: int = 1000):
        """
        Args:
            G: NetworkX graph
            mc_iterations: MC runs for influence estimation
        """
        self.G = G
        self.mc_iterations = mc_iterations
        self.nodes = list(G.nodes())
    
    def select_seeds(self, k: int, verbose: bool = True) -> Tuple[List[str], float]:
        """
        Select k seeds greedily.
        
        Args:
            k: Budget (number of seeds)
            verbose: Print progress
            
        Returns:
            (seed_set, total_computation_time)
        """
        
        start_time = time.time()
        seed_set = []
        
        iterator = tqdm(range(k), desc="Greedy Selection", disable=not verbose)
        
        for iteration in iterator:
            best_node = None
            best_gain = -1
            
            # Try each node not in seed set
            for v in self.nodes:
                if v in seed_set:
                    continue
                
                # Compute marginal gain
                gain = get_marginal_gain(
                    self.G, 
                    set(seed_set), 
                    v,
                    mc_iterations=self.mc_iterations
                )
                
                if gain > best_gain:
                    best_gain = gain
                    best_node = v
            
            # Add best node to seed set
            seed_set.append(best_node)
            
            if verbose:
                iterator.set_postfix({'best_gain': f'{best_gain:.4f}'})
        
        elapsed_time = time.time() - start_time
        
        return seed_set, elapsed_time


class DegreeHeuristic:
    """
    Simple degree-based heuristic: select k nodes with highest weighted degree.
    
    Fast: O(n log k)
    Quality: ~70-85% of greedy
    """
    
    def __init__(self, G: nx.Graph):
        self.G = G
    
    def select_seeds(self, k: int, verbose: bool = True) -> Tuple[List[str], float]:
        """
        Select k seeds by weighted degree.
        
        Args:
            k: Budget
            verbose: Print info
            
        Returns:
            (seed_set, computation_time)
        """
        
        start_time = time.time()
        
        # Compute weighted degree for each node
        weighted_degrees = {}
        for node in self.G.nodes():
            weighted_deg = sum(
                self.G[node][neighbor].get('weight', 0.5)
                for neighbor in self.G.neighbors(node)
            )
            weighted_degrees[node] = weighted_deg
        
        # Sort and select top k
        seed_set = sorted(
            weighted_degrees.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]
        seed_set = [node for node, _ in seed_set]
        
        elapsed_time = time.time() - start_time
        
        if verbose:
            print(f"Degree Heuristic selected {k} seeds in {elapsed_time:.4f}s")
        
        return seed_set, elapsed_time


class PageRankIM:
    """
    PageRank-based seed selection.
    
    Intuition: High PageRank nodes have high global importance.
    Fast: O(n + m) with damping factor iterations
    Quality: ~75-88% of greedy
    """
    
    def __init__(self, G: nx.Graph):
        self.G = G
    
    def select_seeds(self, k: int, verbose: bool = True, damping: float = 0.85) -> Tuple[List[str], float]:
        """
        Select k seeds by PageRank.
        
        Args:
            k: Budget
            verbose: Print info
            damping: PageRank damping factor
            
        Returns:
            (seed_set, computation_time)
        """
        
        start_time = time.time()
        
        # Compute PageRank
        pagerank_scores = nx.pagerank(self.G, alpha=damping, weight='weight')
        
        # Sort and select top k
        seed_set = sorted(
            pagerank_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]
        seed_set = [node for node, _ in seed_set]
        
        elapsed_time = time.time() - start_time
        
        if verbose:
            print(f"PageRank selected {k} seeds in {elapsed_time:.4f}s")
        
        return seed_set, elapsed_time


class RandomIM:
    """
    Random baseline: uniformly sample k nodes.
    
    Quality: ~20-30% of greedy (lower bound)
    """
    
    def __init__(self, G: nx.Graph, seed: int = None):
        self.G = G
        np.random.seed(seed)
    
    def select_seeds(self, k: int, verbose: bool = True) -> Tuple[List[str], float]:
        """
        Select k random seeds.
        
        Args:
            k: Budget
            verbose: Print info
            
        Returns:
            (seed_set, computation_time)
        """
        
        start_time = time.time()
        
        nodes = list(self.G.nodes())
        seed_set = list(np.random.choice(nodes, size=k, replace=False))
        
        elapsed_time = time.time() - start_time
        
        if verbose:
            print(f"Random selected {k} seeds in {elapsed_time:.4f}s")
        
        return seed_set, elapsed_time


# Convenience functions

def run_greedy_im(G: nx.Graph, k: int, mc_iterations: int = 1000) -> Tuple[List[str], float]:
    """Wrapper for greedy IM."""
    greedy = GreedyIM(G, mc_iterations=mc_iterations)
    return greedy.select_seeds(k, verbose=True)


def run_degree_heuristic(G: nx.Graph, k: int) -> Tuple[List[str], float]:
    """Wrapper for degree heuristic."""
    degree = DegreeHeuristic(G)
    return degree.select_seeds(k, verbose=True)


def run_pagerank_im(G: nx.Graph, k: int) -> Tuple[List[str], float]:
    """Wrapper for PageRank IM."""
    pagerank = PageRankIM(G)
    return pagerank.select_seeds(k, verbose=True)


def run_random_im(G: nx.Graph, k: int, seed: int = None) -> Tuple[List[str], float]:
    """Wrapper for random baseline."""
    random = RandomIM(G, seed=seed)
    return random.select_seeds(k, verbose=True)


if __name__ == "__main__":
    from data_loader import load_graph
    
    print("Testing baseline algorithms...")
    G = load_graph("data/raw/cybercrime_edge_list.txt")
    
    k = 3
    print(f"\nSelecting {k} seeds from {G.number_of_nodes()} nodes\n")
    
    # Test each baseline
    print("\n1. Random IM")
    seeds_random, time_random = run_random_im(G, k)
    print(f"   Seeds: {seeds_random}")
    
    print("\n2. Degree Heuristic")
    seeds_degree, time_degree = run_degree_heuristic(G, k)
    print(f"   Seeds: {seeds_degree}")
    
    print("\n3. PageRank IM")
    seeds_pagerank, time_pagerank = run_pagerank_im(G, k)
    print(f"   Seeds: {seeds_pagerank}")
    
    print("\n4. Greedy IM (takes longer...)")
    seeds_greedy, time_greedy = run_greedy_im(G, k, mc_iterations=500)
    print(f"   Seeds: {seeds_greedy}")

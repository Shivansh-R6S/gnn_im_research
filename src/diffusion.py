

import networkx as nx
import numpy as np
from typing import Set, List, Tuple
from tqdm import tqdm


def simulate_ic_diffusion(
    G: nx.Graph, 
    seed_set: List[str], 
    mc_iterations: int = 1000,
    verbose: bool = False
) -> Tuple[float, float]:
    """
    
    """
    
    influenced_counts = []
    
    iterator = tqdm(range(mc_iterations), desc="MC Simulation", disable=not verbose)
    
    for _ in iterator:
        # Start with seed set activated
        active = set(seed_set)
        frontier = set(seed_set)  # Nodes to attempt activation from
        
        while frontier:
            next_frontier = set()
            
            for u in frontier:
                # Try to activate each neighbor of u
                for v in G.neighbors(u):
                    if v not in active:
                        # Get edge weight (activation probability)
                        weight = G[u][v].get('weight', 0.5)  # Default to 0.5 if no weight
                        
                        # Attempt activation
                        if np.random.random() < weight:
                            active.add(v)
                            next_frontier.add(v)
            
            frontier = next_frontier
        
        influenced_counts.append(len(active))
    
    mean_influenced = np.mean(influenced_counts)
    std_influenced = np.std(influenced_counts)
    
    return mean_influenced, std_influenced


def estimate_influence(
    G: nx.Graph,
    seed_set: List[str],
    mc_iterations: int = 1000,
    return_ci: bool = True
) -> Tuple[float, Tuple[float, float]]:
    """
    Estimate influence spread with confidence interval.
    
    Args:
        G: NetworkX graph
        seed_set: List of seed nodes
        mc_iterations: Number of MC runs
        return_ci: Return 95% confidence interval
        
    Returns:
        (mean_sigma, (ci_lower, ci_upper))
    """
    mean, std = simulate_ic_diffusion(G, seed_set, mc_iterations=mc_iterations)
    
    if return_ci:
        # 95% CI using normal approximation
        se = std / np.sqrt(mc_iterations)
        ci_lower = mean - 1.96 * se
        ci_upper = mean + 1.96 * se
        return mean, (ci_lower, ci_upper)
    else:
        return mean, (0, 0)


def batch_evaluate_seeds(
    G: nx.Graph,
    seed_sets: dict,
    mc_iterations: int = 1000,
    verbose: bool = True
) -> dict:
    """
    Evaluate multiple seed sets and return influence spreads.
    
    Args:
        G: NetworkX graph
        seed_sets: Dict mapping algorithm name -> list of seed nodes
        mc_iterations: Number of MC runs per evaluation
        verbose: Print progress
        
    Returns:
        Dict mapping algorithm name -> (mean_sigma, ci_lower, ci_upper)
    """
    
    results = {}
    
    for algo_name, seeds in seed_sets.items():
        if verbose:
            print(f"\nEvaluating {algo_name} with seeds: {seeds}")
        
        mean, (ci_lower, ci_upper) = estimate_influence(
            G, seeds, mc_iterations=mc_iterations, return_ci=True
        )
        
        results[algo_name] = {
            'mean': mean,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'seeds': seeds
        }
        
        if verbose:
            print(f"  σ = {mean:.2f} (95% CI: [{ci_lower:.2f}, {ci_upper:.2f}])")
    
    return results


def get_marginal_gain(
    G: nx.Graph,
    current_seeds: Set[str],
    candidate_node: str,
    mc_iterations: int = 100
) -> float:
    """
    Compute marginal gain of adding candidate_node to current_seeds.
    
    Marginal Gain = σ(S ∪ {v}) - σ(S)
    
    Args:
        G: NetworkX graph
        current_seeds: Current seed set
        candidate_node: Node to evaluate
        mc_iterations: MC runs (use fewer for speed during greedy)
        
    Returns:
        Marginal gain value
    """
    
    current_sigma, _ = simulate_ic_diffusion(G, list(current_seeds), mc_iterations=mc_iterations)
    new_sigma, _ = simulate_ic_diffusion(G, list(current_seeds) + [candidate_node], mc_iterations=mc_iterations)
    
    return new_sigma - current_sigma


if __name__ == "__main__":
    # Test with example graph
    from data_loader import load_graph
    
    print("Testing diffusion simulation...")
    G = load_graph("data/raw/cybercrime_edge_list.txt")
    
    # Test single seed
    seed_set = ["Fraud"]
    sigma, ci = estimate_influence(G, seed_set, mc_iterations=500)
    print(f"Influence from {seed_set}: {sigma:.2f} (95% CI: {ci})")
    
    # Test multiple seeds
    seed_set = ["Fraud", "Extortion"]
    sigma, ci = estimate_influence(G, seed_set, mc_iterations=500)
    print(f"Influence from {seed_set}: {sigma:.2f} (95% CI: {ci})")

#!/usr/bin/env python3


import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import pickle
import time

from src import (
    load_graph,
    GreedyIM,
    DegreeHeuristic,
    PageRankIM,
    RandomIM,
    estimate_influence,
    get_marginal_gain,
)
from src.gnn_model import GNNInfluenceMaximizer, get_device


def train_gnn_on_greedy(G, greedy_seeds: List[str], mc_iterations: int = 500):
    """Train GNN to mimic greedy baseline."""
    
    print("\n" + "="*70)
    print("TRAINING GNN ON GREEDY BASELINE LABELS")
    print("="*70)
    
    # Compute marginal gains for greedy seeds
    print("\nComputing marginal gains for greedy seeds...")
    marginal_gains = {}
    
    for i, seed in enumerate(greedy_seeds):
        print(f"  [{i+1}/{len(greedy_seeds)}] Computing gain for '{seed}'")
        
        # Marginal gain = gain from adding this seed
        # Approximate by: influence with this seed vs without
        current_seeds = greedy_seeds[:i]
        gain = get_marginal_gain(G, set(current_seeds), seed, mc_iterations=mc_iterations // 2)
        marginal_gains[seed] = max(0, gain)  # Ensure non-negative
    
    # Train GNN
    device = get_device()
    gnn = GNNInfluenceMaximizer(G, hidden_dim=128, num_layers=2, encoder_type='graphsage', device=device)
    
    gnn.train_supervised(
        greedy_seeds=greedy_seeds,
        greedy_marginal_gains=marginal_gains,
        epochs=50,
        lr=0.001,
        verbose=True
    )
    
    return gnn


def run_experiment_1(dataset_path: str = "data/raw/city_influence_edge_list.txt"):
    """Run Experiment 1: Influence Spread vs Budget."""
    
    # Configuration
    budgets = [1, 2, 3, 4, 5]  # Different k values
    mc_iterations = 500  # For evaluation
    
    print("\n" + "="*70)
    print("EXPERIMENT 1: INFLUENCE SPREAD VS BUDGET")
    print("="*70)
    print(f"Dataset: {dataset_path}")
    print(f"Budgets: {budgets}")
    print(f"MC iterations: {mc_iterations}")
    
    # Load graph
    print(f"\nLoading graph...")
    G = load_graph(dataset_path)
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    
    # Step 1: Run Greedy IM (reference)
    print(f"\n[Step 1] Running Greedy IM...")
    greedy = GreedyIM(G, mc_iterations=mc_iterations)
    greedy_seeds, greedy_time = greedy.select_seeds(k=max(budgets), verbose=True)
    print(f"  Final seeds: {greedy_seeds}")
    print(f"  Computation time: {greedy_time:.4f}s")
    
    # Step 2: Train GNN on greedy labels
    gnn = train_gnn_on_greedy(G, greedy_seeds, mc_iterations=250)
    
    # Step 3: Run all baselines and GNN for each budget
    print(f"\n[Step 2] Running all algorithms for each budget...")
    print("="*70)
    
    results = {
        'budget': [],
        'Greedy_mean': [],
        'Greedy_ci_lower': [],
        'Greedy_ci_upper': [],
        'GNN_mean': [],
        'GNN_ci_lower': [],
        'GNN_ci_upper': [],
        'Degree_mean': [],
        'Degree_ci_lower': [],
        'Degree_ci_upper': [],
        'PageRank_mean': [],
        'PageRank_ci_lower': [],
        'PageRank_ci_upper': [],
        'Random_mean': [],
        'Random_ci_lower': [],
        'Random_ci_upper': [],
    }
    
    for k in budgets:
        print(f"\nBudget k={k}:")
        print("-" * 70)
        
        # Get seeds for each algorithm
        seed_sets = {}
        
        # Greedy (use first k seeds)
        seed_sets['Greedy'] = greedy_seeds[:k]
        print(f"  Greedy:  {seed_sets['Greedy']}")
        
        # GNN
        gnn_seeds = gnn.select_seeds(k)
        seed_sets['GNN'] = gnn_seeds
        print(f"  GNN:     {gnn_seeds}")
        
        # Degree
        degree = DegreeHeuristic(G)
        degree_seeds, _ = degree.select_seeds(k, verbose=False)
        seed_sets['Degree'] = degree_seeds
        print(f"  Degree:  {degree_seeds}")
        
        # PageRank
        pagerank = PageRankIM(G)
        pagerank_seeds, _ = pagerank.select_seeds(k, verbose=False)
        seed_sets['PageRank'] = pagerank_seeds
        print(f"  PageRank: {pagerank_seeds}")
        
        # Random
        random = RandomIM(G, seed=42)
        random_seeds, _ = random.select_seeds(k, verbose=False)
        seed_sets['Random'] = random_seeds
        print(f"  Random:  {random_seeds}")
        
        # Evaluate all
        print(f"\n  Evaluating influence spreads...")
        results['budget'].append(k)
        
        for algo_name, seeds in seed_sets.items():
            mean, (ci_lower, ci_upper) = estimate_influence(G, seeds, mc_iterations=mc_iterations)
            
            results[f'{algo_name}_mean'].append(mean)
            results[f'{algo_name}_ci_lower'].append(ci_lower)
            results[f'{algo_name}_ci_upper'].append(ci_upper)
            
            print(f"    {algo_name:10s}: σ = {mean:7.2f} (95% CI: [{ci_lower:7.2f}, {ci_upper:7.2f}])")
    
    # Step 4: Create results dataframe
    print(f"\n[Step 3] Results Summary")
    print("="*70)
    
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    
    # Save results
    os.makedirs('results', exist_ok=True)
    df_results.to_csv('results/exp1_influence_spread.csv', index=False)
    print(f"\nResults saved to: results/exp1_influence_spread.csv")
    
    # Step 5: Plot results
    print(f"\n[Step 4] Creating visualization...")
    
    plt.figure(figsize=(12, 6))
    
    algorithms = ['Greedy', 'GNN', 'Degree', 'PageRank', 'Random']
    colors = {'Greedy': 'red', 'GNN': 'blue', 'Degree': 'green', 'PageRank': 'orange', 'Random': 'gray'}
    
    for algo in algorithms:
        means = df_results[f'{algo}_mean'].values
        lower = df_results[f'{algo}_ci_lower'].values
        upper = df_results[f'{algo}_ci_upper'].values
        
        plt.errorbar(
            df_results['budget'],
            means,
            yerr=[means - lower, upper - means],
            label=algo,
            color=colors[algo],
            marker='o',
            linewidth=2,
            markersize=8,
            capsize=5,
            capthick=1
        )
    
    plt.xlabel('Budget (k)', fontsize=12, fontweight='bold')
    plt.ylabel('Influence Spread (σ)', fontsize=12, fontweight='bold')
    plt.title('Experiment 1: Influence Spread vs Budget', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = 'results/exp1_influence_spread.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    plt.close()
    
    # Step 6: Compute approximation ratios
    print(f"\n[Step 5] Approximation Ratios vs Greedy")
    print("="*70)
    
    for algo in ['GNN', 'Degree', 'PageRank', 'Random']:
        ratios = (df_results[f'{algo}_mean'] / df_results['Greedy_mean'] * 100).values
        print(f"{algo:10s}: {ratios.mean():.1f}% ± {ratios.std():.1f}% of Greedy")
    
    print(f"\n{'='*70}")
    print("EXPERIMENT 1 COMPLETE")
    print(f"{'='*70}\n")
    
    return df_results, gnn


if __name__ == "__main__":
    df_results, gnn = run_experiment_1()

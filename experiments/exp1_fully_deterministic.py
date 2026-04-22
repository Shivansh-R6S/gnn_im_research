#!/usr/bin/env python3
"""
Experiment 1 FULLY DETERMINISTIC: Multiple GNN Architectures with Complete Reproducibility

This version ensures 100% reproducible results by:
1. Setting seeds at the very beginning
2. Using seeded random generators throughout
3. Avoiding any source of randomness
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================================
# SET SEEDS IMMEDIATELY AT IMPORT TIME
# ============================================================================
import random
import numpy as np
import torch

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Now import everything else
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

print(f"\n{'='*70}")
print(f"RANDOM SEED SET TO {RANDOM_SEED} AT IMPORT TIME")
print(f"{'='*70}\n")


def ensure_seed():
    """Ensure seed is reset (call before any random operation)."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)


def train_gnn_on_greedy(G, greedy_seeds: List[str], mc_iterations: int = 500, 
                        hidden_dim: int = 128, num_layers: int = 2, 
                        encoder_type: str = 'graphsage', epochs: int = 50):
    """Train GNN to mimic greedy baseline."""
    
    ensure_seed()
    
    print(f"\n{'='*70}")
    print(f"Training GNN: {encoder_type.upper()} (hidden_dim={hidden_dim}, layers={num_layers}, epochs={epochs})")
    print(f"{'='*70}")
    
    # Compute marginal gains for greedy seeds
    print("\nComputing marginal gains for greedy seeds...")
    marginal_gains = {}
    
    for i, seed in enumerate(greedy_seeds):
        current_seeds = greedy_seeds[:i]
        gain = get_marginal_gain(G, set(current_seeds), seed, mc_iterations=mc_iterations // 2)
        marginal_gains[seed] = max(0, gain)
    
    # Train GNN
    device = get_device()
    gnn = GNNInfluenceMaximizer(
        G, 
        hidden_dim=hidden_dim, 
        num_layers=num_layers, 
        encoder_type=encoder_type, 
        device=device
    )
    
    gnn.train_supervised(
        greedy_seeds=greedy_seeds,
        greedy_marginal_gains=marginal_gains,
        epochs=epochs,
        lr=0.001,
        verbose=True
    )
    
    return gnn


def run_fully_deterministic_experiment_1(dataset_path: str = "data/raw/cybercrime_edge_list.txt"):
    """Run Experiment 1 with FULL reproducibility guarantee."""
    
    ensure_seed()
    
    budgets = [1, 2, 3, 4, 5]
    mc_iterations = 500
    
    print("\n" + "="*70)
    print("EXPERIMENT 1 FULLY DETERMINISTIC: Multiple GNN Architectures")
    print("="*70)
    print(f"Dataset: {dataset_path}")
    print(f"Budgets: {budgets}")
    print(f"MC iterations: {mc_iterations}")
    print(f"Random Seed: {RANDOM_SEED} (set at import and before each operation)")
    
    # Load graph
    print(f"\nLoading graph...")
    G = load_graph(dataset_path)
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    
    # Step 1: Run Greedy IM (reference)
    print(f"\n[Step 1] Running Greedy IM...")
    ensure_seed()
    greedy = GreedyIM(G, mc_iterations=mc_iterations)
    greedy_seeds, greedy_time = greedy.select_seeds(k=max(budgets), verbose=True)
    print(f"  Final seeds: {greedy_seeds}")
    print(f"  Computation time: {greedy_time:.4f}s")
    
    # Step 2: Test multiple GNN architectures
    print(f"\n[Step 2] Training Multiple GNN Architectures...")
    print("="*70)
    
    gnn_configs = [
        {'hidden_dim': 128, 'num_layers': 2, 'encoder_type': 'graphsage', 'epochs': 50, 'name': 'GraphSAGE-2L-128'},
        {'hidden_dim': 128, 'num_layers': 3, 'encoder_type': 'graphsage', 'epochs': 100, 'name': 'GraphSAGE-3L-128'},
        {'hidden_dim': 256, 'num_layers': 2, 'encoder_type': 'graphsage', 'epochs': 100, 'name': 'GraphSAGE-2L-256'},
        {'hidden_dim': 128, 'num_layers': 2, 'encoder_type': 'gat', 'epochs': 50, 'name': 'GAT-2L-128-4H'},
        {'hidden_dim': 128, 'num_layers': 3, 'encoder_type': 'gat', 'epochs': 100, 'name': 'GAT-3L-128-8H'},
    ]
    
    trained_gnns = {}
    for config in gnn_configs:
        ensure_seed()
        config_name = config.pop('name')
        trained_gnns[config_name] = train_gnn_on_greedy(G, greedy_seeds, mc_iterations=250, **config)
    
    # Step 3: Evaluate all methods for each budget
    print(f"\n[Step 3] Evaluating All Methods...")
    print("="*70)
    
    results_data = []
    
    for k in budgets:
        ensure_seed()
        print(f"\nBudget k={k}:")
        print("-" * 70)
        
        # Get seeds for each method
        seed_sets = {}
        
        # Greedy
        seed_sets['Greedy'] = greedy_seeds[:k]
        
        # GNN variants
        for gnn_name, gnn_model in trained_gnns.items():
            ensure_seed()
            seed_sets[gnn_name] = gnn_model.select_seeds(k)
        
        # Baselines
        ensure_seed()
        degree = DegreeHeuristic(G)
        degree_seeds, _ = degree.select_seeds(k, verbose=False)
        seed_sets['Degree'] = degree_seeds
        
        ensure_seed()
        pagerank = PageRankIM(G)
        pagerank_seeds, _ = pagerank.select_seeds(k, verbose=False)
        seed_sets['PageRank'] = pagerank_seeds
        
        ensure_seed()
        random_im = RandomIM(G, seed=RANDOM_SEED)
        random_seeds, _ = random_im.select_seeds(k, verbose=False)
        seed_sets['Random'] = random_seeds
        
        # Evaluate all
        print(f"Evaluating influence spreads...")
        ensure_seed()
        greedy_mean = estimate_influence(G, seed_sets['Greedy'], mc_iterations=mc_iterations)[0]
        
        for algo_name, seeds in seed_sets.items():
            ensure_seed()
            mean, (ci_lower, ci_upper) = estimate_influence(G, seeds, mc_iterations=mc_iterations)
            
            ratio = (mean / greedy_mean * 100) if greedy_mean > 0 else 0
            
            results_data.append({
                'budget': k,
                'algorithm': algo_name,
                'sigma_mean': mean,
                'sigma_ci_lower': ci_lower,
                'sigma_ci_upper': ci_upper,
                'ratio_to_greedy': ratio
            })
            
            print(f"  {algo_name:20s}: σ = {mean:7.2f} (CI: [{ci_lower:7.2f}, {ci_upper:7.2f}]) → {ratio:6.1f}% of Greedy")
    
    # Create results dataframe
    df_results = pd.DataFrame(results_data)
    
    # Step 4: Summary statistics
    print(f"\n[Step 4] Summary Statistics")
    print("="*70)
    
    summary = df_results.groupby('algorithm').agg({
        'sigma_mean': ['mean', 'std'],
        'ratio_to_greedy': ['mean', 'std']
    }).round(2)
    print(summary)
    
    # Step 5: Save results
    os.makedirs('results', exist_ok=True)
    
    csv_path = 'results/exp1_improved_results.csv'
    df_results.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Step 6: Create visualizations
    print(f"\n[Step 5] Creating Visualizations...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1 = axes[0]
    algorithms = ['Greedy', 'GraphSAGE-2L-128', 'GraphSAGE-3L-128', 'GraphSAGE-2L-256', 
                  'GAT-2L-128-4H', 'GAT-3L-128-8H', 'Degree', 'PageRank', 'Random']
    colors = {
        'Greedy': 'red',
        'GraphSAGE-2L-128': 'blue',
        'GraphSAGE-3L-128': 'darkblue',
        'GraphSAGE-2L-256': 'cyan',
        'GAT-2L-128-4H': 'green',
        'GAT-3L-128-8H': 'darkgreen',
        'Degree': 'orange',
        'PageRank': 'purple',
        'Random': 'gray'
    }
    
    for algo in algorithms:
        algo_data = df_results[df_results['algorithm'] == algo].sort_values('budget')
        if len(algo_data) > 0:
            means = algo_data['sigma_mean'].values
            lower = algo_data['sigma_ci_lower'].values
            upper = algo_data['sigma_ci_upper'].values
            
            ax1.errorbar(
                algo_data['budget'],
                means,
                yerr=[means - lower, upper - means],
                label=algo,
                color=colors.get(algo, 'black'),
                marker='o',
                linewidth=2,
                markersize=6,
                capsize=3,
                capthick=1
            )
    
    ax1.set_xlabel('Budget (k)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Influence Spread (σ)', fontsize=12, fontweight='bold')
    ax1.set_title('Experiment 1 Fully Deterministic: Influence Spread vs Budget', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    
    for algo in algorithms:
        algo_data = df_results[df_results['algorithm'] == algo].sort_values('budget')
        if len(algo_data) > 0:
            ratios = algo_data['ratio_to_greedy'].values
            
            ax2.plot(
                algo_data['budget'],
                ratios,
                label=algo,
                color=colors.get(algo, 'black'),
                marker='o',
                linewidth=2,
                markersize=6
            )
    
    ax2.axhline(y=100, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Perfect (100%)')
    ax2.set_xlabel('Budget (k)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Approximation Ratio (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Approximation Ratio vs Budget', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([85, 105])
    
    plt.tight_layout()
    plot_path = 'results/exp1_improved_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    plt.close()
    
    # Step 7: Summary report
    print(f"\n[Step 6] Summary Report")
    print("="*70)
    
    gnn_algos = ['GraphSAGE-2L-128', 'GraphSAGE-3L-128', 'GraphSAGE-2L-256', 'GAT-2L-128-4H', 'GAT-3L-128-8H']
    
    for algo in gnn_algos:
        algo_data = df_results[df_results['algorithm'] == algo]
        if len(algo_data) > 0:
            avg_ratio = algo_data['ratio_to_greedy'].mean()
            std_ratio = algo_data['ratio_to_greedy'].std()
            avg_sigma = algo_data['sigma_mean'].mean()
            
            print(f"{algo:20s}: avg ratio = {avg_ratio:6.2f}% ± {std_ratio:5.2f}%, avg σ = {avg_sigma:6.2f}")
    
    print(f"\n{'='*70}")
    print("EXPERIMENT 1 FULLY DETERMINISTIC COMPLETE")
    print(f"{'='*70}\n")
    
    return df_results


if __name__ == "__main__":
    df_results = run_fully_deterministic_experiment_1()

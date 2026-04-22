#!/usr/bin/env python3
"""
Experiment 2: Ensemble Learning + Autoencoders for GNN-IM

This experiment combines:
1. Ensemble of 5 GNN models (from exp1)
2. Autoencoder pre-training for better representations
3. Comparison of all approaches

Expected improvements:
- Ensemble: +5-10% over best individual GNN
- VAE-based: +5-15% over baseline GNN
- Combined: +10-20% over baseline GNN
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
import torch

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import pickle
import time

from src import load_graph, GreedyIM, DegreeHeuristic, PageRankIM, RandomIM

# Import FIXED versions
import importlib.util

spec_diffusion = importlib.util.spec_from_file_location(
    "diffusion_fixed", 
    os.path.join(os.path.dirname(__file__), '..', 'src', 'diffusion_fixed.py')
)
diffusion_fixed = importlib.util.module_from_spec(spec_diffusion)
spec_diffusion.loader.exec_module(diffusion_fixed)

# Import functions from diffusion_fixed
estimate_influence = diffusion_fixed.estimate_influence
get_marginal_gain = diffusion_fixed.get_marginal_gain

spec_gnn = importlib.util.spec_from_file_location(
    "gnn_model_fixed",
    os.path.join(os.path.dirname(__file__), '..', 'src', 'gnn_model_fixed.py')
)
gnn_model_fixed = importlib.util.module_from_spec(spec_gnn)
spec_gnn.loader.exec_module(gnn_model_fixed)

spec_ensemble = importlib.util.spec_from_file_location(
    "ensemble",
    os.path.join(os.path.dirname(__file__), '..', 'src', 'ensemble.py')
)
ensemble_module = importlib.util.module_from_spec(spec_ensemble)
spec_ensemble.loader.exec_module(ensemble_module)

spec_autoencoder = importlib.util.spec_from_file_location(
    "autoencoder",
    os.path.join(os.path.dirname(__file__), '..', 'src', 'autoencoder.py')
)
autoencoder_module = importlib.util.module_from_spec(spec_autoencoder)
spec_autoencoder.loader.exec_module(autoencoder_module)

GNNInfluenceMaximizer = gnn_model_fixed.GNNInfluenceMaximizer
get_device = gnn_model_fixed.get_device
GNNEnsemble = ensemble_module.GNNEnsemble
VariationalGraphAutoencoder = autoencoder_module.VariationalGraphAutoencoder
VAEInfluenceMaximizer = autoencoder_module.VAEInfluenceMaximizer

diffusion_fixed.set_diffusion_seed(RANDOM_SEED)


def ensure_seed():
    """Ensure seed is reset."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    diffusion_fixed.set_diffusion_seed(RANDOM_SEED)


def create_ensemble_models(G, greedy_seeds: List[str], mc_iterations: int = 250):
    """Create and train ensemble of GNN models."""
    
    print(f"\n{'='*70}")
    print(f"Creating Ensemble: Training 5 GNN Models")
    print(f"{'='*70}")
    
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
        epochs = config.pop('epochs')  # Extract epochs before passing to __init__
        
        print(f"\nTraining {config_name}...")
        
        # Compute marginal gains
        marginal_gains = {}
        for i, seed in enumerate(greedy_seeds):
            current_seeds = greedy_seeds[:i]
            ensure_seed()
            gain = get_marginal_gain(G, set(current_seeds), seed, mc_iterations=mc_iterations // 2)
            marginal_gains[seed] = max(0, gain)
        
        # Train GNN
        device = get_device()
        gnn = GNNInfluenceMaximizer(G, **config, device=device)
        
        gnn.train_supervised(
            greedy_seeds=greedy_seeds,
            greedy_marginal_gains=marginal_gains,
            epochs=epochs,
            lr=0.001,
            verbose=True
        )
        
        trained_gnns[config_name] = gnn
    
    return trained_gnns


def create_vae_model(G, greedy_seeds: List[str], greedy_marginal_gains: Dict, epochs_vae: int = 100, epochs_im: int = 50):
    """Create and train VAE-based influence maximization model."""
    
    print(f"\n{'='*70}")
    print(f"Creating VAE-based IM Model")
    print(f"{'='*70}")
    
    device = get_device()
    
    # Create VAE
    print(f"\n[Step 1] Pre-training VAE on graph reconstruction...")
    vae = VariationalGraphAutoencoder(
        num_nodes=G.number_of_nodes(),
        input_dim=1,
        hidden_dim=128,
        latent_dim=64,
        num_layers=2,
        device=device
    )
    
    ensure_seed()
    vae.pretrain(G, epochs=epochs_vae, lr=0.001, beta=0.5, verbose=True)
    
    # Create VAE-based IM model
    print(f"\n[Step 2] Fine-tuning VAE for IM task...")
    vae_im = VAEInfluenceMaximizer(G, vae, device=device)
    
    ensure_seed()
    vae_im.train_supervised(
        greedy_seeds=greedy_seeds,
        greedy_marginal_gains=greedy_marginal_gains,
        epochs=epochs_im,
        lr=0.001,
        verbose=True
    )
    
    return vae_im


def run_experiment_2(dataset_path: str = "data/raw/cybercrime_edge_list.txt"):
    """Run Experiment 2: Ensemble + Autoencoders."""
    
    ensure_seed()
    
    budgets = [1, 2, 3, 4, 5]
    mc_iterations = 500
    
    print("\n" + "="*70)
    print("EXPERIMENT 2: Ensemble Learning + Autoencoders for GNN-IM")
    print("="*70)
    print(f"Dataset: {dataset_path}")
    print(f"Budgets: {budgets}")
    print(f"MC iterations: {mc_iterations}")
    
    # Load graph
    print(f"\nLoading graph...")
    G = load_graph(dataset_path)
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    
    # Step 1: Run Greedy IM
    print(f"\n[Step 1] Running Greedy IM (Reference)...")
    ensure_seed()
    greedy = GreedyIM(G, mc_iterations=mc_iterations)
    greedy_seeds, greedy_time = greedy.select_seeds(k=max(budgets), verbose=True)
    print(f"  Seeds: {greedy_seeds}, Time: {greedy_time:.4f}s")
    
    # Compute marginal gains for training
    greedy_marginal_gains = {}
    for i, seed in enumerate(greedy_seeds):
        current_seeds = greedy_seeds[:i]
        ensure_seed()
        gain = get_marginal_gain(G, set(current_seeds), seed, mc_iterations=mc_iterations // 2)
        greedy_marginal_gains[seed] = max(0, gain)
    
    # Step 2: Create Ensemble Models
    print(f"\n[Step 2] Creating Ensemble of 5 GNN Models...")
    ensemble_models = create_ensemble_models(G, greedy_seeds, mc_iterations=250)
    ensemble = GNNEnsemble(list(ensemble_models.values()), device=get_device())
    
    # Print ensemble diversity
    diversity = ensemble.evaluate_diversity()
    
    # Step 3: Create VAE-based Model
    print(f"\n[Step 3] Creating VAE-based IM Model...")
    vae_im = create_vae_model(G, greedy_seeds, greedy_marginal_gains, epochs_vae=100, epochs_im=50)
    
    # Step 4: Evaluate all methods
    print(f"\n[Step 4] Evaluating All Methods...")
    print("="*70)
    
    results_data = []
    
    for k in budgets:
        ensure_seed()
        print(f"\nBudget k={k}:")
        print("-" * 70)
        
        seed_sets = {}
        
        # Greedy (reference)
        seed_sets['Greedy'] = greedy_seeds[:k]
        
        # Individual GNN models
        for gnn_name, gnn_model in ensemble_models.items():
            ensure_seed()
            seed_sets[gnn_name] = gnn_model.select_seeds(k)
        
        # Ensemble (average scores)
        ensure_seed()
        seed_sets['Ensemble-Avg'] = ensemble.select_seeds(k, method='average')
        
        # Ensemble (majority voting)
        ensure_seed()
        seed_sets['Ensemble-Vote'] = ensemble.select_seeds(k, method='vote')
        
        # VAE-based IM
        ensure_seed()
        seed_sets['VAE-IM'] = vae_im.select_seeds(k)
        
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
        greedy_mean = diffusion_fixed.estimate_influence(G, seed_sets['Greedy'], mc_iterations=mc_iterations)[0]
        
        for algo_name, seeds in seed_sets.items():
            ensure_seed()
            mean, (ci_lower, ci_upper) = diffusion_fixed.estimate_influence(G, seeds, mc_iterations=mc_iterations)
            ratio = (mean / greedy_mean * 100) if greedy_mean > 0 else 0
            
            results_data.append({
                'budget': k,
                'algorithm': algo_name,
                'sigma_mean': mean,
                'sigma_ci_lower': ci_lower,
                'sigma_ci_upper': ci_upper,
                'ratio_to_greedy': ratio
            })
            
            print(f"  {algo_name:20s}: σ = {mean:7.2f} → {ratio:6.1f}% of Greedy")
    
    df_results = pd.DataFrame(results_data)
    
    # Step 5: Summary
    print(f"\n[Step 5] Summary Statistics")
    print("="*70)
    
    summary = df_results.groupby('algorithm').agg({
        'sigma_mean': ['mean', 'std'],
        'ratio_to_greedy': ['mean', 'std']
    }).round(2)
    print(summary)
    
    # Step 6: Save results
    os.makedirs('results', exist_ok=True)
    
    csv_path = 'results/exp2_ensemble_vae_results.csv'
    df_results.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Step 7: Create visualizations
    print(f"\n[Step 6] Creating Visualizations...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1, ax2 = axes[0], axes[1]
    
    # Select key algorithms to plot (too many is cluttered)
    plot_algos = [
        'Greedy', 
        'GraphSAGE-2L-128',  # Best individual
        'Ensemble-Avg',       # Ensemble average
        'Ensemble-Vote',      # Ensemble voting
        'VAE-IM',             # VAE-based
        'Degree', 
        'Random'
    ]
    
    colors = {
        'Greedy': 'red',
        'GraphSAGE-2L-128': 'blue',
        'Ensemble-Avg': 'darkgreen',
        'Ensemble-Vote': 'green',
        'VAE-IM': 'purple',
        'Degree': 'orange',
        'Random': 'gray'
    }
    
    # Plot 1: Influence Spread
    for algo in plot_algos:
        algo_data = df_results[df_results['algorithm'] == algo].sort_values('budget')
        if len(algo_data) > 0:
            ax1.errorbar(
                algo_data['budget'],
                algo_data['sigma_mean'],
                yerr=[algo_data['sigma_mean'] - algo_data['sigma_ci_lower'],
                      algo_data['sigma_ci_upper'] - algo_data['sigma_mean']],
                label=algo,
                color=colors.get(algo, 'black'),
                marker='o',
                linewidth=2,
                markersize=6,
                capsize=3
            )
    
    ax1.set_xlabel('Budget (k)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Influence Spread (σ)', fontsize=12, fontweight='bold')
    ax1.set_title('Experiment 2: Influence Spread Comparison', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Approximation Ratio
    for algo in plot_algos:
        algo_data = df_results[df_results['algorithm'] == algo].sort_values('budget')
        if len(algo_data) > 0:
            ax2.plot(
                algo_data['budget'],
                algo_data['ratio_to_greedy'],
                label=algo,
                color=colors.get(algo, 'black'),
                marker='o',
                linewidth=2,
                markersize=6
            )
    
    ax2.axhline(y=100, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Budget (k)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Approximation Ratio (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Approximation Ratio Comparison', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([85, 105])
    
    plt.tight_layout()
    plot_path = 'results/exp2_ensemble_vae_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    plt.close()
    
    # Step 8: Summary report
    print(f"\n[Step 7] Final Report")
    print("="*70)
    
    key_algos = ['GraphSAGE-2L-128', 'Ensemble-Avg', 'Ensemble-Vote', 'VAE-IM']
    
    for algo in key_algos:
        algo_data = df_results[df_results['algorithm'] == algo]
        if len(algo_data) > 0:
            avg_ratio = algo_data['ratio_to_greedy'].mean()
            std_ratio = algo_data['ratio_to_greedy'].std()
            print(f"{algo:20s}: avg ratio = {avg_ratio:6.2f}% ± {std_ratio:5.2f}%")
    
    print(f"\n{'='*70}")
    print("EXPERIMENT 2 COMPLETE")
    print(f"{'='*70}\n")
    
    return df_results


if __name__ == "__main__":
    df_results = run_experiment_2()

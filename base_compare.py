#!/usr/bin/env python3
"""
Quick Start: Test Data Loading & Baseline Algorithms

This script:
1. Loads the cybercrime dataset
2. Prints graph statistics
3. Runs baseline algorithms (Random, Degree, PageRank, Greedy)
4. Evaluates and compares results
"""

import sys
import os

# Add parent dir to path so we can import src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import (
    load_graph, 
    print_graph_info, 
    run_random_im, 
    run_degree_heuristic, 
    run_pagerank_im, 
    run_greedy_im,
    batch_evaluate_seeds
)


def main():
    """Run quick start tests."""
    
    # Configuration
    dataset_path = "data/raw/cybercrime_edge_list.txt"
    k = 3  # Number of seeds
    mc_iterations = 500  # MC runs for evaluation
    
    print("\n" + "="*70)
    print("GNN-BASED INFLUENCE MAXIMIZATION - QUICK START")
    print("="*70)
    
    # Step 1: Load graph
    print(f"\n[Step 1] Loading graph from: {dataset_path}")
    try:
        G = load_graph(dataset_path)
    except FileNotFoundError:
        print(f"ERROR: Could not find {dataset_path}")
        print("Please ensure the cybercrime_edge_list.txt is in data/raw/ directory")
        return
    
    # Step 2: Print statistics
    print(f"\n[Step 2] Graph Statistics")
    print_graph_info(G)
    
    # Step 3: Run baselines
    print(f"\n[Step 3] Running Baseline Algorithms (k={k} seeds, {mc_iterations} MC iterations for evaluation)")
    print("-" * 70)
    
    baseline_results = {}
    
    # Random IM
    print(f"\nRunning Random IM...")
    seeds_random, time_random = run_random_im(G, k)
    baseline_results['Random'] = seeds_random
    print(f"  Time: {time_random:.4f}s")
    print(f"  Seeds: {seeds_random}")
    
    # Degree Heuristic
    print(f"\nRunning Degree Heuristic...")
    seeds_degree, time_degree = run_degree_heuristic(G, k)
    baseline_results['Degree'] = seeds_degree
    print(f"  Time: {time_degree:.4f}s")
    print(f"  Seeds: {seeds_degree}")
    
    # PageRank IM
    print(f"\nRunning PageRank IM...")
    seeds_pagerank, time_pagerank = run_pagerank_im(G, k)
    baseline_results['PageRank'] = seeds_pagerank
    print(f"  Time: {time_pagerank:.4f}s")
    print(f"  Seeds: {seeds_pagerank}")
    
    # Greedy IM (reference)
    print(f"\nRunning Greedy IM (reference, takes longer)...")
    seeds_greedy, time_greedy = run_greedy_im(G, k, mc_iterations=mc_iterations)
    baseline_results['Greedy'] = seeds_greedy
    print(f"  Time: {time_greedy:.4f}s")
    print(f"  Seeds: {seeds_greedy}")
    
    # Step 4: Evaluate all methods
    print(f"\n[Step 4] Evaluating All Methods (Influence Spread)")
    print("-" * 70)
    
    eval_results = batch_evaluate_seeds(G, baseline_results, mc_iterations=mc_iterations, verbose=True)
    
    # Step 5: Compare results
    print(f"\n[Step 5] Comparison Summary")
    print("-" * 70)
    print(f"{'Algorithm':<15} {'σ (spread)':<15} {'95% CI':<30} {'Time (s)':<12}")
    print("-" * 70)
    
    times = {
        'Random': time_random,
        'Degree': time_degree,
        'PageRank': time_pagerank,
        'Greedy': time_greedy
    }
    
    for algo, results in eval_results.items():
        mean = results['mean']
        ci_lower = results['ci_lower']
        ci_upper = results['ci_upper']
        elapsed = times[algo]
        
        ci_str = f"[{ci_lower:.2f}, {ci_upper:.2f}]"
        print(f"{algo:<15} {mean:<15.2f} {ci_str:<30} {elapsed:<12.4f}")
    
    # Speedup vs Greedy
    print(f"\n[Step 6] Speedup vs Greedy Baseline")
    print("-" * 70)
    greedy_time = time_greedy
    for algo in ['Random', 'Degree', 'PageRank']:
        algo_time = times[algo]
        # Avoid division by zero for very fast algorithms
        if algo_time > 0:
            speedup = greedy_time / algo_time
            print(f"{algo:<15} {speedup:.2f}× faster than Greedy")
        else:
            print(f"{algo:<15} Extremely fast (too fast to measure accurately)")
    
    # Approximation ratio vs Greedy
    print(f"\n[Step 7] Approximation Ratio vs Greedy (Quality)")
    print("-" * 70)
    greedy_spread = eval_results['Greedy']['mean']
    for algo in ['Random', 'Degree', 'PageRank']:
        spread = eval_results[algo]['mean']
        ratio = (spread / greedy_spread * 100) if greedy_spread > 0 else 0
        print(f"{algo:<15} {ratio:.1f}% of Greedy spread")
    
    print("\n" + "="*70)
    print("QUICK START COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("  1. Verify baseline results above")
    print("  2. Run: python src/gnn_model.py to build GNN")
    print("  3. Run: python experiments/exp1_influence_spread.py for full experiments")
    print()


if __name__ == "__main__":
    main()

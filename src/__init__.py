"""
GNN-Based Influence Maximization Research Package
"""

from .data_loader import load_graph, save_graph, load_saved_graph, get_graph_statistics, print_graph_info
from .baselines import (
    GreedyIM, DegreeHeuristic, PageRankIM, RandomIM,
    run_greedy_im, run_degree_heuristic, run_pagerank_im, run_random_im
)
from .diffusion import simulate_ic_diffusion, estimate_influence, batch_evaluate_seeds, get_marginal_gain

__version__ = "0.1.0"
__all__ = [
    'load_graph',
    'save_graph',
    'load_saved_graph',
    'get_graph_statistics',
    'print_graph_info',
    'GreedyIM',
    'DegreeHeuristic',
    'PageRankIM',
    'RandomIM',
    'run_greedy_im',
    'run_degree_heuristic',
    'run_pagerank_im',
    'run_random_im',
    'simulate_ic_diffusion',
    'estimate_influence',
    'batch_evaluate_seeds',
    'get_marginal_gain',
]

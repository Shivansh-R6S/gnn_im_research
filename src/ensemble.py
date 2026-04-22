"""
Ensemble Learning for GNN-based Influence Maximization

Combines predictions from multiple GNN models to improve seed selection.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm


class GNNEnsemble:
    """Ensemble of GNN models for influence maximization."""
    
    def __init__(self, trained_models: List, device: str = 'cpu'):
        """
        Args:
            trained_models: List of trained GNNInfluenceMaximizer models
            device: 'cpu' or 'cuda'
        """
        self.models = trained_models
        self.num_models = len(trained_models)
        self.device = device
        
        # Get node mapping from first model
        self.idx_to_node = trained_models[0].idx_to_node
        self.num_nodes = trained_models[0].num_nodes
        
        print(f"[INFO] Ensemble initialized with {self.num_models} models")
    
    def select_seeds(self, k: int, method: str = 'average') -> List[str]:
        """
        Select seeds using ensemble method.
        
        Args:
            k: Number of seeds to select
            method: 'average' (average scores) or 'vote' (majority voting)
            
        Returns:
            List of k seed node names
        """
        if method == 'average':
            return self._average_scores(k)
        elif method == 'vote':
            return self._majority_vote(k)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _average_scores(self, k: int) -> List[str]:
        """
        Select seeds by averaging predicted influence scores.
        
        Args:
            k: Number of seeds
            
        Returns:
            List of k seed node names
        """
        
        print(f"\n[Ensemble] Selecting {k} seeds via score averaging...")
        
        # Collect scores from each model
        all_scores = []
        for i, model in enumerate(self.models):
            with torch.no_grad():
                scores = model.forward().squeeze(1).cpu().numpy()
            all_scores.append(scores)
            print(f"  Model {i+1}/{self.num_models}: Scores computed")
        
        # Average across models
        ensemble_scores = np.mean(all_scores, axis=0)
        ensemble_std = np.std(all_scores, axis=0)
        
        # Select top-k
        top_k_indices = np.argsort(-ensemble_scores)[:k]
        seeds = [self.idx_to_node[idx] for idx in top_k_indices]
        
        print(f"[Ensemble] Selected seeds: {seeds}")
        print(f"[Ensemble] Score stats - Mean: {ensemble_scores.mean():.4f}, Std: {ensemble_std.mean():.4f}")
        
        return seeds
    
    def _majority_vote(self, k: int) -> List[str]:
        """
        Select seeds by majority voting.
        
        Each model votes for its top-2k seeds, seeds with most votes are selected.
        
        Args:
            k: Number of seeds
            
        Returns:
            List of k seed node names
        """
        
        print(f"\n[Ensemble] Selecting {k} seeds via majority voting...")
        
        # Get top-2k from each model (to ensure diversity)
        all_seed_lists = []
        for i, model in enumerate(self.models):
            seeds = model.select_seeds(min(k * 2, self.num_nodes))
            all_seed_lists.append(seeds)
            print(f"  Model {i+1}/{self.num_models}: Top seeds = {seeds[:min(k, len(seeds))]}")
        
        # Count votes
        vote_counts = {}
        for seeds in all_seed_lists:
            for seed in seeds:
                vote_counts[seed] = vote_counts.get(seed, 0) + 1
        
        # Select top-k by vote count
        top_seeds = sorted(vote_counts.items(), 
                          key=lambda x: x[1], 
                          reverse=True)[:k]
        seeds = [seed for seed, count in top_seeds]
        
        print(f"[Ensemble] Selected seeds (by votes): {seeds}")
        print(f"[Ensemble] Vote counts: {dict(top_seeds)}")
        
        return seeds
    
    def get_ensemble_scores(self) -> np.ndarray:
        """
        Get averaged scores for all nodes.
        
        Returns:
            Array of shape (num_nodes,) with averaged influence scores
        """
        
        all_scores = []
        for model in self.models:
            with torch.no_grad():
                scores = model.forward().squeeze(1).cpu().numpy()
            all_scores.append(scores)
        
        ensemble_scores = np.mean(all_scores, axis=0)
        return ensemble_scores
    
    def evaluate_diversity(self) -> Dict:
        """
        Evaluate diversity of ensemble predictions.
        
        Returns:
            Dictionary with diversity metrics
        """
        
        print("\n[Ensemble] Evaluating prediction diversity...")
        
        # Collect scores from each model
        all_scores = []
        for model in self.models:
            with torch.no_grad():
                scores = model.forward().squeeze(1).cpu().numpy()
            all_scores.append(scores)
        
        all_scores = np.array(all_scores)  # Shape: (num_models, num_nodes)
        
        # Compute correlation between model pairs
        correlations = []
        for i in range(self.num_models):
            for j in range(i+1, self.num_models):
                corr = np.corrcoef(all_scores[i], all_scores[j])[0, 1]
                correlations.append(corr)
        
        metrics = {
            'mean_correlation': np.mean(correlations),
            'std_correlation': np.std(correlations),
            'min_correlation': np.min(correlations),
            'max_correlation': np.max(correlations),
            'avg_score_std': np.mean(np.std(all_scores, axis=0))
        }
        
        print(f"[Ensemble] Diversity metrics:")
        for key, val in metrics.items():
            print(f"  {key}: {val:.4f}")
        
        return metrics
    
    def select_seeds_with_confidence(self, k: int) -> Tuple[List[str], Dict]:
        """
        Select seeds with confidence estimates.
        
        Args:
            k: Number of seeds
            
        Returns:
            (seeds, confidence_dict) where confidence_dict maps seed -> confidence score
        """
        
        print(f"\n[Ensemble] Selecting {k} seeds with confidence estimates...")
        
        # Collect scores from each model
        all_scores = []
        for model in self.models:
            with torch.no_grad():
                scores = model.forward().squeeze(1).cpu().numpy()
            all_scores.append(scores)
        
        all_scores = np.array(all_scores)  # Shape: (num_models, num_nodes)
        
        # Compute mean and confidence (std)
        mean_scores = np.mean(all_scores, axis=0)
        std_scores = np.std(all_scores, axis=0)
        
        # Select top-k by mean score
        top_k_indices = np.argsort(-mean_scores)[:k]
        seeds = [self.idx_to_node[idx] for idx in top_k_indices]
        
        # Create confidence dict
        confidence = {}
        for idx in top_k_indices:
            node = self.idx_to_node[idx]
            confidence[node] = {
                'mean_score': float(mean_scores[idx]),
                'std_score': float(std_scores[idx]),
                'confidence': float(1.0 / (1.0 + std_scores[idx]))  # Higher std = lower confidence
            }
        
        print(f"[Ensemble] Selected seeds with confidence:")
        for seed in seeds:
            conf = confidence[seed]
            print(f"  {seed}: mean={conf['mean_score']:.4f}, std={conf['std_score']:.4f}, confidence={conf['confidence']:.4f}")
        
        return seeds, confidence


def load_trained_models(model_paths: List[str], device: str = 'cpu'):
    """
    Load pre-trained GNN models from paths.
    
    Args:
        model_paths: List of paths to saved models
        device: 'cpu' or 'cuda'
        
    Returns:
        List of loaded models
    """
    
    models = []
    for path in model_paths:
        # Assuming models are saved as checkpoint
        # Implement loading logic based on your save format
        print(f"Loading model from {path}")
        # model = torch.load(path)
        # models.append(model)
    
    return models


if __name__ == "__main__":
    print("Ensemble module loaded successfully")

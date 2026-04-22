"""
GNN-Based Influence Maximization Model (FIXED FOR DETERMINISM)

Architecture:
- Graph Encoder: GraphSAGE (learns node embeddings from graph structure)
- Influence Decoder: MLP (maps embeddings to influence scores)
- Training: Supervised learning on greedy IM labels

The model learns to predict which nodes are important for influence spread,
eliminating the need for expensive Monte Carlo simulations at inference time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import networkx as nx
from typing import List, Tuple, Dict
import time
from tqdm import tqdm

# Try to import torch_geometric
try:
    from torch_geometric.data import Data
    from torch_geometric.nn import SAGEConv, GATConv
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    print("Warning: torch_geometric not installed. Install with: pip install torch_geometric")
    TORCH_GEOMETRIC_AVAILABLE = False


class GraphSAGEEncoder(nn.Module):
    """GraphSAGE encoder: learns node embeddings via neighborhood aggregation."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        """
        Args:
            input_dim: Feature dimension (1 for nodes with only degree/weight info)
            hidden_dim: Hidden embedding dimension
            num_layers: Number of aggregation layers
        """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_weight: Edge weights (ignored for GraphSAGE)
            
        Returns:
            Node embeddings [num_nodes, hidden_dim]
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        return x


class GATEncoder(nn.Module):
    """Graph Attention Network encoder: learns which neighbors to attend."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, num_heads: int = 4):
        """
        Args:
            input_dim: Feature dimension
            hidden_dim: Hidden embedding dimension
            num_layers: Number of attention layers
            num_heads: Number of attention heads
        """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # GAT layers with better attention handling
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim // num_heads, heads=num_heads, concat=True))
        
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, concat=True))
        
        self.dropout = nn.Dropout(0.3)
        self.batch_norm = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
    
    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_weight: Edge weights (not directly used in GAT)
            
        Returns:
            Node embeddings [num_nodes, hidden_dim]
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.batch_norm[i](x)
                x = F.relu(x)
                x = self.dropout(x)
        
        return x


class InfluenceHead(nn.Module):
    """MLP head that predicts influence scores from embeddings."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        """
        Args:
            input_dim: Embedding dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: Node embeddings [num_nodes, input_dim]
            
        Returns:
            Influence scores [num_nodes, 1]
        """
        return self.net(x)


class GNNInfluenceMaximizer(nn.Module):
    """
    Complete GNN model for influence maximization.
    
    Combines:
    - Graph encoder (learns structure)
    - Influence decoder (predicts scores)
    """
    
    def __init__(
        self,
        G: nx.Graph,
        input_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 2,
        encoder_type: str = 'graphsage',
        device: str = 'cpu'
    ):
        """
        Args:
            G: NetworkX graph
            input_dim: Feature dimension (default: 1, just a placeholder)
            hidden_dim: Hidden embedding dimension
            num_layers: Number of encoder layers
            encoder_type: 'graphsage' or 'gat'
            device: 'cpu' or 'cuda'
        """
        super().__init__()
        
        # ENSURE DETERMINISTIC BEHAVIOR
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)
        
        self.G = G
        self.device = device
        self.num_nodes = G.number_of_nodes()
        self.node_to_idx = {node: idx for idx, node in enumerate(G.nodes())}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
        
        # Convert graph to PyG format
        self.edge_index, self.edge_weight = self._graph_to_edge_tensor()
        
        # Build encoder WITH EXPLICIT SEED
        if encoder_type.lower() == 'graphsage':
            if not TORCH_GEOMETRIC_AVAILABLE:
                raise RuntimeError("torch_geometric required for GraphSAGE. Install with: pip install torch_geometric")
            torch.manual_seed(42)
            self.encoder = GraphSAGEEncoder(input_dim, hidden_dim, num_layers)
        elif encoder_type.lower() == 'gat':
            if not TORCH_GEOMETRIC_AVAILABLE:
                raise RuntimeError("torch_geometric required for GAT. Install with: pip install torch_geometric")
            torch.manual_seed(42)
            self.encoder = GATEncoder(input_dim, hidden_dim, num_layers)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        # Build decoder WITH EXPLICIT SEED
        torch.manual_seed(42)
        self.decoder = InfluenceHead(hidden_dim)
        
        self.to(device)
    
    def _graph_to_edge_tensor(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert NetworkX graph to PyG edge format."""
        edges = []
        weights = []
        
        for u, v, data in self.G.edges(data=True):
            u_idx = self.node_to_idx[u]
            v_idx = self.node_to_idx[v]
            weight = data.get('weight', 0.5)
            
            # Add both directions for undirected graph
            edges.append([u_idx, v_idx])
            edges.append([v_idx, u_idx])
            weights.extend([weight, weight])
        
        edge_index = torch.LongTensor(edges).t().contiguous().to(self.device)
        edge_weight = torch.FloatTensor(weights).to(self.device)
        
        return edge_index, edge_weight
    
    def forward(self, x=None):
        """
        Forward pass: predict influence scores for all nodes.
        
        Args:
            x: Optional node features. If None, uses ones as placeholder.
            
        Returns:
            Influence scores [num_nodes, 1]
        """
        if x is None:
            # Use ones as placeholder features
            x = torch.ones((self.num_nodes, 1), device=self.device)
        
        # Encode (GraphSAGE doesn't use edge weights, just structure)
        embeddings = self.encoder(x, self.edge_index)
        
        # Decode
        scores = self.decoder(embeddings)
        
        return scores
    
    def select_seeds(self, k: int) -> List[str]:
        """
        Select k seeds based on learned influence scores.
        
        Args:
            k: Budget (number of seeds)
            
        Returns:
            List of k seed node names
        """
        with torch.no_grad():
            scores = self.forward()  # [num_nodes, 1]
            scores = scores.squeeze(1).cpu().numpy()  # [num_nodes]
        
        # Get top k indices
        top_k_indices = np.argsort(-scores)[:k]
        
        # Convert to node names
        seeds = [self.idx_to_node[idx] for idx in top_k_indices]
        
        return seeds
    
    def train_supervised(
        self,
        greedy_seeds: List[str],
        greedy_marginal_gains: Dict[str, float],
        epochs: int = 50,
        lr: float = 0.001,
        verbose: bool = True
    ):
        """
        Train GNN on greedy IM labels (supervised learning).
        
        Args:
            greedy_seeds: Seeds selected by greedy algorithm
            greedy_marginal_gains: Dict mapping node -> marginal gain
            epochs: Number of training epochs
            lr: Learning rate
            verbose: Print progress
        """
        
        # ENSURE DETERMINISTIC TRAINING
        torch.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Prepare training targets
        y = torch.zeros(self.num_nodes, device=self.device)
        for node, gain in greedy_marginal_gains.items():
            idx = self.node_to_idx[node]
            y[idx] = gain
        
        # Normalize to [0, 1]
        y = y / (y.max() + 1e-8)
        
        # Optimizer WITH EXPLICIT SEED
        torch.manual_seed(42)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        
        # Training loop
        iterator = tqdm(range(epochs), desc="Training GNN", disable=not verbose)
        
        for epoch in iterator:
            optimizer.zero_grad()
            
            # Forward pass
            scores = self.forward().squeeze(1)  # [num_nodes]
            
            # Loss
            loss = loss_fn(scores, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if verbose:
                iterator.set_postfix({'loss': f'{loss.item():.4f}'})
        
        print(f"\nTraining complete. Final loss: {loss.item():.4f}")
    
    def evaluate(self, greedy_seeds: List[str], greedy_scores: Dict[str, float]) -> float:
        """
        Evaluate GNN prediction quality against greedy baseline.
        
        Args:
            greedy_seeds: Greedy seeds
            greedy_scores: Dict mapping node -> influence score
            
        Returns:
            Mean Absolute Error between predicted and greedy scores
        """
        with torch.no_grad():
            predictions = self.forward().squeeze(1).cpu().numpy()
        
        # Prepare targets
        targets = np.zeros(self.num_nodes)
        for node, score in greedy_scores.items():
            idx = self.node_to_idx[node]
            targets[idx] = score
        
        # Normalize both
        targets = targets / (targets.max() + 1e-8)
        predictions = predictions / (predictions.max() + 1e-8)
        
        mae = np.mean(np.abs(predictions - targets))
        return mae


def get_device() -> str:
    """Get available device (CUDA if available, else CPU)."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == "__main__":
    print("GNN Model Fixed module loaded successfully")

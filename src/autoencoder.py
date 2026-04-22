"""
Variational Autoencoder (VAE) for Graph Structure Learning

Pre-trains on graph reconstruction to learn better node representations,
then fine-tunes for influence maximization task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import networkx as nx
from typing import Tuple, List, Dict
from tqdm import tqdm

try:
    from torch_geometric.nn import SAGEConv, GATConv
except ImportError:
    print("Warning: torch_geometric not installed")


class GraphSAGEEncoder(nn.Module):
    """GraphSAGE encoder for VAE."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x


class VAEMeanVarLayer(nn.Module):
    """Outputs mean and log-variance for VAE."""
    
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.fc_mean = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)
    
    def forward(self, x):
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar


class GraphDecoder(nn.Module):
    """Decoder for reconstructing edge weights from latent representations."""
    
    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Bilinear layer for edge prediction: (z_i, z_j) -> edge_weight
        self.edge_predictor = nn.Bilinear(hidden_dim, hidden_dim, 1)
    
    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Predict edge weights from latent representations.
        
        Args:
            z: Latent representations [num_nodes, latent_dim]
            edge_index: Edge connectivity [2, num_edges]
            
        Returns:
            Predicted edge weights [num_edges, 1]
        """
        # Process latent through hidden layers
        h = F.relu(self.fc(z))
        h = F.relu(self.fc2(h))
        
        # Get source and target nodes for each edge
        src, dst = edge_index
        
        # Predict edge weights using bilinear layer
        edge_weights = self.edge_predictor(h[src], h[dst])
        edge_probs = torch.sigmoid(edge_weights)
        
        return edge_probs


class VariationalGraphAutoencoder(nn.Module):
    """Variational Autoencoder for learning graph representations."""
    
    def __init__(
        self,
        num_nodes: int,
        input_dim: int = 1,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_layers: int = 2,
        device: str = 'cpu'
    ):
        """
        Args:
            num_nodes: Number of nodes in graph
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension for encoder
            latent_dim: Latent space dimension
            num_layers: Number of encoder layers
            device: 'cpu' or 'cuda'
        """
        super().__init__()
        
        torch.manual_seed(42)
        
        self.num_nodes = num_nodes
        self.latent_dim = latent_dim
        self.device = device
        
        # Encoder: Graph -> Latent space
        self.base_encoder = GraphSAGEEncoder(input_dim, hidden_dim, num_layers)
        self.mean_var = VAEMeanVarLayer(hidden_dim, latent_dim)
        
        # Decoder: Latent -> Reconstructed edges
        self.decoder = GraphDecoder(latent_dim, hidden_dim)
        
        self.to(device)
    
    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode graph to latent space.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            
        Returns:
            (mean, logvar) for reparameterization
        """
        h = self.base_encoder(x, edge_index)
        mean, logvar = self.mean_var(h)
        return mean, logvar
    
    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: sample from N(mean, std).
        
        Args:
            mean: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z
    
    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to edge predictions.
        
        Args:
            z: Latent vector
            edge_index: Edge connectivity
            
        Returns:
            Predicted edge weights
        """
        return self.decoder(z, edge_index)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full VAE forward pass.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            
        Returns:
            (z, recon_edges, mean, logvar)
        """
        mean, logvar = self.encode(x, edge_index)
        z = self.reparameterize(mean, logvar)
        recon_edges = self.decode(z, edge_index)
        
        return z, recon_edges, mean, logvar
    
    def loss_function(
        self,
        recon_edges: torch.Tensor,
        true_edges: torch.Tensor,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0
    ) -> torch.Tensor:
        """
        VAE loss = Reconstruction Loss + β * KL Divergence.
        
        Args:
            recon_edges: Reconstructed edge weights
            true_edges: True edge weights
            mean: Mean of latent distribution
            logvar: Log variance of latent distribution
            beta: Weight for KL term
            
        Returns:
            Total loss
        """
        # Reconstruction loss (Binary Cross-Entropy)
        recon_loss = F.binary_cross_entropy(
            recon_edges.squeeze(),
            true_edges,
            reduction='mean'
        )
        
        # KL divergence loss (regularization)
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / self.num_nodes
        
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def pretrain(
        self,
        G: nx.Graph,
        epochs: int = 100,
        lr: float = 0.001,
        beta: float = 0.5,
        verbose: bool = True
    ) -> Dict:
        """
        Pre-train VAE on graph reconstruction.
        
        Args:
            G: NetworkX graph
            epochs: Number of training epochs
            lr: Learning rate
            beta: KL divergence weight
            verbose: Print progress
            
        Returns:
            Training history
        """
        
        print(f"\n{'='*70}")
        print(f"Pre-training VAE on graph reconstruction")
        print(f"{'='*70}")
        print(f"Epochs: {epochs}, Learning Rate: {lr}, Beta: {beta}")
        
        # Prepare data
        x = torch.ones((self.num_nodes, 1), device=self.device)
        
        # Create node index mapping
        node_to_idx = {node: idx for idx, node in enumerate(G.nodes())}
        
        edges = []
        weights = []
        for u, v, data in G.edges(data=True):
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            edges.append([u_idx, v_idx])
            edges.append([v_idx, u_idx])  # Undirected
            weight = data.get('weight', 0.5)
            weights.extend([weight, weight])
        
        edge_index = torch.LongTensor(edges).t().contiguous().to(self.device)
        true_edges = torch.FloatTensor(weights).to(self.device)
        
        # Optimizer
        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        history = {
            'total_loss': [],
            'recon_loss': [],
            'kl_loss': []
        }
        
        # Training loop
        iterator = tqdm(range(epochs), desc="Pre-training VAE", disable=not verbose)
        
        for epoch in iterator:
            optimizer.zero_grad()
            
            # Forward pass
            z, recon_edges, mean, logvar = self(x, edge_index)
            
            # Loss
            total_loss, recon_loss, kl_loss = self.loss_function(
                recon_edges, true_edges, mean, logvar, beta
            )
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Record history
            history['total_loss'].append(total_loss.item())
            history['recon_loss'].append(recon_loss.item())
            history['kl_loss'].append(kl_loss.item())
            
            if verbose:
                iterator.set_postfix({
                    'total': f'{total_loss.item():.4f}',
                    'recon': f'{recon_loss.item():.4f}',
                    'kl': f'{kl_loss.item():.4f}'
                })
        
        print(f"\nPre-training complete!")
        print(f"Final loss: {history['total_loss'][-1]:.4f}")
        
        return history
    
    def get_embeddings(
        self,
        G: nx.Graph
    ) -> torch.Tensor:
        """
        Get learned embeddings for all nodes.
        
        Args:
            G: NetworkX graph
            
        Returns:
            Embeddings [num_nodes, latent_dim]
        """
        
        x = torch.ones((self.num_nodes, 1), device=self.device)
        
        # Create node index mapping
        node_to_idx = {node: idx for idx, node in enumerate(G.nodes())}
        
        edges = []
        for u, v in G.edges():
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            edges.append([u_idx, v_idx])
            edges.append([v_idx, u_idx])
        
        edge_index = torch.LongTensor(edges).t().contiguous().to(self.device)
        
        with torch.no_grad():
            mean, _ = self.encode(x, edge_index)
        
        return mean


class VAEInfluenceMaximizer(nn.Module):
    """Use VAE-learned embeddings for influence maximization."""
    
    def __init__(
        self,
        G: nx.Graph,
        vae: VariationalGraphAutoencoder,
        hidden_dim: int = 64,
        device: str = 'cpu'
    ):
        """
        Args:
            G: NetworkX graph
            vae: Pre-trained VAE model
            hidden_dim: Hidden dimension for influence head
            device: 'cpu' or 'cuda'
        """
        super().__init__()
        
        torch.manual_seed(42)
        
        self.G = G
        self.vae = vae
        self.device = device
        self.num_nodes = vae.num_nodes
        self.latent_dim = vae.latent_dim
        
        # Node mapping
        self.node_to_idx = {node: idx for idx, node in enumerate(G.nodes())}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
        
        # Influence prediction head
        self.influence_head = nn.Sequential(
            nn.Linear(vae.latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.to(device)
    
    def forward(self, x: torch.Tensor = None, edge_index: torch.Tensor = None) -> torch.Tensor:
        """
        Predict influence scores using VAE embeddings.
        
        Args:
            x: Node features (optional, will use ones if None)
            edge_index: Edge connectivity (optional, will compute if None)
            
        Returns:
            Influence scores [num_nodes, 1]
        """
        
        if x is None:
            x = torch.ones((self.num_nodes, 1), device=self.device)
        
        if edge_index is None:
            edges = []
            for u, v in self.G.edges():
                edges.append([u, v])
                edges.append([v, u])
            edge_index = torch.LongTensor(edges).t().contiguous().to(self.device)
        
        # Get VAE embeddings
        with torch.no_grad():
            mean, _ = self.vae.encode(x, edge_index)
        
        # Predict influence scores
        scores = self.influence_head(mean)
        
        return scores
    
    def select_seeds(self, k: int) -> List[str]:
        """
        Select k seeds based on learned influence scores.
        
        Args:
            k: Number of seeds
            
        Returns:
            List of k seed node names
        """
        
        with torch.no_grad():
            scores = self.forward().squeeze(1).cpu().numpy()
        
        top_k_indices = np.argsort(-scores)[:k]
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
        Fine-tune influence head on greedy IM labels.
        
        Args:
            greedy_seeds: Seeds selected by greedy algorithm
            greedy_marginal_gains: Dict mapping node -> marginal gain
            epochs: Number of training epochs
            lr: Learning rate
            verbose: Print progress
        """
        
        print(f"\n{'='*70}")
        print(f"Fine-tuning VAE-based IM on greedy labels")
        print(f"{'='*70}")
        
        # Prepare targets
        y = torch.zeros(self.num_nodes, device=self.device)
        for node, gain in greedy_marginal_gains.items():
            idx = self.node_to_idx[node]
            y[idx] = gain
        
        # Normalize to [0, 1]
        y = y / (y.max() + 1e-8)
        
        # Optimizer (only optimize influence head, freeze VAE)
        optimizer = optim.Adam(self.influence_head.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        
        # Training loop
        iterator = tqdm(range(epochs), desc="Fine-tuning IM", disable=not verbose)
        
        for epoch in iterator:
            optimizer.zero_grad()
            
            # Forward pass
            scores = self.forward().squeeze(1)
            
            # Loss
            loss = loss_fn(scores, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if verbose:
                iterator.set_postfix({'loss': f'{loss.item():.4f}'})
        
        print(f"\nFine-tuning complete!")


if __name__ == "__main__":
    print("Autoencoder module loaded successfully")

"""
models.py — GNN Architectures for Outbreak Prediction
=======================================================
AgriConnect | GNN Module | Harsha G Gowda (1RV24CS406)

Three models defined:
  1. GraphSAGEModel  — core research model (inductive, dynamic-graph-safe)
  2. GCNModel        — baseline 1
  3. GATModel        — baseline 2

All models perform node-level binary classification:
  Output: probability (0–1) that a farm will have a pest outbreak within 7 days.
  Threshold: >= 0.65 → HIGH RISK alert
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GATConv, BatchNorm


# ---------------------------------------------------------------------------
# 1. GraphSAGE Model (Core Research Model)
# ---------------------------------------------------------------------------
class GraphSAGEModel(nn.Module):
    """
    3-layer GraphSAGE (Graph Sample and Aggregate) for outbreak prediction.

    Why GraphSAGE over GCN/GAT?
      - Inductive: handles NEW farms added without full retraining.
      - Scalable: neighbourhood sampling → works on large, evolving graphs.
      - Federated-friendly: trained locally per client, aggregated via FedAvg.

    Architecture:
      SAGEConv(19 → 128) → BN → ReLU → Dropout(0.3)
      SAGEConv(128 → 64) → BN → ReLU → Dropout(0.3)
      SAGEConv(64  → 32) → BN → ReLU
      Linear(32 → 1)     → Sigmoid
    """

    def __init__(
        self,
        in_channels: int = 19,
        hidden1: int = 128,
        hidden2: int = 64,
        hidden3: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden1, aggr="mean")
        self.bn1   = BatchNorm(hidden1)
        self.conv2 = SAGEConv(hidden1, hidden2, aggr="mean")
        self.bn2   = BatchNorm(hidden2)
        self.conv3 = SAGEConv(hidden2, hidden3, aggr="mean")
        self.bn3   = BatchNorm(hidden3)
        self.head  = nn.Linear(hidden3, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 3
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)

        # Output head
        x = self.head(x)
        return torch.sigmoid(x).squeeze(-1)   # shape: (N,)

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Returns the final-layer node embeddings before the linear head (for SHAP)."""
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        return x


# ---------------------------------------------------------------------------
# 2. GCN Baseline
# ---------------------------------------------------------------------------
class GCNModel(nn.Module):
    """
    3-layer Graph Convolutional Network (Kipf & Welling, 2017).
    Used as Baseline 1 for comparison against GraphSAGE.

    Limitations vs GraphSAGE (why it is the baseline, not the main model):
      - Transductive: cannot handle new nodes without retraining entire graph.
      - Full graph propagation: not scalable to very large farm networks.
    """

    def __init__(
        self,
        in_channels: int = 19,
        hidden1: int = 128,
        hidden2: int = 64,
        hidden3: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden1, add_self_loops=True)
        self.bn1   = BatchNorm(hidden1)
        self.conv2 = GCNConv(hidden1, hidden2, add_self_loops=True)
        self.bn2   = BatchNorm(hidden2)
        self.conv3 = GCNConv(hidden2, hidden3, add_self_loops=True)
        self.bn3   = BatchNorm(hidden3)
        self.head  = nn.Linear(hidden3, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.dropout(F.relu(self.bn1(self.conv1(x, edge_index))), p=self.dropout, training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x, edge_index))), p=self.dropout, training=self.training)
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        return torch.sigmoid(self.head(x)).squeeze(-1)


# ---------------------------------------------------------------------------
# 3. GAT Baseline
# ---------------------------------------------------------------------------
class GATModel(nn.Module):
    """
    3-layer Graph Attention Network (Velickovic et al., 2018).
    Used as Baseline 2 for comparison against GraphSAGE.

    Uses multi-head attention (4 heads → concat at each layer).
    Limitation: attention weights fixed after training — not inductive.
    """

    def __init__(
        self,
        in_channels: int = 19,
        hidden1: int = 32,   # per-head; 4 heads → effective 128
        hidden2: int = 16,   # per-head; 4 heads → effective 64
        hidden3: int = 32,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden1, heads=heads, concat=True,  dropout=dropout)
        self.bn1   = BatchNorm(hidden1 * heads)
        self.conv2 = GATConv(hidden1 * heads, hidden2, heads=heads, concat=True, dropout=dropout)
        self.bn2   = BatchNorm(hidden2 * heads)
        self.conv3 = GATConv(hidden2 * heads, hidden3, heads=1, concat=False, dropout=dropout)
        self.bn3   = BatchNorm(hidden3)
        self.head  = nn.Linear(hidden3, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.dropout(F.elu(self.bn1(self.conv1(x, edge_index))), p=self.dropout, training=self.training)
        x = F.dropout(F.elu(self.bn2(self.conv2(x, edge_index))), p=self.dropout, training=self.training)
        x = F.elu(self.bn3(self.conv3(x, edge_index)))
        return torch.sigmoid(self.head(x)).squeeze(-1)


# ---------------------------------------------------------------------------
# Helper — model factory
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "graphsage": GraphSAGEModel,
    "gcn":       GCNModel,
    "gat":       GATModel,
}


def build_model(name: str, in_channels: int = 19, **kwargs) -> nn.Module:
    """Factory function: returns a model by string name."""
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](in_channels=in_channels, **kwargs)


# ---------------------------------------------------------------------------
# Quick parameter count check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for name, Cls in MODEL_REGISTRY.items():
        model = Cls()
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name:12s} : {params:,} trainable parameters")

        # Forward pass sanity check
        x = torch.randn(100, 19)
        ei = torch.randint(0, 100, (2, 300))
        out = model(x, ei)
        print(f"             Output shape: {out.shape} | min={out.min():.3f} max={out.max():.3f}\n")

"""
fl_client.py — Federated Learning Client (Flower)
===================================================
AgriConnect | GNN Module | Harsha G Gowda (1RV24CS406)

Each FL client represents one Karnataka district:
  Client 0 → Kolar
  Client 1 → Tumkur
  Client 2 → Hassan
  Client 3 → Mandya
  Client 4 → Mysuru

The client:
  1. Receives the global GraphSAGE model weights from the FL server.
  2. Trains locally on its district subgraph for 3 epochs.
  3. Sends ONLY updated gradients/weights back — raw farm data NEVER leaves.

Privacy guarantee:
  Raw GPS, crop health, and yield data stays on the district-level client.
  Only floating-point weight tensors are transmitted over the wire.

Usage (run 5 separate processes — one per district):
    python fl_client.py --district Kolar   --server localhost:8080
    python fl_client.py --district Tumkur  --server localhost:8080
    python fl_client.py --district Hassan  --server localhost:8080
    python fl_client.py --district Mandya  --server localhost:8080
    python fl_client.py --district Mysuru  --server localhost:8080

Or run via fl_server.py simulation (recommended for local testing):
    python fl_server.py --simulate
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import flwr as fl
from flwr.common import (
    NDArrays, Scalar, Config, FitRes, EvaluateRes, FitIns, EvaluateIns,
    GetParametersIns, GetParametersRes, Parameters, ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from typing import Dict

from dataset import build_farm_graph, partition_by_district
from models  import GraphSAGEModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RISK_THRESHOLD = 0.65
LOCAL_EPOCHS   = 3
LEARNING_RATE  = 1e-3
WEIGHT_DECAY   = 5e-4


# ---------------------------------------------------------------------------
# Flower NumPyClient for one district
# ---------------------------------------------------------------------------
class AgriConnectGNNClient(fl.client.NumPyClient):
    """
    A Flower federated client that:
      - Holds a district-level subgraph as its local dataset
      - Wraps a GraphSAGE model
      - Performs local training and returns updated weights
      - Privacy: only model weights are shared, never raw farm data
    """

    def __init__(
        self,
        district_name: str,
        local_data,                          # PyG Data for this district
        in_channels: int = 19,
        local_epochs: int = LOCAL_EPOCHS,
    ):
        self.district     = district_name
        self.data         = local_data.to(DEVICE)
        self.local_epochs = local_epochs
        self.in_channels  = in_channels

        # ---- Local model instance ----
        self.model = GraphSAGEModel(in_channels=in_channels).to(DEVICE)

        # ---- Class-imbalance: weight positive class ----
        n_pos = float(self.data.y[self.data.train_mask].sum())
        n_neg = float(self.data.train_mask.sum()) - n_pos
        pos_w = torch.tensor([n_neg / max(n_pos, 1)]).to(DEVICE)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w * 2.0)

        self._log(f"Initialized | Nodes: {self.data.num_nodes} | "
                  f"Train: {self.data.train_mask.sum()} | "
                  f"Positive labels: {int(self.data.y[self.data.train_mask].sum())}")

    def _log(self, msg: str):
        print(f"[FL-Client:{self.district}] {msg}")

    # ---- Flower API ----

    def get_parameters(self, config: Config) -> NDArrays:
        """Return current model weights as numpy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: NDArrays):
        """Load server-aggregated weights into local model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict  = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: Config) -> tuple:
        """
        Local training round:
          1. Load global model weights
          2. Train for self.local_epochs epochs on district subgraph
          3. Return updated weights + training metadata
        """
        self.set_parameters(parameters)

        optimizer = Adam(self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        self.model.train()

        total_loss = 0.0
        for epoch in range(self.local_epochs):
            optimizer.zero_grad()
            out    = self.model(self.data.x, self.data.edge_index)
            logits = torch.logit(out.clamp(1e-6, 1 - 1e-6))
            loss   = self.criterion(
                logits[self.data.train_mask],
                self.data.y[self.data.train_mask].float()
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / self.local_epochs
        n_samples = int(self.data.train_mask.sum())
        self._log(f"Fit complete | Avg Loss: {avg_loss:.4f} | Samples: {n_samples}")

        return self.get_parameters(config={}), n_samples, {"loss": avg_loss}

    def evaluate(self, parameters: NDArrays, config: Config) -> tuple:
        """
        Evaluates the global model on this client's local validation set.
        Returns loss + accuracy metrics to the server.
        """
        self.set_parameters(parameters)
        self.model.eval()

        with torch.no_grad():
            out    = self.model(self.data.x, self.data.edge_index)
            logits = torch.logit(out.clamp(1e-6, 1 - 1e-6))
            loss   = self.criterion(
                logits[self.data.val_mask],
                self.data.y[self.data.val_mask].float()
            ).item()

            probs  = out[self.data.val_mask].cpu().numpy()
            labels = self.data.y[self.data.val_mask].cpu().numpy()
            preds  = (probs >= RISK_THRESHOLD).astype(int)

        accuracy  = float((preds == labels).mean()) if len(labels) > 0 else 0.0
        n_samples = int(self.data.val_mask.sum())

        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.0
        except Exception:
            auc = 0.0

        self._log(f"Eval | Loss: {loss:.4f} | Acc: {accuracy:.4f} | AUC: {auc:.4f}")

        return loss, n_samples, {"accuracy": accuracy, "auc_roc": auc}


# ---------------------------------------------------------------------------
# Client factory — builds a client for a given district name
# ---------------------------------------------------------------------------
def build_client_for_district(district_name: str, n_farms: int = 500, seed: int = 42):
    """
    Builds and returns a Flower client for the specified district.
    Internally generates the full farm graph and extracts the district partition.
    """
    print(f"[FL-Client:{district_name}] Loading district subgraph...")
    data, _ = build_farm_graph(n_farms=n_farms, seed=seed, save_meta=False)
    subgraphs = partition_by_district(data)

    if district_name not in subgraphs:
        raise ValueError(
            f"District '{district_name}' not found. "
            f"Available: {list(subgraphs.keys())}"
        )

    return AgriConnectGNNClient(
        district_name=district_name,
        local_data=subgraphs[district_name],
        in_channels=data.x.shape[1],
    )


# ---------------------------------------------------------------------------
# Entry point — start client connected to FL server
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgriConnect FL Client")
    parser.add_argument(
        "--district", type=str, required=True,
        choices=["Kolar", "Tumkur", "Hassan", "Mandya", "Mysuru"],
        help="Karnataka district this client represents",
    )
    parser.add_argument("--server",   type=str, default="localhost:8080",
                        help="FL server address (host:port)")
    parser.add_argument("--n_farms",  type=int, default=500)
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    client = build_client_for_district(args.district, args.n_farms, args.seed)

    print(f"[FL-Client:{args.district}] Connecting to server at {args.server}...")
    fl.client.start_client(
        server_address=args.server,
        client=client.to_client(),
    )

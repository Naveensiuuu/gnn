"""
fl_server.py — Federated Learning Server (Flower)
===================================================
AgriConnect | GNN Module | Harsha G Gowda (1RV24CS406)

Orchestrates the Federated Training protocol:
  - Aggregates weight updates from 5 district clients using FedAvg
  - Runs 10 global rounds, each client trains 3 local epochs
  - Tracks federated accuracy across rounds (research result)
  - Saves the final global GraphSAGE model for deployment

Privacy guarantee:
  The server NEVER sees raw farm data — only model weights are received.
  FedAvg computes a weighted average of weights proportional to dataset size.

Two modes:
  1. Simulation mode (--simulate) — runs all 5 clients in the same process.
     Best for local development / testing on a single machine.
  2. Real mode (default) — starts a gRPC server; clients connect externally.
     Use when deploying to district-level edge servers.

Usage:
    # Simulation (recommended for local testing):
    python fl_server.py --simulate --rounds 10 --n_farms 500

    # Real distributed server:
    python fl_server.py --host 0.0.0.0 --port 8080 --rounds 10
"""

import os
import json
import argparse
import numpy as np
import torch
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Metrics
from typing import List, Tuple, Dict, Optional
from sklearn.metrics import roc_auc_score, f1_score

from dataset import build_farm_graph, partition_by_district
from models  import GraphSAGEModel
from fl_client import AgriConnectGNNClient

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RISK_THRESHOLD = 0.65
NUM_ROUNDS     = 10
MIN_CLIENTS    = 5
DISTRICTS      = ["Kolar", "Tumkur", "Hassan", "Mandya", "Mysuru"]
CHECKPOINT_DIR = "checkpoints"


# ---------------------------------------------------------------------------
# Custom FedAvg strategy with metric aggregation and logging
# ---------------------------------------------------------------------------
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregates evaluation metrics from all clients using weighted average.
    Weight = number of local validation samples.
    """
    total_samples = sum(n for n, _ in metrics)
    if total_samples == 0:
        return {}

    agg_accuracy = sum(n * m.get("accuracy", 0.0) for n, m in metrics) / total_samples
    agg_auc      = sum(n * m.get("auc_roc",  0.0) for n, m in metrics) / total_samples

    return {"accuracy": agg_accuracy, "auc_roc": agg_auc}


class AgriConnectFedAvg(FedAvg):
    """
    Extended FedAvg strategy for AgriConnect.
    Adds:
      - Per-round metric logging
      - Model checkpoint saving after each round
      - Research result tracking (federated AUC-ROC across rounds)
    """

    def __init__(self, model_template: GraphSAGEModel, results_log: list, **kwargs):
        super().__init__(**kwargs)
        self.model_template = model_template
        self.results_log    = results_log
        self.round_counter  = 0

    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures,
    ):
        """Called after each global round with client evaluation results."""
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        if aggregated_metrics:
            auc = aggregated_metrics.get("auc_roc", 0.0)
            acc = aggregated_metrics.get("accuracy", 0.0)
            print(
                f"\n  [FL-Server] Round {server_round:2d}/{NUM_ROUNDS} | "
                f"Federated AUC-ROC: {auc:.4f} | "
                f"Federated Accuracy: {acc:.4f}"
            )
            self.results_log.append({
                "round":    server_round,
                "auc_roc":  round(float(auc), 4),
                "accuracy": round(float(acc), 4),
                "loss":     round(float(aggregated_loss) if aggregated_loss else 0.0, 5),
            })

        return aggregated_loss, aggregated_metrics

    def aggregate_fit(self, server_round: int, results, failures):
        """Called after each global round with client fit results. Saves checkpoint."""
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            # Save global model checkpoint
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            params_numpy = fl.common.parameters_to_ndarrays(aggregated_parameters)
            params_dict  = zip(self.model_template.state_dict().keys(), params_numpy)
            state_dict   = {k: torch.tensor(v) for k, v in params_dict}
            self.model_template.load_state_dict(state_dict, strict=True)

            ckpt_path = os.path.join(CHECKPOINT_DIR, f"graphsage_federated_round{server_round:02d}.pt")
            torch.save(self.model_template.state_dict(), ckpt_path)

            if server_round == NUM_ROUNDS:
                final_path = os.path.join(CHECKPOINT_DIR, "graphsage_federated_final.pt")
                torch.save(self.model_template.state_dict(), final_path)
                print(f"\n  [FL-Server] Final global model saved: {final_path}")

        return aggregated_parameters, aggregated_metrics


# ---------------------------------------------------------------------------
# Simulation mode — runs all clients in-process using Flower's simulation API
# ---------------------------------------------------------------------------
def run_simulation(n_farms: int = 500, seed: int = 42, n_rounds: int = NUM_ROUNDS):
    """
    Runs the complete federated training loop locally using Flower's
    virtual client engine (no actual network connections required).

    This is the RECOMMENDED mode for local development and reproducibility.
    """
    print("\n" + "="*65)
    print("  AGRICONNECT FEDERATED GRAPHSAGE — SIMULATION MODE")
    print("="*65)
    print(f"  Districts: {DISTRICTS}")
    print(f"  Rounds   : {n_rounds}")
    print(f"  Farms    : {n_farms}")
    print(f"  Device   : {DEVICE}")
    print("="*65 + "\n")

    # ---- Build the full graph once and partition ----
    print("[FL-Server] Building synthetic Karnataka farm graph...")
    data, _ = build_farm_graph(n_farms=n_farms, seed=seed, save_meta=True)
    subgraphs = partition_by_district(data)
    in_channels = data.x.shape[1]

    # ---- Pre-build all 5 clients ----
    clients = {}
    for district in DISTRICTS:
        if district in subgraphs:
            clients[district] = AgriConnectGNNClient(
                district_name=district,
                local_data=subgraphs[district],
                in_channels=in_channels,
            )
        else:
            print(f"  [FL-Server] Warning: No nodes found for district '{district}'. Skipping.")

    n_active_clients = len(clients)
    print(f"\n[FL-Server] Active clients: {n_active_clients} / {len(DISTRICTS)}")

    if n_active_clients < 2:
        print("[FL-Server] ERROR: Need at least 2 clients for federated training.")
        return

    # ---- Results log ----
    results_log = []
    model_template = GraphSAGEModel(in_channels=in_channels).to(DEVICE)

    # ---- Manual federated training loop ----
    # (Using manual loop instead of fl.simulation for graph data compatibility)
    print("\n[FL-Server] Starting federated training rounds...\n")

    # Initial global weights
    global_weights = [val.cpu().numpy() for _, val in model_template.state_dict().items()]

    for round_num in range(1, n_rounds + 1):
        print(f"  {'─'*55}")
        print(f"  Global Round {round_num}/{n_rounds}")
        print(f"  {'─'*55}")

        client_weights   = []
        client_sizes     = []
        client_eval_data = []

        # ---- Client local training ----
        for district, client in clients.items():
            updated_weights, n_samples, fit_metrics = client.fit(
                parameters=global_weights, config={}
            )
            client_weights.append(updated_weights)
            client_sizes.append(n_samples)
            print(f"    ✓ {district:10s} | samples: {n_samples} | loss: {fit_metrics['loss']:.4f}")

        # ---- FedAvg aggregation ----
        total_samples  = sum(client_sizes)
        global_weights = [
            sum(w[i] * (s / total_samples) for w, s in zip(client_weights, client_sizes))
            for i in range(len(global_weights))
        ]

        # ---- Distribute updated global weights ----
        params_dict = zip(model_template.state_dict().keys(), global_weights)
        state_dict  = {k: torch.tensor(v) for k, v in params_dict}
        model_template.load_state_dict(state_dict, strict=True)

        # ---- Client evaluation with global model ----
        eval_results = []
        for district, client in clients.items():
            loss, n_val, eval_metrics = client.evaluate(
                parameters=global_weights, config={}
            )
            if n_val > 0:
                eval_results.append((n_val, eval_metrics))

        # ---- Aggregate evaluation ----
        if eval_results:
            agg = weighted_average(eval_results)
            results_log.append({
                "round":    round_num,
                "auc_roc":  round(float(agg.get("auc_roc", 0)), 4),
                "accuracy": round(float(agg.get("accuracy", 0)), 4),
            })
            print(
                f"\n  [Round {round_num:2d}] "
                f"Federated AUC-ROC: {agg.get('auc_roc', 0):.4f} | "
                f"Accuracy: {agg.get('accuracy', 0):.4f}\n"
            )

        # ---- Save round checkpoint ----
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"graphsage_federated_round{round_num:02d}.pt")
        torch.save(model_template.state_dict(), ckpt_path)

    # ---- Final global model evaluation on full test set ----
    print("\n" + "="*65)
    print("  FEDERATED TRAINING COMPLETE — FINAL EVALUATION")
    print("="*65)

    model_template.eval()
    data = data.to(DEVICE)
    with torch.no_grad():
        probs  = model_template(data.x, data.edge_index)[data.test_mask].cpu().numpy()
        labels = data.y[data.test_mask].cpu().numpy()
        preds  = (probs >= RISK_THRESHOLD).astype(int)

    try:
        auc = roc_auc_score(labels, probs)
        f1  = f1_score(labels, preds, zero_division=0)
    except Exception:
        auc, f1 = 0.0, 0.0

    print(f"  Final Federated AUC-ROC  : {auc:.4f}")
    print(f"  Final Federated F1       : {f1:.4f}")
    print(f"  Final Accuracy           : {float((preds == labels).mean()):.4f}")

    # ---- Save final model ----
    final_path = os.path.join(CHECKPOINT_DIR, "graphsage_federated_final.pt")
    torch.save(model_template.state_dict(), final_path)
    print(f"\n  Final global model saved: {final_path}")

    # ---- Save results ----
    results_output = {
        "final_test_auc_roc":  round(float(auc), 4),
        "final_test_f1":       round(float(f1), 4),
        "final_test_accuracy": round(float((preds == labels).mean()), 4),
        "round_history":       results_log,
    }
    with open("federated_results.json", "w") as fp:
        json.dump(results_output, fp, indent=2)
    print("  Results saved to federated_results.json")
    print("="*65 + "\n")

    return results_output


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgriConnect FL Server")
    parser.add_argument("--simulate", action="store_true",
                        help="Run simulation (all clients in-process)")
    parser.add_argument("--host",    type=str, default="0.0.0.0")
    parser.add_argument("--port",    type=int, default=8080)
    parser.add_argument("--rounds",  type=int, default=NUM_ROUNDS)
    parser.add_argument("--n_farms", type=int, default=500)
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    if args.simulate:
        run_simulation(n_farms=args.n_farms, seed=args.seed, n_rounds=args.rounds)
    else:
        print(f"[FL-Server] Starting gRPC server on {args.host}:{args.port}")
        print(f"[FL-Server] Waiting for {MIN_CLIENTS} clients to connect...")
        model_template = GraphSAGEModel(in_channels=19).to(DEVICE)
        results_log    = []

        strategy = AgriConnectFedAvg(
            model_template=model_template,
            results_log=results_log,
            min_fit_clients=MIN_CLIENTS,
            min_evaluate_clients=MIN_CLIENTS,
            min_available_clients=MIN_CLIENTS,
            evaluate_metrics_aggregation_fn=weighted_average,
        )

        fl.server.start_server(
            server_address=f"{args.host}:{args.port}",
            config=fl.server.ServerConfig(num_rounds=args.rounds),
            strategy=strategy,
        )

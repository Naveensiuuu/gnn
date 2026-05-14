"""
test_gnn.py — GNN Benchmark & Comparison Script
=================================================
AgriConnect | GNN Module | Harsha G Gowda (1RV24CS406)

Runs the complete research benchmark:
  1. Trains GraphSAGE, GCN, GAT (centralized) and compares metrics
  2. Runs federated GraphSAGE simulation and records per-round AUC-ROC
  3. Computes the Centralized vs Federated performance gap (target: < 5%)
  4. Generates a benchmark summary JSON for inclusion in the research paper

Usage:
    python test_gnn.py                # full benchmark (may take 10–20 min)
    python test_gnn.py --quick        # 30 epochs, 100 farms (for smoke test)
    python test_gnn.py --no_fed       # skip federated training
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from dataset import build_farm_graph
from models  import build_model

# ---------------------------------------------------------------------------
# Imports from other modules (with error handling for missing deps)
# ---------------------------------------------------------------------------
try:
    from train_centralized import train_model, DEFAULT_CONFIG
    TRAIN_AVAILABLE = True
except ImportError as e:
    print(f"[test_gnn] Warning: train_centralized import failed: {e}")
    TRAIN_AVAILABLE = False

try:
    from fl_server import run_simulation
    FL_AVAILABLE = True
except ImportError as e:
    print(f"[test_gnn] Warning: fl_server import failed: {e}")
    FL_AVAILABLE = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RISK_THRESHOLD = 0.65


# ---------------------------------------------------------------------------
# Quick model evaluation (loads a saved checkpoint)
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_checkpoint(
    model_name: str,
    ckpt_path: str,
    data,
    mask_name: str = "test_mask",
) -> dict:
    """Loads a checkpoint and evaluates on the given mask."""
    model = build_model(model_name, in_channels=data.x.shape[1]).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    mask  = getattr(data, mask_name)
    probs = model(data.x.to(DEVICE), data.edge_index.to(DEVICE))[mask].cpu().numpy()
    labels = data.y[mask].cpu().numpy()
    preds  = (probs >= RISK_THRESHOLD).astype(int)

    try:
        auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
    except Exception:
        auc = 0.0

    return {
        "auc_roc":   round(float(auc), 4),
        "f1":        round(float(f1_score(labels, preds, zero_division=0)), 4),
        "precision": round(float(precision_score(labels, preds, zero_division=0)), 4),
        "recall":    round(float(recall_score(labels, preds, zero_division=0)), 4),
        "accuracy":  round(float((preds == labels).mean()), 4),
    }


# ---------------------------------------------------------------------------
# Comparison table printer
# ---------------------------------------------------------------------------
def print_benchmark_table(centralized: dict, federated: Optional[dict] = None):
    """Prints a formatted research benchmark table."""
    SEP = "=" * 75
    print(f"\n{SEP}")
    print("  AGRICONNECT — GNN RESEARCH BENCHMARK RESULTS")
    print(f"  Dataset : Synthetic Karnataka Farm Network (500 nodes, 5 districts)")
    print(f"  Device  : {DEVICE}")
    print(SEP)

    headers = f"  {'Model':20s} | {'AUC-ROC':>8} | {'F1':>7} | {'Precision':>10} | {'Recall':>8} | {'Accuracy':>9}"
    print(headers)
    print("  " + "-" * 71)

    for name, metrics in centralized.items():
        star = " ★" if name == "graphsage" else "  "
        row = (
            f"  {name.upper():20s}{star}| "
            f"{metrics['auc_roc']:>8.4f} | "
            f"{metrics['f1']:>7.4f} | "
            f"{metrics['precision']:>10.4f} | "
            f"{metrics['recall']:>8.4f} | "
            f"{metrics['accuracy']:>9.4f}"
        )
        print(row)

    if federated:
        print("  " + "-" * 71)
        row = (
            f"  {'GRAPHSAGE (FEDERATED)':20s}  | "
            f"{federated['auc_roc']:>8.4f} | "
            f"{federated['f1']:>7.4f} | "
            f"{federated['precision']:>10.4f} | "
            f"{federated['recall']:>8.4f} | "
            f"{federated['accuracy']:>9.4f}"
        )
        print(row)

    print(SEP)

    # Research targets
    print("\n  RESEARCH TARGETS CHECK:")
    sage_auc = centralized.get("graphsage", {}).get("auc_roc", 0)
    gcn_auc  = centralized.get("gcn",       {}).get("auc_roc", 0)
    gat_auc  = centralized.get("gat",       {}).get("auc_roc", 0)

    targets = [
        ("GraphSAGE AUC-ROC > 85%",
         sage_auc >= 0.85, f"{sage_auc:.4f}"),
        ("GraphSAGE > GCN",
         sage_auc > gcn_auc,
         f"GraphSAGE {sage_auc:.4f} vs GCN {gcn_auc:.4f}"),
        ("GraphSAGE > GAT",
         sage_auc > gat_auc,
         f"GraphSAGE {sage_auc:.4f} vs GAT {gat_auc:.4f}"),
    ]

    if federated:
        gap = abs(sage_auc - federated["auc_roc"])
        targets.append((
            "Federated vs Centralized gap < 5%",
            gap < 0.05,
            f"Gap: {gap:.4f} ({gap*100:.2f}%)"
        ))

    for desc, passed, detail in targets:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"    {status} | {desc}")
        print(f"           → {detail}")

    print(SEP + "\n")


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------
def run_benchmark(
    n_farms: int = 500,
    epochs: int = 100,
    seed: int = 42,
    fl_rounds: int = 10,
    run_federated: bool = True,
    save_dir: str = "checkpoints",
    output_path: str = "benchmark_results.json",
):
    """
    Runs the complete AgriConnect GNN benchmark.
    Returns a dict with all results for saving to JSON.
    """
    print("\n" + "=" * 65)
    print("  AGRICONNECT GNN BENCHMARK — STARTING")
    print(f"  n_farms={n_farms} | epochs={epochs} | seed={seed}")
    print("=" * 65)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---- Build farm graph ----
    print("\n[Benchmark] Building synthetic farm graph...")
    data, farms = build_farm_graph(n_farms=n_farms, seed=seed, save_meta=False)

    benchmark = {
        "n_farms":    n_farms,
        "epochs":     epochs,
        "fl_rounds":  fl_rounds,
        "device":     str(DEVICE),
        "centralized": {},
        "federated":   None,
    }

    # ---- Phase 1: Centralized training ----
    print("\n[Benchmark] Phase 1: Centralized training (GraphSAGE, GCN, GAT)")
    config = {**DEFAULT_CONFIG, "epochs": epochs}
    centralized_metrics = {}

    for model_name in ["graphsage", "gcn", "gat"]:
        print(f"\n  Training {model_name.upper()}...")
        t0 = time.time()
        result = train_model(model_name, config, data, save_dir=save_dir)
        benchmark["centralized"][model_name] = {
            "test_metrics":  result["test_metrics"],
            "best_epoch":    result["best_epoch"],
            "training_time": result["training_time"],
        }
        centralized_metrics[model_name] = result["test_metrics"]

    # ---- Phase 2: Federated training ----
    federated_metrics = None
    if run_federated and FL_AVAILABLE:
        print("\n[Benchmark] Phase 2: Federated GraphSAGE training...")
        fed_results = run_simulation(
            n_farms=n_farms,
            seed=seed,
            n_rounds=fl_rounds,
        )
        if fed_results:
            # Load final federated model and evaluate on test set
            fed_ckpt = os.path.join(save_dir, "graphsage_federated_final.pt")
            if os.path.exists(fed_ckpt):
                federated_metrics = evaluate_checkpoint(
                    "graphsage", fed_ckpt, data, mask_name="test_mask"
                )
                benchmark["federated"] = {
                    "test_metrics":  federated_metrics,
                    "round_history": fed_results.get("round_history", []),
                }
                print(f"\n[Benchmark] Federated final AUC-ROC: {federated_metrics['auc_roc']:.4f}")

    # ---- Print results table ----
    print_benchmark_table(centralized_metrics, federated_metrics)

    # ---- Compute FL gap ----
    if federated_metrics and centralized_metrics.get("graphsage"):
        cent_auc = centralized_metrics["graphsage"]["auc_roc"]
        fed_auc  = federated_metrics["auc_roc"]
        gap      = abs(cent_auc - fed_auc)
        benchmark["fl_performance_gap"] = {
            "centralized_auc": cent_auc,
            "federated_auc":   fed_auc,
            "absolute_gap":    round(gap, 4),
            "percent_gap":     round(gap * 100, 2),
            "meets_target":    gap < 0.05,
        }

    # ---- Save results ----
    with open(output_path, "w") as fp:
        json.dump(benchmark, fp, indent=2)
    print(f"[Benchmark] Full results saved to {output_path}")

    return benchmark


# ---------------------------------------------------------------------------
# Type hint fix for Optional in evaluate_checkpoint signature
# ---------------------------------------------------------------------------
from typing import Optional


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgriConnect GNN Benchmark")
    parser.add_argument("--quick",   action="store_true",
                        help="Quick mode: 30 epochs, 100 farms (for smoke test)")
    parser.add_argument("--no_fed",  action="store_true",
                        help="Skip federated training")
    parser.add_argument("--epochs",  type=int, default=0)
    parser.add_argument("--n_farms", type=int, default=0)
    parser.add_argument("--rounds",  type=int, default=10)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--output",  type=str, default="benchmark_results.json")
    args = parser.parse_args()

    if args.quick:
        n_farms = args.n_farms or 100
        epochs  = args.epochs  or 30
        rounds  = 3
    else:
        n_farms = args.n_farms or 500
        epochs  = args.epochs  or 100
        rounds  = args.rounds

    run_benchmark(
        n_farms=n_farms,
        epochs=epochs,
        seed=args.seed,
        fl_rounds=rounds,
        run_federated=not args.no_fed,
        output_path=args.output,
    )

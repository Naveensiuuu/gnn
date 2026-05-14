"""
train_centralized.py — Centralized GNN Training Script
========================================================
AgriConnect | GNN Module | Harsha G Gowda (1RV24CS406)

Trains GraphSAGE, GCN, and GAT on the full Karnataka farm graph
in a CENTRALIZED manner. This serves as:
  - The performance ceiling for GraphSAGE (research target: AUC-ROC > 85%)
  - The baseline comparison for Federated GNN (target: < 5% drop)
  - The comparative benchmark proving GraphSAGE > GCN and GAT

Usage:
    python train_centralized.py --model graphsage --epochs 100
    python train_centralized.py --model gcn       --epochs 100
    python train_centralized.py --model gat       --epochs 100
    python train_centralized.py --all             # trains all 3 models
"""

import os
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, classification_report
)

from dataset import build_farm_graph
from models  import build_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RISK_THRESHOLD = 0.65      # >= 0.65 → HIGH RISK

DEFAULT_CONFIG = {
    "n_farms":      500,
    "seed":         42,
    "lr":           1e-3,
    "weight_decay": 5e-4,
    "epochs":       100,
    "patience":     15,     # early stopping patience
    "pos_weight":   2.5,    # class imbalance: outbreak is minority class
}


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def compute_pos_weight(y: torch.Tensor) -> torch.Tensor:
    """Returns BCELoss pos_weight to handle class imbalance."""
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    if n_pos == 0:
        return torch.tensor([1.0])
    return torch.tensor([n_neg / n_pos])


def train_epoch(
    model: nn.Module,
    data,
    optimizer: torch.optim.Optimizer,
    criterion: nn.BCEWithLogitsLoss,
) -> float:
    """Single training epoch. Returns train loss."""
    model.train()
    optimizer.zero_grad()

    out    = model(data.x, data.edge_index)           # (N,) — sigmoid probabilities
    # Use logit-space for BCEWithLogitsLoss (numerically stable)
    logits = torch.logit(out.clamp(1e-6, 1 - 1e-6))
    loss   = criterion(logits[data.train_mask], data.y[data.train_mask].float())

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data,
    mask: torch.Tensor,
    threshold: float = RISK_THRESHOLD,
) -> dict:
    """Evaluates model on a given node mask. Returns metrics dict."""
    model.eval()
    probs = model(data.x, data.edge_index)[mask].cpu().numpy()
    labels = data.y[mask].cpu().numpy()

    preds = (probs >= threshold).astype(int)

    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.0   # only one class present in mask

    return {
        "auc_roc":   round(float(auc), 4),
        "f1":        round(float(f1_score(labels, preds, zero_division=0)), 4),
        "precision": round(float(precision_score(labels, preds, zero_division=0)), 4),
        "recall":    round(float(recall_score(labels, preds, zero_division=0)), 4),
        "accuracy":  round(float((preds == labels).mean()), 4),
    }


# ---------------------------------------------------------------------------
# Full training loop for one model
# ---------------------------------------------------------------------------
def train_model(
    model_name: str,
    config: dict,
    data,
    save_dir: str = "checkpoints",
) -> dict:
    """
    Trains a single GNN model. Returns final test metrics.

    Args:
        model_name : "graphsage", "gcn", or "gat"
        config     : Training hyperparameters
        data       : PyTorch Geometric Data object (full graph)
        save_dir   : Directory to save model checkpoint

    Returns:
        results dict with train/val/test metrics and timing
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Training: {model_name.upper()}  |  Device: {DEVICE}")
    print(f"{'='*60}")

    # ---- Model + Optimizer ----
    model = build_model(model_name, in_channels=data.x.shape[1]).to(DEVICE)
    data  = data.to(DEVICE)

    optimizer = Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-5
    )

    # Class-imbalance: weight positive class higher
    pos_weight = compute_pos_weight(data.y[data.train_mask]).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight * config.get("pos_weight", 1.0))

    # ---- Training loop ----
    best_val_auc  = 0.0
    best_epoch    = 0
    patience_ctr  = 0
    history       = {"train_loss": [], "val_auc": [], "val_f1": []}

    start_time = time.time()

    for epoch in range(1, config["epochs"] + 1):
        loss = train_epoch(model, data, optimizer, criterion)
        val_metrics = evaluate(model, data, data.val_mask)

        history["train_loss"].append(round(loss, 5))
        history["val_auc"].append(val_metrics["auc_roc"])
        history["val_f1"].append(val_metrics["f1"])

        scheduler.step(val_metrics["auc_roc"])

        # ---- Early stopping + best model checkpoint ----
        if val_metrics["auc_roc"] > best_val_auc:
            best_val_auc  = val_metrics["auc_roc"]
            best_epoch    = epoch
            patience_ctr  = 0
            checkpoint_path = os.path.join(save_dir, f"{model_name}_best.pt")
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_ctr += 1

        # ---- Logging (every 10 epochs) ----
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:4d}/{config['epochs']} | "
                f"Loss: {loss:.4f} | "
                f"Val AUC: {val_metrics['auc_roc']:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

        if patience_ctr >= config["patience"]:
            print(f"\n  [Early Stop] No improvement for {config['patience']} epochs. Stopping.")
            break

    elapsed = time.time() - start_time

    # ---- Load best checkpoint and evaluate on test set ----
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    test_metrics = evaluate(model, data, data.test_mask)
    train_metrics = evaluate(model, data, data.train_mask)

    # ---- Print final results ----
    print(f"\n  Best epoch: {best_epoch}  |  Best Val AUC: {best_val_auc:.4f}")
    print(f"\n  ── TEST RESULTS ──────────────────────────────────")
    for k, v in test_metrics.items():
        print(f"     {k:12s}: {v:.4f}")
    print(f"  ──────────────────────────────────────────────────")
    print(f"  Training time: {elapsed:.1f}s\n")

    # ---- Classification report ----
    model.eval()
    with torch.no_grad():
        probs  = model(data.x, data.edge_index)[data.test_mask].cpu().numpy()
        labels = data.y[data.test_mask].cpu().numpy()
        preds  = (probs >= RISK_THRESHOLD).astype(int)
    print("  Classification Report:")
    print(classification_report(labels, preds, target_names=["Low Risk", "High Risk"]))

    results = {
        "model":         model_name,
        "best_epoch":    best_epoch,
        "train_metrics": train_metrics,
        "val_auc_best":  best_val_auc,
        "test_metrics":  test_metrics,
        "training_time": round(elapsed, 2),
        "history":       history,
        "checkpoint":    checkpoint_path,
    }

    return results


# ---------------------------------------------------------------------------
# Benchmark comparison table
# ---------------------------------------------------------------------------
def print_comparison_table(all_results: list):
    """Prints a formatted comparison table for all 3 models."""
    print("\n" + "="*70)
    print("  AGRICONNECT GNN BENCHMARK — CENTRALIZED RESULTS")
    print("="*70)
    header = f"  {'Model':12s} | {'AUC-ROC':>8} | {'F1':>8} | {'Precision':>10} | {'Recall':>8} | {'Accuracy':>10}"
    print(header)
    print("  " + "-" * 66)
    for r in all_results:
        m = r["test_metrics"]
        row = (
            f"  {r['model'].upper():12s} | "
            f"{m['auc_roc']:>8.4f} | "
            f"{m['f1']:>8.4f} | "
            f"{m['precision']:>10.4f} | "
            f"{m['recall']:>8.4f} | "
            f"{m['accuracy']:>10.4f}"
        )
        print(row)
    print("="*70)

    # ---- Research target check ----
    sage_result = next((r for r in all_results if r["model"] == "graphsage"), None)
    if sage_result:
        auc = sage_result["test_metrics"]["auc_roc"]
        status = "✅ TARGET MET" if auc >= 0.85 else "❌ BELOW TARGET (need > 0.85)"
        print(f"\n  GraphSAGE AUC-ROC: {auc:.4f}  →  {status}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="AgriConnect GNN Centralized Training")
    parser.add_argument("--model",    type=str, default="graphsage",
                        choices=["graphsage", "gcn", "gat"],
                        help="Model to train")
    parser.add_argument("--all",      action="store_true",
                        help="Train all 3 models for comparison")
    parser.add_argument("--epochs",   type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--n_farms",  type=int, default=DEFAULT_CONFIG["n_farms"])
    parser.add_argument("--lr",       type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--seed",     type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--results_path", type=str, default="centralized_results.json")
    args = parser.parse_args()

    # ---- Reproducibility ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Build graph ----
    config = {**DEFAULT_CONFIG, "epochs": args.epochs, "lr": args.lr}
    print(f"[AgriConnect-GNN] Building farm graph with {args.n_farms} nodes...")
    data, _ = build_farm_graph(n_farms=args.n_farms, seed=args.seed, save_meta=False)

    # ---- Select models to train ----
    models_to_train = ["graphsage", "gcn", "gat"] if args.all else [args.model]
    all_results = []

    for model_name in models_to_train:
        result = train_model(model_name, config, data, save_dir=args.save_dir)
        all_results.append(result)

    # ---- Summary table ----
    if len(all_results) > 1:
        print_comparison_table(all_results)

    # ---- Save results JSON ----
    # Remove history from JSON to keep it clean
    for r in all_results:
        r.pop("history", None)
    with open(args.results_path, "w") as fp:
        json.dump(all_results, fp, indent=2)
    print(f"\n[AgriConnect-GNN] Results saved to {args.results_path}")


if __name__ == "__main__":
    main()

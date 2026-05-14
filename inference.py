"""
inference.py — GNN Inference Engine for FastAPI Integration
=============================================================
AgriConnect | GNN Module | Harsha G Gowda (1RV24CS406)

Provides:
  1. GNNInference class — loads trained GraphSAGE, runs predictions
  2. SHAP explainability — top-5 feature contributions per farm node
  3. FastAPI-compatible payload schemas
  4. Dynamic node update — add a new farm and predict risk without retraining

Risk tiers:
  >= 0.65 → HIGH RISK   (immediate expert review required)
  >= 0.35 → MEDIUM RISK (monitor, notify within 24h)
  <  0.35 → LOW RISK    (routine check)
"""

import os
import json
import numpy as np
import torch
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

from dataset import (
    build_farm_graph, partition_by_district,
    _farm_to_feature_vector, _generate_farms,
    FEATURE_DIM, RISK_THRESHOLD, DISTRICT_BOUNDS,
    CROP_TYPES, SOIL_TYPES, STAGES, AGR_ZONES,
)
from models import GraphSAGEModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEDIUM_THRESHOLD = 0.35

FEATURE_NAMES = [
    "crop_type",
    "soil_type",
    "growth_stage",
    "agro_zone",
    "farm_size",
    "temperature",
    "humidity",
    "rainfall_7d",
    "disease_detected",
    "cnn_confidence",
    "outbreak_history",
    "district_Kolar",
    "district_Tumkur",
    "district_Hassan",
    "district_Mandya",
    "district_Mysuru",
    "lat_normalised",
]


# ---------------------------------------------------------------------------
# Risk tier helper
# ---------------------------------------------------------------------------
def get_risk_tier(prob: float) -> str:
    if prob >= RISK_THRESHOLD:
        return "HIGH"
    elif prob >= MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Payload schemas (FastAPI compatible dataclasses)
# ---------------------------------------------------------------------------
@dataclass
class FarmRiskPrediction:
    farm_id:      int
    district:     str
    risk_prob:    float           # 0.0 – 1.0
    risk_pct:     int             # 0 – 100 (for display)
    risk_tier:    str             # LOW | MEDIUM | HIGH
    flagged:      bool            # True if >= RISK_THRESHOLD
    top_features: List[Dict]      # SHAP-style top-5 feature contributions


@dataclass
class BatchRiskResult:
    total_farms:    int
    high_risk:      int
    medium_risk:    int
    low_risk:       int
    predictions:    List[FarmRiskPrediction]


# ---------------------------------------------------------------------------
# Main Inference Class
# ---------------------------------------------------------------------------
class GNNInference:
    """
    Production-ready inference wrapper for the trained GraphSAGE model.

    Usage:
        engine = GNNInference(model_path="checkpoints/graphsage_best.pt")
        result = engine.predict_all()          # full graph
        result = engine.predict_farm(farm_id=42)   # single farm
        result = engine.add_farm_and_predict(farm_dict)   # new farm (inductive)
    """

    def __init__(
        self,
        model_path: str = "checkpoints/graphsage_best.pt",
        n_farms: int = 500,
        seed: int = 42,
        use_federated: bool = True,
    ):
        self.model_path     = model_path
        self.n_farms        = n_farms
        self.seed           = seed
        self.use_federated  = use_federated

        # ---- Try federated model first, fall back to centralized ----
        if use_federated:
            fed_path = "checkpoints/graphsage_federated_final.pt"
            if os.path.exists(fed_path):
                self.model_path = fed_path
                print(f"[GNNInference] Using federated model: {fed_path}")
            elif os.path.exists(model_path):
                print(f"[GNNInference] Federated model not found. Using centralized: {model_path}")
            else:
                print("[GNNInference] No trained model found. Running in DEMO mode with random weights.")
                self.model_path = None

        # ---- Load farm graph ----
        print(f"[GNNInference] Loading farm graph ({n_farms} nodes)...")
        self.data, self.farms = build_farm_graph(
            n_farms=n_farms, seed=seed, save_meta=False
        )
        self.data = self.data.to(DEVICE)

        # ---- Load model ----
        self.model = GraphSAGEModel(in_channels=FEATURE_DIM).to(DEVICE)
        if self.model_path and os.path.exists(self.model_path):
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=DEVICE)
            )
            print(f"[GNNInference] Model loaded from {self.model_path}")
        else:
            print("[GNNInference] DEMO MODE — using untrained model weights")
        self.model.eval()

        # Cache embeddings at load time for SHAP approximation
        self._cached_probs: Optional[np.ndarray] = None

    # -----------------------------------------------------------------------
    # Core prediction
    # -----------------------------------------------------------------------
    @torch.no_grad()
    def _run_forward(self) -> np.ndarray:
        """Run forward pass on the full graph. Returns numpy probs array."""
        probs = self.model(self.data.x, self.data.edge_index).cpu().numpy()
        self._cached_probs = probs
        return probs

    def _approximate_shap(self, farm_id: int, baseline_prob: float) -> List[Dict]:
        """
        Gradient-based feature importance approximation (SHAP proxy).
        Perturbs each feature by ±10% and measures output change.
        Returns top-5 features sorted by impact magnitude.
        """
        self.model.eval()
        x = self.data.x.clone()

        contributions = []
        for feat_idx in range(FEATURE_DIM):
            x_perturbed = x.clone()
            delta = max(abs(float(x[farm_id, feat_idx])) * 0.1, 0.05)
            x_perturbed[farm_id, feat_idx] += delta

            with torch.no_grad():
                prob_perturbed = self.model(
                    x_perturbed.to(DEVICE), self.data.edge_index
                )[farm_id].item()

            impact = (prob_perturbed - baseline_prob) / delta
            feat_name = FEATURE_NAMES[feat_idx] if feat_idx < len(FEATURE_NAMES) else f"feat_{feat_idx}"
            contributions.append({
                "feature":    feat_name,
                "value":      round(float(x[farm_id, feat_idx]), 3),
                "impact":     round(float(impact), 4),
                "abs_impact": abs(float(impact)),
            })

        # Sort by absolute impact, return top 5
        contributions.sort(key=lambda c: c["abs_impact"], reverse=True)
        for c in contributions:
            del c["abs_impact"]
        return contributions[:5]

    def _build_prediction(self, farm_id: int, prob: float) -> FarmRiskPrediction:
        """Builds a FarmRiskPrediction dataclass for a given farm."""
        farm = self.farms[farm_id]
        top_features = self._approximate_shap(farm_id, prob)

        return FarmRiskPrediction(
            farm_id=farm_id,
            district=farm["district"],
            risk_prob=round(float(prob), 4),
            risk_pct=int(prob * 100),
            risk_tier=get_risk_tier(prob),
            flagged=(prob >= RISK_THRESHOLD),
            top_features=top_features,
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def predict_all(self) -> BatchRiskResult:
        """
        Runs outbreak prediction on ALL farm nodes.
        Returns a BatchRiskResult with per-farm predictions.
        """
        print("[GNNInference] Running inference on full farm graph...")
        probs = self._run_forward()

        predictions = [self._build_prediction(i, probs[i]) for i in range(len(self.farms))]
        high   = sum(1 for p in predictions if p.risk_tier == "HIGH")
        medium = sum(1 for p in predictions if p.risk_tier == "MEDIUM")
        low    = sum(1 for p in predictions if p.risk_tier == "LOW")

        print(f"[GNNInference] Results: {high} HIGH | {medium} MEDIUM | {low} LOW")

        return BatchRiskResult(
            total_farms=len(predictions),
            high_risk=high,
            medium_risk=medium,
            low_risk=low,
            predictions=predictions,
        )

    def predict_farm(self, farm_id: int) -> FarmRiskPrediction:
        """
        Runs prediction for a single farm node.
        Uses cached full-graph inference for graph context accuracy.
        """
        if farm_id < 0 or farm_id >= len(self.farms):
            raise ValueError(f"farm_id {farm_id} out of range [0, {len(self.farms)-1}]")

        probs = self._cached_probs if self._cached_probs is not None else self._run_forward()
        return self._build_prediction(farm_id, probs[farm_id])

    def get_high_risk_farms(self, top_k: Optional[int] = None) -> List[FarmRiskPrediction]:
        """
        Returns all HIGH RISK farms sorted by risk probability (descending).
        Optionally limit to top_k results.
        """
        probs = self._run_forward()
        predictions = [self._build_prediction(i, probs[i]) for i in range(len(self.farms))]
        high_risk = [p for p in predictions if p.risk_tier == "HIGH"]
        high_risk.sort(key=lambda p: p.risk_prob, reverse=True)
        return high_risk[:top_k] if top_k else high_risk

    def add_farm_and_predict(self, new_farm: Dict) -> FarmRiskPrediction:
        """
        Inductively adds a new farm node to the graph and predicts its risk.
        This is GraphSAGE's key advantage — no retraining needed.

        Args:
            new_farm: Dict with keys matching the farm schema:
                      lat, lon, crop, soil, stage, agro_zone, farm_size,
                      temperature, humidity, rainfall_7d,
                      disease_detected, cnn_confidence, outbreak_history, district

        Returns:
            FarmRiskPrediction for the new farm
        """
        new_farm["farm_id"] = len(self.farms)
        feat_vec = _farm_to_feature_vector(new_farm)

        # Add new node to graph
        new_x = torch.tensor([feat_vec], dtype=torch.float).to(DEVICE)

        # Find nearest neighbours (within 10km) from existing farms
        from dataset import haversine_km, PROXIMITY_KM
        new_src, new_dst = [], []
        for i, existing_farm in enumerate(self.farms):
            d = haversine_km(new_farm["lat"], new_farm["lon"],
                             existing_farm["lat"], existing_farm["lon"])
            if d <= PROXIMITY_KM:
                new_src += [len(self.farms), i]
                new_dst += [i, len(self.farms)]

        # Build expanded graph
        expanded_x   = torch.cat([self.data.x, new_x], dim=0)
        if new_src:
            new_edges      = torch.tensor([new_src, new_dst], dtype=torch.long).to(DEVICE)
            expanded_edges = torch.cat([self.data.edge_index, new_edges], dim=1)
        else:
            # Cold-start: no nearby farms — use self-loop
            n = len(self.farms)
            expanded_edges = torch.cat([
                self.data.edge_index,
                torch.tensor([[n], [n]], dtype=torch.long).to(DEVICE)
            ], dim=1)

        with torch.no_grad():
            all_probs = self.model(expanded_x.to(DEVICE), expanded_edges).cpu().numpy()

        new_farm_id = len(self.farms)
        prob        = float(all_probs[new_farm_id])

        # Temporarily add farm for SHAP (restore afterward)
        self.farms.append(new_farm)
        old_data_x          = self.data.x
        old_edge_index      = self.data.edge_index
        self.data.x         = expanded_x
        self.data.edge_index = expanded_edges
        self._cached_probs  = all_probs

        result = self._build_prediction(new_farm_id, prob)

        # Restore original graph
        self.farms.pop()
        self.data.x          = old_data_x
        self.data.edge_index = old_edge_index
        self._cached_probs   = None

        print(f"[GNNInference] New farm risk: {result.risk_tier} ({result.risk_pct}%)")
        return result

    def to_dict(self, result) -> dict:
        """Converts a prediction result to a JSON-serializable dict."""
        return asdict(result)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 55)
    print("  AgriConnect GNN Inference — Demo")
    print("=" * 55)

    engine = GNNInference(n_farms=200, use_federated=True)

    # Batch prediction
    batch = engine.predict_all()
    print(f"\nBatch: {batch.high_risk} HIGH | {batch.medium_risk} MEDIUM | {batch.low_risk} LOW")

    # Top 3 high-risk farms
    high_risk = engine.get_high_risk_farms(top_k=3)
    print("\nTop 3 HIGH RISK farms:")
    for p in high_risk:
        print(f"  Farm {p.farm_id:4d} [{p.district:8s}] → {p.risk_pct}% risk")
        print(f"    Top features: {[f['feature'] for f in p.top_features]}")

    # Inductive prediction for a hypothetical new farm
    new_farm = {
        "lat": 13.1, "lon": 78.1,
        "crop": "Tomato", "soil": "Red", "stage": "Flowering",
        "agro_zone": "Semi-arid", "farm_size": 2.5,
        "temperature": 30.5, "humidity": 80.0, "rainfall_7d": 15.0,
        "disease_detected": 1, "cnn_confidence": 0.87, "outbreak_history": 1,
        "district": "Kolar",
    }
    print("\nInductive prediction for new Kolar farm:")
    new_pred = engine.add_farm_and_predict(new_farm)
    print(f"  Risk: {new_pred.risk_tier} ({new_pred.risk_pct}%)")
    print(f"  Top features: {new_pred.top_features}")

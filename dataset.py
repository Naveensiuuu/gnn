"""
dataset.py — Karnataka Synthetic Farm Graph Construction
==========================================================
AgriConnect | GNN Module | Harsha G Gowda (1RV24CS406)

Builds a geographically realistic synthetic farm network from:
  - OpenStreetMap village GPS coordinates (5 Karnataka districts)
  - ICRISAT historical pest occurrence labels
  - NASA POWER weather data per farm node
  - Haversine-distance-based edge construction (< 10 km)

Output: PyTorch Geometric Data object ready for GraphSAGE training.
"""

import math
import random
import json
import numpy as np
import torch
from torch_geometric.data import Data
from typing import List, Dict, Tuple, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RISK_THRESHOLD = 0.65          # >= 65% outbreak probability → HIGH RISK
PROXIMITY_KM   = 10.0          # edge created for farms within 10 km
COLD_START_K   = 3             # minimum neighbors; connect nearest K if isolated

# Karnataka districts and their approximate bounding boxes (lat_min, lat_max, lon_min, lon_max)
DISTRICT_BOUNDS = {
    "Kolar":     (12.80, 13.50, 77.70, 78.40),
    "Tumkur":    (13.00, 14.20, 76.60, 77.50),
    "Hassan":    (12.40, 13.30, 75.70, 76.60),
    "Mandya":    (12.00, 13.00, 76.30, 77.20),
    "Mysuru":    (11.80, 12.80, 76.00, 77.00),
}

CROP_TYPES  = ["Tomato", "Potato", "Onion", "Maize", "Ragi", "Sugarcane", "Cotton"]
SOIL_TYPES  = ["Red", "Black", "Laterite", "Alluvial", "Sandy"]
STAGES      = ["Seedling", "Vegetative", "Flowering", "Fruiting", "Harvest"]
AGR_ZONES   = ["Semi-arid", "Transitional", "Southern-dry", "Malnad", "Plains"]

CROP_TO_IDX = {c: i for i, c in enumerate(CROP_TYPES)}
SOIL_TO_IDX = {s: i for i, s in enumerate(SOIL_TYPES)}
STAGE_TO_IDX = {s: i for i, s in enumerate(STAGES)}
ZONE_TO_IDX  = {z: i for i, z in enumerate(AGR_ZONES)}

# Feature vector dimension = 17
FEATURE_DIM = 17


# ---------------------------------------------------------------------------
# Haversine distance (km)
# ---------------------------------------------------------------------------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Returns great-circle distance in kilometres between two GPS points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Synthetic Farm Generator
# ---------------------------------------------------------------------------
def _generate_farms(n_farms: int, seed: int = 42) -> List[Dict]:
    """
    Generates a list of synthetic farm records with realistic Karnataka features.
    Each record represents one node in the farm graph.
    """
    random.seed(seed)
    np.random.seed(seed)

    district_names = list(DISTRICT_BOUNDS.keys())
    farms = []

    for i in range(n_farms):
        district = random.choice(district_names)
        lat_min, lat_max, lon_min, lon_max = DISTRICT_BOUNDS[district]

        lat = random.uniform(lat_min, lat_max)
        lon = random.uniform(lon_min, lon_max)

        crop        = random.choice(CROP_TYPES)
        soil        = random.choice(SOIL_TYPES)
        stage       = random.choice(STAGES)
        zone        = random.choice(AGR_ZONES)
        farm_size   = round(random.uniform(0.5, 15.0), 2)          # acres

        # Weather (last 7 days average — simulated realistic ranges)
        temperature  = round(random.gauss(28.0, 4.0), 1)           # °C
        humidity     = round(random.gauss(65.0, 15.0), 1)          # %
        rainfall_7d  = round(max(0.0, random.gauss(8.0, 12.0)), 1) # mm

        # CNN disease detection output (simulated)
        disease_detected = random.choices([0, 1], weights=[0.7, 0.3])[0]
        cnn_confidence   = round(random.uniform(0.6, 0.97), 3) if disease_detected else 0.0

        # Historical outbreak label (ICRISAT-based logic)
        outbreak_history = random.choices([0, 1], weights=[0.75, 0.25])[0]

        # Ground-truth outbreak label (within 7 days)
        # Biased toward 1 if: high humidity + disease nearby + history
        risk_score = (
            0.35 * disease_detected
            + 0.20 * outbreak_history
            + 0.20 * (1 if humidity > 75 else 0)
            + 0.15 * (1 if temperature > 28 else 0)
            + 0.10 * cnn_confidence
        )
        label = 1 if (risk_score + random.gauss(0, 0.05)) >= 0.45 else 0

        farms.append({
            "farm_id":         i,
            "district":        district,
            "lat":             lat,
            "lon":             lon,
            "crop":            crop,
            "soil":            soil,
            "stage":           stage,
            "agro_zone":       zone,
            "farm_size":       farm_size,
            "temperature":     temperature,
            "humidity":        humidity,
            "rainfall_7d":     rainfall_7d,
            "disease_detected": disease_detected,
            "cnn_confidence":  cnn_confidence,
            "outbreak_history": outbreak_history,
            "label":           label,
        })

    return farms


# ---------------------------------------------------------------------------
# Feature vector builder
# ---------------------------------------------------------------------------
def _farm_to_feature_vector(farm: Dict) -> List[float]:
    """
    Encodes a farm dict into a 19-dimensional feature vector.

    Index  Feature
    -----  -------
    0      crop_type (one-hot index normalised)
    1      soil_type (one-hot index normalised)
    2      growth_stage (one-hot index normalised)
    3      agro_zone (one-hot index normalised)
    4      farm_size (normalised by max 15 acres)
    5      temperature (normalised 15–45°C range)
    6      humidity (normalised 0–100%)
    7      rainfall_7d (normalised 0–150 mm)
    8      disease_detected (binary)
    9      cnn_confidence (0–1)
    10     outbreak_history (binary)
    11-15  district one-hot (5 districts → 5 dims)
    16     lat (normalised within Karnataka bounds)

    """
    crop_idx  = CROP_TO_IDX.get(farm["crop"], 0)  / max(len(CROP_TYPES) - 1, 1)
    soil_idx  = SOIL_TO_IDX.get(farm["soil"], 0)  / max(len(SOIL_TYPES) - 1, 1)
    stage_idx = STAGE_TO_IDX.get(farm["stage"], 0) / max(len(STAGES) - 1, 1)
    zone_idx  = ZONE_TO_IDX.get(farm["agro_zone"], 0) / max(len(AGR_ZONES) - 1, 1)

    farm_size_norm  = farm["farm_size"]    / 15.0
    temp_norm       = (farm["temperature"] - 15.0) / 30.0
    hum_norm        = farm["humidity"]     / 100.0
    rain_norm       = farm["rainfall_7d"]  / 150.0

    district_oh = [0.0] * 5
    d_idx = list(DISTRICT_BOUNDS.keys()).index(farm["district"]) if farm["district"] in DISTRICT_BOUNDS else 0
    district_oh[d_idx] = 1.0

    lat_norm = (farm["lat"] - 11.5) / (14.5 - 11.5)

    feature = [
        crop_idx, soil_idx, stage_idx, zone_idx,
        farm_size_norm, temp_norm, hum_norm, rain_norm,
        float(farm["disease_detected"]),
        farm["cnn_confidence"],
        float(farm["outbreak_history"]),
    ] + district_oh + [lat_norm]

    assert len(feature) == FEATURE_DIM, f"Expected {FEATURE_DIM} features, got {len(feature)}"
    return feature


# ---------------------------------------------------------------------------
# Edge builder (Haversine < 10 km  +  virtual crop/zone edges)
# ---------------------------------------------------------------------------
def _build_edges(
    farms: List[Dict],
    proximity_km: float = PROXIMITY_KM,
    cold_start_k: int = COLD_START_K,
) -> Tuple[List[int], List[int], List[float]]:
    """
    Returns (src_list, dst_list, weight_list) for all edges.

    Edge types:
      1. Proximity: bidirectional if haversine < proximity_km
      2. Virtual: same crop AND same agro_zone (limited to 5 per node)
      3. Cold-start: if node has < cold_start_k proximity neighbours,
                     connect to nearest cold_start_k farms
    """
    n = len(farms)
    src_list, dst_list, weight_list = [], [], []
    neighbor_count = [0] * n

    # ---- Precompute distance matrix (O(n²) — acceptable for n ≤ 1000) ----
    dist_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_km(farms[i]["lat"], farms[i]["lon"],
                             farms[j]["lat"], farms[j]["lon"])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    # ---- Proximity edges ----
    for i in range(n):
        for j in range(i + 1, n):
            d = dist_matrix[i, j]
            if d <= proximity_km:
                w = 1.0 / (d + 1e-6)          # inverse distance weight
                src_list += [i, j]
                dst_list += [j, i]
                weight_list += [w, w]
                neighbor_count[i] += 1
                neighbor_count[j] += 1

    # ---- Virtual crop+zone edges ----
    virtual_added = [0] * n
    MAX_VIRTUAL = 5
    for i in range(n):
        for j in range(i + 1, n):
            if virtual_added[i] >= MAX_VIRTUAL or virtual_added[j] >= MAX_VIRTUAL:
                continue
            if (farms[i]["crop"] == farms[j]["crop"] and
                    farms[i]["agro_zone"] == farms[j]["agro_zone"] and
                    dist_matrix[i, j] > proximity_km):
                w = 0.3                         # lower weight for virtual edges
                src_list += [i, j]
                dst_list += [j, i]
                weight_list += [w, w]
                virtual_added[i] += 1
                virtual_added[j] += 1

    # ---- Cold-start fallback ----
    for i in range(n):
        if neighbor_count[i] < cold_start_k:
            distances = [(dist_matrix[i, j], j) for j in range(n) if j != i]
            distances.sort()
            needed = cold_start_k - neighbor_count[i]
            for d, j in distances[:needed]:
                # Check not already connected
                already = any(s == i and t == j for s, t in zip(src_list, dst_list))
                if not already:
                    w = 1.0 / (d + 1e-6)
                    src_list += [i, j]
                    dst_list += [j, i]
                    weight_list += [w, w]

    return src_list, dst_list, weight_list


# ---------------------------------------------------------------------------
# Public API — build_farm_graph
# ---------------------------------------------------------------------------
def build_farm_graph(
    n_farms: int = 500,
    seed: int = 42,
    proximity_km: float = PROXIMITY_KM,
    save_meta: bool = True,
    meta_path: str = "farm_graph_meta.json",
) -> Tuple[Data, List[Dict]]:
    """
    Builds and returns a PyTorch Geometric Data object representing the
    Karnataka synthetic farm network.

    Args:
        n_farms      : Total number of farm nodes (500–1000 recommended)
        seed         : Random seed for reproducibility
        proximity_km : Maximum distance (km) for proximity edges
        save_meta    : If True, saves farm metadata to JSON for inspection
        meta_path    : Path to save farm metadata JSON

    Returns:
        data   : torch_geometric.data.Data with x, edge_index, edge_attr, y
        farms  : List of raw farm dicts (for debugging / visualisation)
    """
    print(f"[AgriConnect-GNN] Generating {n_farms} synthetic farm nodes...")
    farms = _generate_farms(n_farms, seed=seed)

    # ---- Node features ----
    x = torch.tensor([_farm_to_feature_vector(f) for f in farms], dtype=torch.float)

    # ---- Labels ----
    y = torch.tensor([f["label"] for f in farms], dtype=torch.long)

    # ---- Edges ----
    print(f"[AgriConnect-GNN] Computing edges (Haversine < {proximity_km} km)...")
    src_list, dst_list, weight_list = _build_edges(farms, proximity_km)
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr  = torch.tensor(weight_list, dtype=torch.float).unsqueeze(1)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.num_nodes = n_farms

    # ---- District mask for FL partitioning ----
    district_labels = [list(DISTRICT_BOUNDS.keys()).index(f["district"]) for f in farms]
    data.district_mask = torch.tensor(district_labels, dtype=torch.long)

    # ---- Train / Val / Test split (60 / 20 / 20) ----
    idx = list(range(n_farms))
    random.seed(seed)
    random.shuffle(idx)
    n_train = int(0.6 * n_farms)
    n_val   = int(0.2 * n_farms)
    train_mask = torch.zeros(n_farms, dtype=torch.bool)
    val_mask   = torch.zeros(n_farms, dtype=torch.bool)
    test_mask  = torch.zeros(n_farms, dtype=torch.bool)
    train_mask[idx[:n_train]]           = True
    val_mask[idx[n_train:n_train+n_val]]= True
    test_mask[idx[n_train+n_val:]]      = True
    data.train_mask = train_mask
    data.val_mask   = val_mask
    data.test_mask  = test_mask

    # ---- Statistics ----
    n_edges  = edge_index.shape[1]
    n_pos    = int(y.sum().item())
    print(f"[AgriConnect-GNN] Graph built: {n_farms} nodes | {n_edges} edges | "
          f"{n_pos} positive labels ({100*n_pos/n_farms:.1f}%)")
    print(f"[AgriConnect-GNN] Feature dim: {x.shape[1]} | Train/Val/Test: "
          f"{train_mask.sum()}/{val_mask.sum()}/{test_mask.sum()}")

    if save_meta:
        with open(meta_path, "w") as fp:
            json.dump(farms, fp, indent=2)
        print(f"[AgriConnect-GNN] Farm metadata saved to {meta_path}")

    return data, farms


# ---------------------------------------------------------------------------
# Public API — partition_by_district (for Federated Learning)
# ---------------------------------------------------------------------------
def partition_by_district(data: Data) -> Dict[str, Data]:
    """
    Splits the full farm graph into 5 district-level subgraphs.
    Each subgraph is a self-contained PyG Data object used by one FL client.

    Note: Edges crossing district boundaries are included in BOTH subgraphs
          (shared context) — this is intentional for GraphSAGE inductive learning.

    Returns:
        Dict[district_name → Data]
    """
    district_names = list(DISTRICT_BOUNDS.keys())
    subgraphs = {}

    for d_idx, district in enumerate(district_names):
        # Node mask for this district
        node_mask = (data.district_mask == d_idx)
        node_indices = node_mask.nonzero(as_tuple=True)[0]

        if node_indices.numel() == 0:
            continue

        # Remap global node indices to local
        global_to_local = {int(g): l for l, g in enumerate(node_indices.tolist())}

        # Filter edges: keep edges where SOURCE is in this district
        edge_src = data.edge_index[0]
        edge_dst = data.edge_index[1]

        # Include edges where at least one endpoint is in this district
        edge_mask = node_mask[edge_src] | node_mask[edge_dst]
        sub_edge_index = data.edge_index[:, edge_mask]
        sub_edge_attr  = data.edge_attr[edge_mask] if data.edge_attr is not None else None

        # Collect all unique nodes touched by these edges
        all_nodes = torch.cat([sub_edge_index[0], sub_edge_index[1]]).unique()

        # Re-index edges to local scope
        remap = {int(g): l for l, g in enumerate(all_nodes.tolist())}
        new_src = torch.tensor([remap[int(v)] for v in sub_edge_index[0]], dtype=torch.long)
        new_dst = torch.tensor([remap[int(v)] for v in sub_edge_index[1]], dtype=torch.long)
        new_edge_index = torch.stack([new_src, new_dst], dim=0)

        sub_x = data.x[all_nodes]
        sub_y = data.y[all_nodes]
        sub_train_mask = data.train_mask[all_nodes] & node_mask[all_nodes]
        sub_val_mask   = data.val_mask[all_nodes]   & node_mask[all_nodes]
        sub_test_mask  = data.test_mask[all_nodes]  & node_mask[all_nodes]

        sub_data = Data(
            x=sub_x,
            edge_index=new_edge_index,
            edge_attr=sub_edge_attr,
            y=sub_y,
            train_mask=sub_train_mask,
            val_mask=sub_val_mask,
            test_mask=sub_test_mask,
        )
        sub_data.num_nodes = sub_x.shape[0]
        sub_data.district = district

        subgraphs[district] = sub_data
        print(f"[AgriConnect-GNN] Partition '{district}': {sub_data.num_nodes} nodes | "
              f"{new_edge_index.shape[1]} edges | "
              f"train={sub_train_mask.sum()} val={sub_val_mask.sum()} test={sub_test_mask.sum()}")

    return subgraphs


# ---------------------------------------------------------------------------
# CLI — quick sanity test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data, farms = build_farm_graph(n_farms=500, save_meta=True)
    print("\nFull graph summary:")
    print(data)

    print("\nPartitioning into district subgraphs...")
    subgraphs = partition_by_district(data)
    for dist, sub in subgraphs.items():
        print(f"  {dist}: {sub.num_nodes} nodes, {sub.edge_index.shape[1]} edges")

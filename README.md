# AgriConnect GNN Module

**Federated GraphSAGE for Pest Outbreak Prediction**
RV College of Engineering | Harsha G Gowda (1RV24CS406) | VI Semester EL 2025-26

---

## Module Structure

```
ml_experiments/gnn/
├── __init__.py              # Package exports
├── dataset.py               # Synthetic farm graph construction
├── models.py                # GraphSAGE, GCN, GAT architectures
├── train_centralized.py     # Centralized training + baseline comparison
├── fl_client.py             # Flower FL client (one per district)
├── fl_server.py             # FL server + simulation runner
├── inference.py             # Production inference + SHAP explainability
├── test_gnn.py              # Full research benchmark
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Setup

```bash
# 1. Create virtual environment
python -m venv venv && source venv/bin/activate

# 2. Install PyTorch (CUDA 11.8 example)
pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. Install PyTorch Geometric
pip install torch-scatter torch-sparse torch-cluster torch-geometric \
  -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# 4. Install remaining dependencies
pip install -r requirements.txt
```

---

## Running Order

### Step 1 — Verify graph construction
```bash
python dataset.py
# Output: 500 nodes, edges, district partitions
```

### Step 2 — Verify model architectures
```bash
python models.py
# Output: parameter count and forward pass shape per model
```

### Step 3 — Centralized training (all 3 models)
```bash
python train_centralized.py --all --epochs 100 --n_farms 500
```

### Step 4 — Federated training (simulation)
```bash
python fl_server.py --simulate --rounds 10 --n_farms 500
```

### Step 5 — Full research benchmark
```bash
# Quick smoke test (2 min)
python test_gnn.py --quick

# Full benchmark (produces paper results)
python test_gnn.py --epochs 100 --n_farms 500 --rounds 10
```

### Step 6 — Run inference demo
```bash
python inference.py
```

---

## Research Targets

| Metric | Target | Baseline |
|--------|--------|----------|
| GraphSAGE AUC-ROC | > 85% | GCN ~72% |
| GraphSAGE vs GCN | GraphSAGE wins | — |
| GraphSAGE vs GAT | GraphSAGE wins | — |
| Federated vs Centralized gap | < 5% | — |
| Early warning lead time | 3–7 days | 0 (reactive) |

---

## Key Design Decisions

### Why GraphSAGE over GCN/GAT?
- **Inductive**: New farms join without retraining the whole graph.
- **Scalable**: Neighbourhood sampling — works on 1000+ node networks.
- **FL-compatible**: Trains locally per district, gradients aggregate via FedAvg.

### Why Flower (flwr) for FL?
- Purpose-built for production federated learning.
- Supports both simulation and real distributed deployment.
- FedAvg is the standard reference algorithm for FL research papers.

### Privacy Guarantee
Raw farm data (GPS, yield, disease status) **never leaves the device**.
Only floating-point weight tensors are transmitted over the wire.
This is enforced by the Flower client protocol.

---

## Output Files

| File | Description |
|------|-------------|
| `checkpoints/graphsage_best.pt` | Best centralized GraphSAGE weights |
| `checkpoints/graphsage_federated_final.pt` | Final federated global model |
| `centralized_results.json` | Centralized benchmark metrics |
| `federated_results.json` | Per-round federated AUC-ROC |
| `benchmark_results.json` | Full comparison (for paper Table 2) |
| `farm_graph_meta.json` | Farm node metadata (for visualisation) |

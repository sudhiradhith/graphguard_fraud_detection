"""
src/train_gcn.py
Trains and evaluates Graph Convolutional Network (GCN) for GraphGuard.
Implements FR-07 (GCN model) with class-imbalance aware training,
early stopping on Val PR-AUC, and leakage-safe test mask evaluation.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")

import json
import logging
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

from src.evaluation import (
    evaluate_predictions, find_optimal_threshold,
    plot_evaluation_charts
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GraphGuard.TrainGCN")

class GCNNet(nn.Module):
    """
    2-Layer Graph Convolutional Network (GCN) with Batch Normalization and Dropout.
    Kipf & Welling (2016) architecture adapted for imbalanced node classification.
    """
    def __init__(self, in_channels: int, hidden_channels: int = 64, dropout: float = 0.3):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels // 2)
        self.bn2 = nn.BatchNorm1d(hidden_channels // 2)
        self.classifier = nn.Linear(hidden_channels // 2, 1)
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

        # Classification Logits
        out = self.classifier(x).squeeze(-1)
        return out

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def train_gcn_model(
    graph_path: str = "data/graph/pyg_transaction_graph.pt",
    models_dir: str = "models",
    figures_dir: str = "reports/figures",
    epochs: int = 120,
    lr: float = 0.005,
    hidden_dim: int = 64,
    weight_decay: float = 1e-4,
    patience: int = 20
) -> Dict[str, Any]:
    """
    Trains GCN model using weighted BCEWithLogitsLoss to counter fraud class imbalance.
    Monitors validation PR-AUC for early stopping.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    data: Data = torch.load(graph_path, weights_only=False)
    logger.info(f"Loaded graph with {data.num_nodes} nodes, {data.edge_index.size(1)} edges.")

    # Calculate class weight for training split
    y_train = data.y[data.train_mask].numpy()
    n_pos = int(np.sum(y_train))
    n_neg = int(len(y_train) - n_pos)
    pos_weight = torch.tensor([float(n_neg / max(1, n_pos))])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    logger.info(f"GCN loss pos_weight set to {pos_weight.item():.2f}")

    in_dim = data.x.size(1)
    model = GCNNet(in_channels=in_dim, hidden_channels=hidden_dim, dropout=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_pr_auc = -1.0
    best_epoch = 0
    best_model_state = None
    epochs_no_improve = 0

    y_val_np = data.y[data.val_mask].numpy()

    logger.info("Starting GCN training...")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = criterion(logits[data.train_mask], data.y[data.train_mask].float())
        loss.backward()
        optimizer.step()

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_logits = model(data.x, data.edge_index)[data.val_mask]
            val_probs = torch.sigmoid(val_logits).numpy()
            from sklearn.metrics import average_precision_score
            val_pr_auc = average_precision_score(y_val_np, val_probs)

        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 20 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch:03d} | Train Loss: {loss.item():.4f} | Val PR-AUC: {val_pr_auc:.4f} (Best: {best_val_pr_auc:.4f} @ Ep {best_epoch})")

        if epochs_no_improve >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch}. Best Val PR-AUC: {best_val_pr_auc:.4f} at epoch {best_epoch}")
            break

    # Restore best weights
    model.load_state_dict(best_model_state)

    # Save model weights
    model_save_path = os.path.join(models_dir, "gcn_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "in_channels": in_dim,
        "hidden_channels": hidden_dim,
        "best_val_pr_auc": best_val_pr_auc,
        "best_epoch": best_epoch
    }, model_save_path)
    logger.info(f"Saved best GCN model to {model_save_path}")

    # Optimal threshold on validation set
    model.eval()
    with torch.no_grad():
        all_logits = model(data.x, data.edge_index)
        all_probs = torch.sigmoid(all_logits).numpy()

    val_probs = all_probs[data.val_mask.numpy()]
    best_thresh = find_optimal_threshold(y_val_np, val_probs)

    # Final evaluation on held-out test set
    y_test_np = data.y[data.test_mask].numpy()
    test_probs = all_probs[data.test_mask.numpy()]
    metrics = evaluate_predictions(y_test_np, test_probs, threshold=best_thresh, model_name="GCN")

    plot_evaluation_charts(y_test_np, test_probs, model_name="GCN", output_dir=figures_dir, threshold=best_thresh)

    metrics_path = os.path.join(models_dir, "gcn_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save full graph predictions
    preds_path = os.path.join(models_dir, "gcn_predictions.pt")
    torch.save({
        "probabilities": torch.tensor(all_probs, dtype=torch.float32),
        "test_threshold": best_thresh
    }, preds_path)

    return metrics

if __name__ == "__main__":
    metrics = train_gcn_model()
    print("GCN Evaluation Results:", json.dumps(metrics, indent=2))

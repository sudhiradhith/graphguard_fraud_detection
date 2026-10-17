"""
src/train_gat.py
Trains and evaluates Graph Attention Network (GAT) for GraphGuard.
Implements FR-07 (GAT model) with multi-head attention weights,
weighted BCE loss for class imbalance, early stopping on Val PR-AUC,
and attention weight extraction for explainability.
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
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

from src.evaluation import (
    evaluate_predictions, find_optimal_threshold,
    plot_evaluation_charts
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GraphGuard.TrainGAT")

class GATNet(nn.Module):
    """
    2-Layer Graph Attention Network (GAT) with Multi-Head Attention.
    Veličković et al. (2018) architecture adapted for imbalanced node classification.
    """
    def __init__(self, in_channels: int, hidden_channels: int = 64, heads: int = 4, dropout: float = 0.2):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels // heads, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels, hidden_channels // 2, heads=1, concat=False, dropout=dropout)
        self.classifier = nn.Linear(hidden_channels // 2, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Layer 1 (Multi-head attention)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2 (Single-head attention aggregation)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Classification Logits
        out = self.classifier(x).squeeze(-1)
        return out

    def get_attention_weights(self, x: torch.Tensor, edge_index: torch.Tensor):
        """
        Returns edge attention coefficients for GNN explainability (FR-10).
        """
        _, (edge_idx, alpha) = self.conv1(x, edge_index, return_attention_weights=True)
        return edge_idx, alpha

def train_gat_model(
    graph_path: str = "data/graph/pyg_transaction_graph.pt",
    models_dir: str = "models",
    figures_dir: str = "reports/figures",
    epochs: int = 120,
    lr: float = 0.005,
    hidden_dim: int = 64,
    heads: int = 4,
    weight_decay: float = 1e-4,
    patience: int = 20
) -> Dict[str, Any]:
    """
    Trains GAT model with weighted BCE loss to counter fraud class imbalance.
    Evaluates on test mask and saves checkpoints.
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
    logger.info(f"GAT loss pos_weight set to {pos_weight.item():.2f}")

    in_dim = data.x.size(1)
    model = GATNet(in_channels=in_dim, hidden_channels=hidden_dim, heads=heads, dropout=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_pr_auc = -1.0
    best_epoch = 0
    best_model_state = None
    epochs_no_improve = 0

    y_val_np = data.y[data.val_mask].numpy()

    logger.info("Starting GAT training...")
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
    model_save_path = os.path.join(models_dir, "gat_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "in_channels": in_dim,
        "hidden_channels": hidden_dim,
        "heads": heads,
        "best_val_pr_auc": best_val_pr_auc,
        "best_epoch": best_epoch
    }, model_save_path)
    logger.info(f"Saved best GAT model to {model_save_path}")

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
    metrics = evaluate_predictions(y_test_np, test_probs, threshold=best_thresh, model_name="GAT")

    plot_evaluation_charts(y_test_np, test_probs, model_name="GAT", output_dir=figures_dir, threshold=best_thresh)

    metrics_path = os.path.join(models_dir, "gat_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save full graph predictions
    preds_path = os.path.join(models_dir, "gat_predictions.pt")
    torch.save({
        "probabilities": torch.tensor(all_probs, dtype=torch.float32),
        "test_threshold": best_thresh
    }, preds_path)

    return metrics

if __name__ == "__main__":
    metrics = train_gat_model()
    print("GAT Evaluation Results:", json.dumps(metrics, indent=2))

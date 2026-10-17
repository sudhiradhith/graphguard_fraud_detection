"""
src/model_comparison.py
Consolidates evaluation metrics across all 6 models:
  - Logistic Regression, Random Forest, XGBoost
  - GCN, GraphSAGE, GAT
Generates comparison table, markdown report, and combined multi-model benchmark charts.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")

import json
import logging
import numpy as np
import pandas as pd
import torch

from src.evaluation import plot_multi_model_comparison

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GraphGuard.ModelComparison")

def generate_model_comparison_benchmark(
    models_dir: str = "models",
    figures_dir: str = "reports/figures",
    graph_path: str = "data/graph/pyg_transaction_graph.pt"
) -> pd.DataFrame:
    """
    Loads metrics and test predictions from all models, constructs comparison DataFrame,
    saves benchmark JSON/CSV, and plots combined benchmark curves.
    """
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Load ML Baseline metrics & predictions
    ml_metrics_path = os.path.join(models_dir, "ml_baseline_metrics.json")
    with open(ml_metrics_path, "r") as f:
        ml_metrics = json.load(f)

    ml_preds_path = os.path.join(models_dir, "ml_baseline_test_predictions.json")
    with open(ml_preds_path, "r") as f:
        ml_preds = json.load(f)

    # 2. Load GNN metrics & predictions
    gnn_models = ["gcn", "graphsage", "gat"]
    gnn_metrics = {}
    gnn_preds = {}

    pyg_data = torch.load(graph_path, weights_only=False)
    y_test = pyg_data.y[pyg_data.test_mask].numpy()

    for gm in gnn_models:
        met_p = os.path.join(models_dir, f"{gm}_metrics.json")
        if os.path.exists(met_p):
            with open(met_p, "r") as f:
                gnn_metrics[gm] = json.load(f)

        pred_p = os.path.join(models_dir, f"{gm}_predictions.pt")
        if os.path.exists(pred_p):
            data = torch.load(pred_p, weights_only=False)
            probs = data["probabilities"][pyg_data.test_mask].numpy()
            gnn_preds[gm] = {
                "y_true": y_test,
                "y_prob": probs,
                "best_threshold": data["test_threshold"]
            }

    # 3. Build unified rows
    rows = []

    # Baselines
    for name in ["Logistic Regression", "Random Forest", "XGBoost"]:
        if name in ml_metrics:
            m = ml_metrics[name]
            rows.append({
                "Model": name,
                "Family": "Tabular ML",
                "PR-AUC (Primary)": m["pr_auc"],
                "ROC-AUC": m["roc_auc"],
                "F1-Score": m["f1"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "TP": m["true_positives"],
                "FP": m["false_positives"],
                "FN": m["false_negatives"],
                "TN": m["true_negatives"],
                "Threshold": m["threshold"]
            })

    # GNNs
    gnn_display_names = {"gcn": "GCN", "graphsage": "GraphSAGE", "gat": "GAT"}
    for gm, display_name in gnn_display_names.items():
        if gm in gnn_metrics:
            m = gnn_metrics[gm]
            rows.append({
                "Model": display_name,
                "Family": "Graph Neural Network",
                "PR-AUC (Primary)": m["pr_auc"],
                "ROC-AUC": m["roc_auc"],
                "F1-Score": m["f1"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "TP": m["true_positives"],
                "FP": m["false_positives"],
                "FN": m["false_negatives"],
                "TN": m["true_negatives"],
                "Threshold": m["threshold"]
            })

    df_comp = pd.DataFrame(rows)
    df_comp = df_comp.sort_values(by="PR-AUC (Primary)", ascending=False).reset_index(drop=True)

    # Save CSV & JSON
    csv_path = os.path.join(models_dir, "model_comparison_benchmark.csv")
    df_comp.to_csv(csv_path, index=False)

    json_path = os.path.join(models_dir, "model_comparison_benchmark.json")
    df_comp.to_json(json_path, orient="records", indent=2)

    logger.info(f"Model comparison benchmark saved to {csv_path} and {json_path}")

    # 4. Plot Combined Curves
    combined_dict = {}
    for name in ["Logistic Regression", "Random Forest", "XGBoost"]:
        if name in ml_preds:
            combined_dict[name] = {
                "y_true": np.array(ml_preds[name]["y_true"]),
                "y_prob": np.array(ml_preds[name]["y_prob"])
            }
    for gm, display_name in gnn_display_names.items():
        if gm in gnn_preds:
            combined_dict[display_name] = {
                "y_true": gnn_preds[gm]["y_true"],
                "y_prob": gnn_preds[gm]["y_prob"]
            }

    plot_multi_model_comparison(combined_dict, output_dir=figures_dir)

    return df_comp

if __name__ == "__main__":
    df = generate_model_comparison_benchmark()
    print("\n=== GRAPHGUARD MODEL BENCHMARK COMPARISON ===")
    print(df.to_string(index=False))

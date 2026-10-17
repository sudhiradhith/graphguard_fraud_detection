"""
src/evaluation.py
Comprehensive Evaluation, Metric Reporting, and Risk Scoring Module for GraphGuard.
Implements FR-08 (Evaluation), FR-09 (Risk Scoring), and PRD Section 10.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, Optional, List
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GraphGuard.Evaluation")

# Presentation Risk Score Bands (FR-09)
DEFAULT_RISK_BANDS = {
    "Low": (0, 30),
    "Moderate": (30, 60),
    "High": (60, 85),
    "Critical": (85, 100)
}

def compute_risk_score(probability: np.ndarray or float) -> np.ndarray:
    """
    Implements FR-09: Converts predicted fraud probability [0.0, 1.0]
    to a presentation risk score on [0, 100].
    """
    scores = np.round(np.clip(probability, 0.0, 1.0) * 100.0).astype(int)
    return scores

def get_risk_tier(score: int or float, bands: Dict[str, Tuple[int, int]] = DEFAULT_RISK_BANDS) -> str:
    """
    Maps 0-100 risk score to configurable presentation tier.
    """
    for tier, (low, high) in bands.items():
        if low <= score < high:
            return tier
    return "Critical" if score >= 85 else "Low"

def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Finds threshold that maximizes F1 score on validation set
    to handle severe fraud class imbalance.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    return float(np.clip(best_thresh, 0.05, 0.95))

def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: Optional[float] = None,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Evaluates predictions with imbalance-centric metrics:
    PR-AUC (Primary), ROC-AUC, F1, Precision, Recall, Confusion Matrix.
    """
    if threshold is None:
        threshold = 0.5

    y_pred = (y_prob >= threshold).astype(int)

    # Imbalance metrics
    pr_auc = float(average_precision_score(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    metrics = {
        "model_name": model_name,
        "threshold": float(threshold),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "total_samples": int(len(y_true)),
        "fraud_samples": int(np.sum(y_true)),
        "fraud_prevalence": float(np.mean(y_true))
    }

    logger.info(
        f"[{model_name}] PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f} | "
        f"F1: {f1:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f} | (Thresh: {threshold:.3f})"
    )
    return metrics

def plot_evaluation_charts(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    output_dir: str = "reports/figures",
    threshold: float = 0.5
):
    """
    Generates and saves PR curve, ROC curve, and confusion matrix figure.
    """
    os.makedirs(output_dir, exist_ok=True)
    clean_name = model_name.lower().replace(" ", "_")

    # 1. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    no_skill = np.mean(y_true)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#1E88E5", lw=2, label=f"{model_name} (PR-AUC = {pr_auc:.3f})")
    ax.plot([0, 1], [no_skill, no_skill], linestyle="--", color="#E53935", label=f"No Skill ({no_skill:.2%})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve - {model_name}")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    pr_path = os.path.join(output_dir, f"pr_curve_{clean_name}.png")
    fig.savefig(pr_path, dpi=200)
    plt.close(fig)

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#43A047", lw=2, label=f"{model_name} (ROC-AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#757575", label="Random Chance (0.50)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - {model_name}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    roc_path = os.path.join(output_dir, f"roc_curve_{clean_name}.png")
    fig.savefig(roc_path, dpi=200)
    plt.close(fig)

    # 3. Confusion Matrix
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Legitimate (0)", "Fraud (1)"],
                yticklabels=["Legitimate (0)", "Fraud (1)"])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title(f"Confusion Matrix ({model_name}) @ Thresh={threshold:.2f}")
    fig.tight_layout()
    cm_path = os.path.join(output_dir, f"cm_{clean_name}.png")
    fig.savefig(cm_path, dpi=200)
    plt.close(fig)

    logger.info(f"Saved evaluation figures for {model_name} to {output_dir}")

def plot_multi_model_comparison(
    models_dict: Dict[str, Dict[str, np.ndarray]], # model_name -> {'y_true': ..., 'y_prob': ...}
    output_dir: str = "reports/figures"
):
    """
    Plots multi-model combined PR-AUC and ROC-AUC benchmark comparison charts.
    """
    os.makedirs(output_dir, exist_ok=True)
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    # 1. Combined PR Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    sample_y = next(iter(models_dict.values()))["y_true"]
    no_skill = np.mean(sample_y)
    ax.plot([0, 1], [no_skill, no_skill], linestyle=":", color="black", label=f"No Skill ({no_skill:.1%})")

    for i, (name, data) in enumerate(models_dict.items()):
        yt, yp = data["y_true"], data["y_prob"]
        prec, rec, _ = precision_recall_curve(yt, yp)
        score = average_precision_score(yt, yp)
        color = palette[i % len(palette)]
        ax.plot(rec, prec, lw=2, color=color, label=f"{name} (PR-AUC: {score:.3f})")

    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (Positive Predictive Value)")
    ax.set_title("Model Comparison: Precision-Recall Curves (Fraud Detection Benchmark)")
    ax.legend(loc="lower left", frameon=True)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "combined_pr_curves.png"), dpi=250)
    plt.close(fig)

    # 2. Combined ROC Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], linestyle=":", color="black", label="Random Baseline (0.500)")

    for i, (name, data) in enumerate(models_dict.items()):
        yt, yp = data["y_true"], data["y_prob"]
        fpr, tpr, _ = roc_curve(yt, yp)
        score = roc_auc_score(yt, yp)
        color = palette[i % len(palette)]
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{name} (ROC-AUC: {score:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Model Comparison: Receiver Operating Characteristic (ROC) Curves")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "combined_roc_curves.png"), dpi=250)
    plt.close(fig)

    logger.info("Saved combined PR and ROC comparison charts to reports/figures")

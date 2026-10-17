"""
src/train_ml.py
Trains and evaluates traditional Machine Learning baselines for GraphGuard:
  1. Logistic Regression (class_weight='balanced')
  2. Random Forest (class_weight='balanced_subsample')
  3. XGBoost (scale_pos_weight tuned for imbalance)
Implements FR-04 and leakage-aware chronological evaluation.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import joblib
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from src.feature_engineering import prepare_feature_data
from src.evaluation import (
    evaluate_predictions, find_optimal_threshold,
    plot_evaluation_charts, compute_risk_score
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GraphGuard.TrainML")

def train_and_evaluate_baselines(
    processed_dir: str = "data/processed",
    models_dir: str = "models",
    figures_dir: str = "reports/figures"
) -> Dict[str, Any]:
    """
    Trains Logistic Regression, Random Forest, and XGBoost baselines on chronological splits.
    Evaluates on test split, plots figures, and serializes trained models.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Load or prepare feature data
    feat_data = prepare_feature_data()
    X_train, y_train = feat_data["X_train"], feat_data["y_train"]
    X_val, y_val = feat_data["X_val"], feat_data["y_val"]
    X_test, y_test = feat_data["X_test"], feat_data["y_test"]

    n_pos = int(np.sum(y_train))
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = float(n_neg / max(1, n_pos))
    logger.info(f"Training split imbalance: {n_neg} Legit vs {n_pos} Fraud (scale_pos_weight: {scale_pos_weight:.2f})")

    results = {}
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=42,
            n_jobs=-1
        )
    }

    test_predictions = {}

    for name, model in models.items():
        logger.info(f"--- Training {name} ---")
        if name == "XGBoost":
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            model.fit(X_train, y_train)

        # Tune threshold on validation set
        val_probs = model.predict_proba(X_val)[:, 1]
        best_thresh = find_optimal_threshold(y_val, val_probs)
        logger.info(f"[{name}] Optimal F1 validation threshold: {best_thresh:.4f}")

        # Evaluate on test set
        test_probs = model.predict_proba(X_test)[:, 1]
        metrics = evaluate_predictions(y_test, test_probs, threshold=best_thresh, model_name=name)

        # Plot figures
        plot_evaluation_charts(y_test, test_probs, model_name=name, output_dir=figures_dir, threshold=best_thresh)

        # Save model
        clean_name = name.lower().replace(" ", "_")
        model_path = os.path.join(models_dir, f"{clean_name}.joblib")
        joblib.dump(model, model_path)
        logger.info(f"Saved {name} model to {model_path}")

        results[name] = metrics
        test_predictions[name] = {
            "y_true": y_test.tolist(),
            "y_prob": test_probs.tolist(),
            "best_threshold": best_thresh
        }

    # Save metrics JSON
    metrics_path = os.path.join(models_dir, "ml_baseline_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    preds_path = os.path.join(models_dir, "ml_baseline_test_predictions.json")
    with open(preds_path, "w") as f:
        json.dump(test_predictions, f)

    logger.info(f"Baseline ML training and evaluation complete. Saved summary to {metrics_path}")
    return results

if __name__ == "__main__":
    metrics = train_and_evaluate_baselines()
    print("Baseline ML Results:")
    print(json.dumps(metrics, indent=2))

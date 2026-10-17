"""
src/explainability.py
Hybrid Explainability Engine for GraphGuard combining Tabular SHAP and Graph Neighborhood Context.
Implements FR-10 (Explainability), FR-11 (Network Investigation), and PRD Section 11.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import joblib
import logging
import numpy as np
import pandas as pd
import shap
import torch
import networkx as nx
from typing import Dict, Any, List, Optional, Tuple

from src.feature_engineering import FeaturePipeline
from src.evaluation import compute_risk_score, get_risk_tier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GraphGuard.Explainability")

class ExplainabilityEngine:
    """
    Coordinates SHAP tabular explanations and Graph relational context explanations.
    """
    def __init__(
        self,
        engineered_csv: str = "data/processed/transactions_engineered.csv",
        pipeline_path: str = "models/feature_pipeline.joblib",
        xgb_model_path: str = "models/xgboost.joblib",
        graph_nx_path: str = "data/graph/heterogeneous_graph.joblib",
        pyg_graph_path: str = "data/graph/pyg_transaction_graph.pt"
    ):
        logger.info("Initializing Explainability Engine...")
        self.df = pd.read_csv(engineered_csv)
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])
        self.tx_id_to_idx = {row["transaction_id"]: i for i, row in self.df.iterrows()}

        self.pipeline = FeaturePipeline.load(pipeline_path)
        self.feature_names = self.pipeline.feature_names
        
        # Load XGBoost model for tabular explanations
        self.xgb_model = joblib.load(xgb_model_path) if os.path.exists(xgb_model_path) else None
        if self.xgb_model is not None:
            # Use small background sample for TreeExplainer
            background_sample = self.pipeline.transform(self.df.head(100))
            self.explainer = shap.TreeExplainer(self.xgb_model)
        else:
            self.explainer = None

        # Load NetworkX graph for relational queries
        self.G_nx: nx.Graph = joblib.load(graph_nx_path) if os.path.exists(graph_nx_path) else None
        logger.info("Explainability Engine initialized successfully.")

    def explain_tabular_shap(self, transaction_id: str, top_k: int = 6) -> Dict[str, Any]:
        """
        Computes SHAP feature importance contributions for a specific transaction.
        """
        if transaction_id not in self.tx_id_to_idx:
            return {"error": f"Transaction {transaction_id} not found."}
        
        idx = self.tx_id_to_idx[transaction_id]
        row_df = self.df.iloc[[idx]]
        X_row = self.pipeline.transform(row_df)

        if self.explainer is None:
            return {"error": "SHAP Explainer not initialized."}

        shap_values = self.explainer.shap_values(X_row)
        # Handle binary classification shap output
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        elif len(shap_values.shape) == 2:
            sv = shap_values[0]
        else:
            sv = shap_values

        # Pair feature names with values and contributions
        contributions = []
        raw_values = row_df.iloc[0].to_dict()

        for name, sh_val in zip(self.feature_names, sv):
            contributions.append({
                "feature": name,
                "shap_value": float(sh_val),
                "abs_impact": float(abs(sh_val)),
                "direction": "Risk Escalator" if sh_val > 0 else "Risk Mitigator"
            })

        contributions.sort(key=lambda x: x["abs_impact"], reverse=True)
        top_factors = contributions[:top_k]

        return {
            "transaction_id": transaction_id,
            "top_factors": top_factors,
            "base_value": float(self.explainer.expected_value) if hasattr(self.explainer, "expected_value") else 0.0
        }

    def explain_graph_context(self, transaction_id: str) -> Dict[str, Any]:
        """
        Extracts multi-entity relationship risk indicators:
          - Shared device usage & device fraud history
          - Shared IP velocity & IP fraud history
          - Customer transaction frequency
          - High-risk merchant connectivity
        """
        if self.G_nx is None or transaction_id not in self.G_nx:
            return {"error": f"Transaction {transaction_id} not found in graph."}

        tx_node = self.G_nx.nodes[transaction_id]
        neighbors = list(self.G_nx.neighbors(transaction_id))

        context = {
            "transaction_id": transaction_id,
            "amount": tx_node.get("amount", 0.0),
            "timestamp": tx_node.get("timestamp"),
            "connected_entities": {},
            "relational_red_flags": []
        }

        # Analyze each connected entity
        for nb in neighbors:
            nb_data = self.G_nx.nodes[nb]
            ntype = nb_data.get("node_type", "unknown")
            
            # Find 2-hop transactions connected through this entity
            entity_txs = [
                t for t in self.G_nx.neighbors(nb)
                if self.G_nx.nodes[t].get("node_type") == "transaction" and t != transaction_id
            ]
            fraud_txs = [t for t in entity_txs if self.G_nx.nodes[t].get("is_fraud") == 1]
            fraud_rate = len(fraud_txs) / max(1, len(entity_txs))

            context["connected_entities"][ntype] = {
                "entity_id": nb,
                "total_shared_transactions": len(entity_txs),
                "fraudulent_shared_transactions": len(fraud_txs),
                "fraud_rate_on_entity": float(fraud_rate)
            }

            # Generate relational red flags
            if ntype == "device" and len(entity_txs) >= 3:
                context["relational_red_flags"].append(
                    f"Device {nb} shared across {len(entity_txs) + 1} distinct transactions ({len(fraud_txs)} previously fraudulent)."
                )
            if ntype == "ip_address" and len(entity_txs) >= 4:
                context["relational_red_flags"].append(
                    f"High IP velocity: {nb} generated {len(entity_txs) + 1} transactions."
                )
            if ntype == "device" and fraud_rate >= 0.5 and len(fraud_txs) > 0:
                context["relational_red_flags"].append(
                    f"Critical Entity: Device has {fraud_rate:.0%} historical fraud association."
                )

        return context

    def get_unified_investigation(
        self,
        transaction_id: str,
        predicted_prob: float
    ) -> Dict[str, Any]:
        """
        Creates an end-to-end investigation dossier for dashboard consumption:
        Risk score, classification badge, tabular SHAP drivers, and graph relational alerts.
        """
        score = int(compute_risk_score(np.array([predicted_prob]))[0])
        tier = get_risk_tier(score)

        shap_exp = self.explain_tabular_shap(transaction_id)
        graph_exp = self.explain_graph_context(transaction_id)

        # Raw transaction data
        idx = self.tx_id_to_idx.get(transaction_id)
        raw_record = self.df.iloc[idx].to_dict() if idx is not None else {}

        return {
            "transaction_id": transaction_id,
            "risk_score": score,
            "risk_tier": tier,
            "predicted_probability": float(predicted_prob),
            "actual_label": int(raw_record.get("is_fraud", -1)),
            "fraud_scenario": str(raw_record.get("fraud_scenario", "N/A")),
            "raw_attributes": raw_record,
            "shap_explanation": shap_exp,
            "graph_explanation": graph_exp
        }

if __name__ == "__main__":
    engine = ExplainabilityEngine()
    test_tx = "TXN_000010"
    report = engine.get_unified_investigation(test_tx, predicted_prob=0.88)
    print("Investigation Dossier for", test_tx)
    print(json.dumps(report, indent=2, default=str))

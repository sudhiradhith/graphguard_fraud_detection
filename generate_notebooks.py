"""
generate_notebooks.py
Programmatically generates the 10 portfolio-grade Jupyter Notebooks for GraphGuard.
"""

import os
import nbformat as nbf

NOTEBOOKS_DIR = "notebooks"
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

def create_nb(filename, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    filepath = os.path.join(NOTEBOOKS_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Generated {filepath}")

# ==========================================
# 01_data_understanding.ipynb
# ==========================================
create_nb("01_data_understanding.ipynb", [
    nbf.v4.new_markdown_cell("""# GraphGuard: 01 - Data Understanding & Schema Inspection
**Domain**: FinTech • Banking • Digital Payments  
**Objective**: Ingest raw financial transactions, validate schema, verify data types, inspect missingness, duplicate records, and target class balance (FR-01)."""),
    nbf.v4.new_code_cell("""import os
import pandas as pd
import numpy as np

# Load raw transactions
raw_path = os.path.join("..", "data", "raw", "transactions.csv")
df_raw = pd.read_csv(raw_path)
print(f"Loaded raw dataset with {len(df_raw):,} records and {df_raw.shape[1]} columns.")
df_raw.head()"""),
    nbf.v4.new_markdown_cell("""### 1. Schema Validation & Datatypes Inspection"""),
    nbf.v4.new_code_cell("""print("Dataset Info:")
df_raw.info()"""),
    nbf.v4.new_markdown_cell("""### 2. Missing Value & Duplicate Analysis"""),
    nbf.v4.new_code_cell("""null_counts = df_raw.isnull().sum()
dups = df_raw.duplicated().sum()
dup_tx = df_raw["transaction_id"].duplicated().sum()

print("Missing values per column:\\n", null_counts[null_counts > 0])
print(f"\\nExact duplicate rows: {dups}")
print(f"Duplicate transaction IDs: {dup_tx}")"""),
    nbf.v4.new_markdown_cell("""### 3. Fraud Label Prevalence & Class Imbalance"""),
    nbf.v4.new_code_cell("""fraud_counts = df_raw["is_fraud"].value_counts()
fraud_rate = df_raw["is_fraud"].mean()

print(f"Class Distribution:\\n{fraud_counts}")
print(f"Overall Fraud Prevalence: {fraud_rate:.2%}")
print(f"Severe Class Imbalance Ratio: ~1 : {int(1/fraud_rate)}")""")
])

# ==========================================
# 02_eda.ipynb
# ==========================================
create_nb("02_eda.ipynb", [
    nbf.v4.new_markdown_cell("""# GraphGuard: 02 - Exploratory Data Analysis (EDA)
**Objective**: Analyze transaction amount distributions, diurnal spending curves, merchant category risks, and geographical channels."""),
    nbf.v4.new_code_cell("""import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cleaned_path = os.path.join("..", "data", "processed", "transactions_cleaned.csv")
df = pd.read_csv(cleaned_path)
df["timestamp"] = pd.to_datetime(df["timestamp"])
print(f"Loaded cleaned transactions: {len(df):,} records")"""),
    nbf.v4.new_markdown_cell("""### 1. Transaction Amount Distribution (Log-Normal vs Fraud Spike)"""),
    nbf.v4.new_code_cell("""fig, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df[df["is_fraud"] == 0]["amount"], bins=50, kde=True, ax=ax[0], color="#2563EB")
ax[0].set_title("Legitimate Transaction Amount Distribution")
ax[0].set_xlabel("Amount ($)")

sns.histplot(df[df["is_fraud"] == 1]["amount"], bins=50, kde=True, ax=ax[1], color="#DC2626")
ax[1].set_title("Fraudulent Transaction Amount Distribution")
ax[1].set_xlabel("Amount ($)")
plt.tight_layout()
plt.show()"""),
    nbf.v4.new_markdown_cell("""### 2. Diurnal Temporal Spending Patterns"""),
    nbf.v4.new_code_cell("""df["hour"] = df["timestamp"].dt.hour
hourly_fraud = df.groupby(["hour", "is_fraud"]).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(10, 4))
hourly_fraud[0].plot(ax=ax, label="Legitimate", color="#10B981", lw=2)
(hourly_fraud[1] * 10).plot(ax=ax, label="Fraud (Scaled 10x)", color="#EF4444", lw=2.5, linestyle="--")
ax.set_title("Hourly Distribution: Legitimate vs Fraud Activity")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Transaction Count")
ax.legend()
plt.show()"""),
    nbf.v4.new_markdown_cell("""### 3. Fraud Risk by Merchant Category"""),
    nbf.v4.new_code_cell("""cat_summary = df.groupby("merchant_category").agg(
    total=("transaction_id", "count"),
    fraud=("is_fraud", "sum"),
    fraud_rate=("is_fraud", "mean")
).sort_values("fraud_rate", ascending=False)
print("Merchant Category Risk Breakdown:")
cat_summary""")
])

# ==========================================
# 03_feature_engineering.ipynb
# ==========================================
create_nb("03_feature_engineering.ipynb", [
    nbf.v4.new_markdown_cell("""# GraphGuard: 03 - Leakage-Safe Feature Engineering
**Objective**: Engineer chronological velocity, historical customer deviations, entity reuse metrics, and fit transformation pipeline strictly on the training partition (FR-03)."""),
    nbf.v4.new_code_cell("""import os
import sys
sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import numpy as np
from src.feature_engineering import prepare_feature_data

# Execute feature pipeline
feat_dict = prepare_feature_data(
    input_csv=os.path.join("..", "data", "processed", "transactions_cleaned.csv"),
    save_pipeline_path=os.path.join("..", "models", "feature_pipeline.joblib"),
    output_processed_dir=os.path.join("..", "data", "processed")
)

print(f"Features Engineered ({len(feat_dict['feature_names'])}):")
for f in feat_dict["feature_names"]:
    print(f" - {f}")"""),
    nbf.v4.new_markdown_cell("""### Train / Validation / Test Chronological Partition Validation"""),
    nbf.v4.new_code_cell("""train_df = feat_dict["train_df"]
val_df = feat_dict["val_df"]
test_df = feat_dict["test_df"]

print(f"Train Partition: {len(train_df)} samples ({train_df['is_fraud'].sum()} frauds) | Max Date: {train_df['timestamp'].max()}")
print(f"Val Partition:   {len(val_df)} samples ({val_df['is_fraud'].sum()} frauds) | Date Range: {val_df['timestamp'].min()} to {val_df['timestamp'].max()}")
print(f"Test Partition:  {len(test_df)} samples ({test_df['is_fraud'].sum()} frauds) | Date Range: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")

assert train_df['timestamp'].max() <= val_df['timestamp'].min(), "Temporal leakage detected between train and val!"
assert val_df['timestamp'].max() <= test_df['timestamp'].min(), "Temporal leakage detected between val and test!"
print("Zero temporal look-ahead leakage verified successfully!")""")
])

# ==========================================
# 04_baseline_ml.ipynb
# ==========================================
create_nb("04_baseline_ml.ipynb", [
    nbf.v4.new_markdown_cell("""# GraphGuard: 04 - Traditional Machine Learning Baselines
**Objective**: Train and evaluate Logistic Regression, Random Forest, and XGBoost baselines using PR-AUC and ROC-AUC (FR-04)."""),
    nbf.v4.new_code_cell("""import os
import sys
sys.path.insert(0, os.path.abspath(".."))

import json
from src.train_ml import train_and_evaluate_baselines

metrics = train_and_evaluate_baselines(
    processed_dir=os.path.join("..", "data", "processed"),
    models_dir=os.path.join("..", "models"),
    figures_dir=os.path.join("..", "reports", "figures")
)
print(json.dumps(metrics, indent=2))""")
])

# ==========================================
# 05_graph_construction.ipynb
# ==========================================
create_nb("05_graph_construction.ipynb", [
    nbf.v4.new_markdown_cell("""# GraphGuard: 05 - Graph Construction Pipeline
**Objective**: Construct multi-entity heterogeneous NetworkX graph and PyTorch Geometric transaction relationship graph (FR-05)."""),
    nbf.v4.new_code_cell("""import os
import sys
sys.path.insert(0, os.path.abspath(".."))

import json
import torch
import joblib
from src.graph_builder import build_and_save_all_graphs

result = build_and_save_all_graphs(
    engineered_csv=os.path.join("..", "data", "processed", "transactions_engineered.csv"),
    feature_pipeline_path=os.path.join("..", "models", "feature_pipeline.joblib"),
    output_dir=os.path.join("..", "data", "graph")
)

pyg_data = result["pyg_data"]
print("PyG Transaction Graph Constructed:")
print(f" - Nodes: {pyg_data.num_nodes}")
print(f" - Directed Edges: {pyg_data.edge_index.size(1)}")
print(f" - Node Feature Dimension: {pyg_data.x.size(1)}")""")
])

# ==========================================
# 06_graph_analysis.ipynb
# ==========================================
create_nb("06_graph_analysis.ipynb", [
    nbf.v4.new_markdown_cell("""# GraphGuard: 06 - Graph Topology Analysis & Ring Detection
**Objective**: Analyze degree distributions, connected components, edge homophily, and suspicious entity clusters (FR-06)."""),
    nbf.v4.new_code_cell("""import os
import json
import pandas as pd
import matplotlib.pyplot as plt

stats_path = os.path.join("..", "data", "graph", "graph_statistics.json")
with open(stats_path, "r") as f:
    stats = json.load(f)

print("Key Graph Structural Metrics:")
print(f" - Mean Degree: {stats['degree_mean']:.2f}")
print(f" - Edge Label Homophily: {stats['edge_homophily']:.2%}")
print(f" - Fraud-Fraud Relational Edges: {stats['fraud_fraud_edges']:,}")
print(f" - Graph Density: {stats['graph_density']:.6f}")

clusters_path = os.path.join("..", "data", "graph", "suspicious_clusters.json")
with open(clusters_path, "r") as f:
    clusters = json.load(f)
print(f"\\nIdentified {len(clusters)} Suspicious Entity Rings/Clusters.")
pd.DataFrame(clusters).head(10)""")
])

# ==========================================
# 07_gcn.ipynb
# ==========================================
create_nb("07_gcn.ipynb", [
    nbf.v4.new_markdown_cell("""# GraphGuard: 07 - Graph Convolutional Network (GCN)
**Objective**: Train and evaluate Kipf & Welling 2-layer GCN with weighted BCE loss and analyze neighborhood over-smoothing (FR-07)."""),
    nbf.v4.new_code_cell("""import os
import sys
sys.path.insert(0, os.path.abspath(".."))

import json
from src.train_gcn import train_gcn_model

metrics_gcn = train_gcn_model(
    graph_path=os.path.join("..", "data", "graph", "pyg_transaction_graph.pt"),
    models_dir=os.path.join("..", "models"),
    figures_dir=os.path.join("..", "reports", "figures"),
    epochs=100
)
print("GCN Test Evaluation Metrics:")
print(json.dumps(metrics_gcn, indent=2))""")
])

# ==========================================
# 08_graphsage.ipynb
# ==========================================
create_nb("08_graphsage.ipynb", [
    nbf.v4.new_markdown_cell("""# GraphGuard: 08 - GraphSAGE Inductive Neighborhood Aggregation
**Objective**: Train and evaluate GraphSAGE with mean aggregation for robust localized fraud ring detection (FR-07)."""),
    nbf.v4.new_code_cell("""import os
import sys
sys.path.insert(0, os.path.abspath(".."))

import json
from src.train_graphsage import train_graphsage_model

metrics_sage = train_graphsage_model(
    graph_path=os.path.join("..", "data", "graph", "pyg_transaction_graph.pt"),
    models_dir=os.path.join("..", "models"),
    figures_dir=os.path.join("..", "reports", "figures"),
    epochs=100
)
print("GraphSAGE Test Evaluation Metrics:")
print(json.dumps(metrics_sage, indent=2))""")
])

# ==========================================
# 09_gat.ipynb
# ==========================================
create_nb("09_gat.ipynb", [
    nbf.v4.new_markdown_cell("""# GraphGuard: 09 - Graph Attention Network (GAT)
**Objective**: Train and evaluate Graph Attention Network with multi-head self-attention and attention extraction for explainability (FR-07)."""),
    nbf.v4.new_code_cell("""import os
import sys
sys.path.insert(0, os.path.abspath(".."))

import json
from src.train_gat import train_gat_model

metrics_gat = train_gat_model(
    graph_path=os.path.join("..", "data", "graph", "pyg_transaction_graph.pt"),
    models_dir=os.path.join("..", "models"),
    figures_dir=os.path.join("..", "reports", "figures"),
    epochs=100
)
print("GAT Test Evaluation Metrics:")
print(json.dumps(metrics_gat, indent=2))""")
])

# ==========================================
# 10_model_comparison.ipynb
# ==========================================
create_nb("10_model_comparison.ipynb", [
    nbf.v4.new_markdown_cell("""# GraphGuard: 10 - Unified Benchmark & Model Comparison
**Objective**: Compare all 6 models (Logistic Regression, Random Forest, XGBoost, GCN, GraphSAGE, GAT) on identical held-out test data (FR-08)."""),
    nbf.v4.new_code_cell("""import os
import sys
sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
from src.model_comparison import generate_model_comparison_benchmark

df_bench = generate_model_comparison_benchmark(
    models_dir=os.path.join("..", "models"),
    figures_dir=os.path.join("..", "reports", "figures"),
    graph_path=os.path.join("..", "data", "graph", "pyg_transaction_graph.pt")
)

print("\\n=== GRAPHGUARD CONSOLIDATED BENCHMARK ===")
display(df_bench)""")
])

print("All 10 Jupyter Notebooks created successfully in notebooks/")

"""
src/graph_builder.py
Constructs both NetworkX Heterogeneous Entity-Transaction Graph and
PyTorch Geometric Transaction-Relationship Graph for GraphGuard.
Implements FR-05 (Graph Builder) and FR-06 (Graph Analysis).
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import joblib
import logging
import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, degree
from typing import Dict, Any, Tuple, Optional, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GraphGuard.GraphBuilder")

def build_heterogeneous_networkx_graph(df: pd.DataFrame) -> nx.Graph:
    """
    Constructs a full multi-entity bipartite/heterogeneous NetworkX graph.
    Nodes:
      - Transaction: 'TXN_...' (type='transaction', amount, is_fraud, timestamp, scenario)
      - Customer: 'CUST_...' (type='customer')
      - Device: 'DEV_...' (type='device')
      - IP: 'IP_...' (type='ip_address')
      - Merchant: 'MERCH_...' (type='merchant', category)
    Edges:
      - Customer -> Transaction (rel='MAKES')
      - Transaction -> Device (rel='USES')
      - Transaction -> IP (rel='ORIGINATES_FROM')
      - Transaction -> Merchant (rel='PAYS')
    """
    logger.info(f"Building heterogeneous NetworkX graph from {len(df)} transactions...")
    G = nx.Graph()

    for _, row in df.iterrows():
        tx_id = str(row["transaction_id"])
        c_id = str(row["customer_id"])
        d_id = str(row["device_id"])
        ip_id = f"IP_{row['ip_address']}"
        m_id = str(row["merchant_id"])

        # Add transaction node
        G.add_node(
            tx_id,
            node_type="transaction",
            amount=float(row["amount"]),
            is_fraud=int(row["is_fraud"]),
            timestamp=str(row["timestamp"]),
            scenario=str(row.get("fraud_scenario", "unknown"))
        )

        # Add entity nodes if not exist
        if not G.has_node(c_id):
            G.add_node(c_id, node_type="customer")
        if not G.has_node(d_id):
            G.add_node(d_id, node_type="device")
        if not G.has_node(ip_id):
            G.add_node(ip_id, node_type="ip_address")
        if not G.has_node(m_id):
            G.add_node(m_id, node_type="merchant", category=str(row.get("merchant_category", "unknown")))

        # Add relational edges
        G.add_edge(c_id, tx_id, relation="MAKES")
        G.add_edge(tx_id, d_id, relation="USES")
        G.add_edge(tx_id, ip_id, relation="ORIGINATES_FROM")
        G.add_edge(tx_id, m_id, relation="PAYS")

    logger.info(f"Heterogeneous graph constructed: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G

def build_pyg_transaction_graph(
    df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    max_neighbors_per_entity: int = 50
) -> Data:
    """
    Constructs a PyTorch Geometric Data object for transaction-level node classification.
    Connects transaction pairs that share:
      1. device_id (device-sharing / bot rings)
      2. ip_address (IP velocity / proxy rings)
      3. customer_id (same-customer transaction sequence)
    
    Includes edge type attributes:
      0: shared_customer
      1: shared_device
      2: shared_ip
    """
    logger.info(f"Building PyG transaction relationship graph for {len(df)} transactions...")
    
    edge_src = []
    edge_dst = []
    edge_types = []

    # Map transaction ID / row index
    tx_to_idx = {row["transaction_id"]: i for i, row in df.iterrows()}

    # Groupings
    logger.info("Computing shared-entity connections...")
    
    def connect_group(col_name: str, rel_type: int):
        grouped = df.groupby(col_name).indices
        count = 0
        for entity_val, indices in grouped.items():
            if len(indices) < 2:
                continue
            # Cap maximum edges per entity to prevent supernode degree explosion
            if len(indices) > max_neighbors_per_entity:
                indices = np.random.choice(indices, size=max_neighbors_per_entity, replace=False)
            
            # Connect pairs
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    u = indices[i]
                    v = indices[j]
                    edge_src.append(u)
                    edge_dst.append(v)
                    edge_types.append(rel_type)
                    count += 1
        return count

    n_cust_edges = connect_group("customer_id", 0)
    n_dev_edges = connect_group("device_id", 1)
    n_ip_edges = connect_group("ip_address", 2)
    
    logger.info(f"Extracted connections: {n_cust_edges} shared-customer, {n_dev_edges} shared-device, {n_ip_edges} shared-IP.")

    if len(edge_src) == 0:
        logger.warning("No edges formed! Adding self-loops.")
        edge_index = torch.arange(len(df), dtype=torch.long).repeat(2, 1)
        edge_type_tensor = torch.zeros(len(df), dtype=torch.long)
    else:
        src = torch.tensor(edge_src, dtype=torch.long)
        dst = torch.tensor(edge_dst, dtype=torch.long)
        raw_edge_index = torch.stack([src, dst], dim=0)
        raw_edge_types = torch.tensor(edge_types, dtype=torch.long)

        # Make undirected
        edge_index, edge_type_tensor = to_undirected(raw_edge_index, raw_edge_types, num_nodes=len(df))

    # Tensors
    x_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    train_mask_tensor = torch.tensor(train_mask, dtype=torch.bool)
    val_mask_tensor = torch.tensor(val_mask, dtype=torch.bool)
    test_mask_tensor = torch.tensor(test_mask, dtype=torch.bool)

    pyg_data = Data(
        x=x_tensor,
        edge_index=edge_index,
        edge_type=edge_type_tensor,
        y=y_tensor,
        train_mask=train_mask_tensor,
        val_mask=val_mask_tensor,
        test_mask=test_mask_tensor
    )

    # Attach metadata
    pyg_data.num_nodes = len(df)
    logger.info(f"PyG graph constructed: {pyg_data.num_nodes} nodes, {pyg_data.edge_index.size(1)} directed edges.")
    return pyg_data

def analyze_graph_structure(G_nx: nx.Graph, pyg_data: Data) -> Dict[str, Any]:
    """
    Implements FR-06: Graph Analysis.
    Calculates degree statistics, connected components, density,
    clustering coefficient, and label homophily.
    """
    logger.info("Performing comprehensive graph structural analysis...")
    stats = {}

    # 1. Heterogeneous Graph Stats
    node_type_counts = {}
    for _, d in G_nx.nodes(data=True):
        ntype = d.get("node_type", "unknown")
        node_type_counts[ntype] = node_type_counts.get(ntype, 0) + 1
    stats["heterogeneous_nodes"] = node_type_counts
    stats["heterogeneous_total_nodes"] = G_nx.number_of_nodes()
    stats["heterogeneous_total_edges"] = G_nx.number_of_edges()

    # Connected components in entity graph
    components = list(nx.connected_components(G_nx))
    stats["heterogeneous_connected_components"] = len(components)
    stats["heterogeneous_largest_component_size"] = max(len(c) for c in components) if components else 0

    # 2. PyG Transaction Graph Stats
    num_nodes = pyg_data.num_nodes
    num_edges = pyg_data.edge_index.size(1)
    stats["tx_graph_nodes"] = num_nodes
    stats["tx_graph_edges"] = num_edges

    # Degree statistics
    deg = degree(pyg_data.edge_index[0], num_nodes=num_nodes).numpy()
    stats["degree_mean"] = float(np.mean(deg))
    stats["degree_std"] = float(np.std(deg))
    stats["degree_median"] = float(np.median(deg))
    stats["degree_max"] = int(np.max(deg))
    stats["degree_min"] = int(np.min(deg))
    stats["isolated_nodes_count"] = int(np.sum(deg == 0))
    stats["isolated_nodes_pct"] = float(np.sum(deg == 0) / num_nodes)

    # Edge density
    possible_edges = num_nodes * (num_nodes - 1)
    stats["graph_density"] = float(num_edges / possible_edges) if possible_edges > 0 else 0.0

    # Label homophily (fraction of edges connecting nodes with same label)
    y = pyg_data.y.numpy()
    src = pyg_data.edge_index[0].numpy()
    dst = pyg_data.edge_index[1].numpy()
    same_label = (y[src] == y[dst]).sum()
    fraud_fraud_edges = ((y[src] == 1) & (y[dst] == 1)).sum()
    stats["edge_homophily"] = float(same_label / num_edges) if num_edges > 0 else 0.0
    stats["fraud_fraud_edges"] = int(fraud_fraud_edges)
    stats["fraud_fraud_edge_pct"] = float(fraud_fraud_edges / num_edges) if num_edges > 0 else 0.0

    # Degree distribution percentiles
    stats["degree_percentiles"] = {
        "p25": float(np.percentile(deg, 25)),
        "p50": float(np.percentile(deg, 50)),
        "p75": float(np.percentile(deg, 75)),
        "p90": float(np.percentile(deg, 90)),
        "p99": float(np.percentile(deg, 99)),
    }

    logger.info(f"Graph analysis complete: Homophily: {stats['edge_homophily']:.2%}, Fraud-Fraud edges: {stats['fraud_fraud_edges']}, Mean Degree: {stats['degree_mean']:.2f}")
    return stats

def extract_transaction_subgraph(
    G_nx: nx.Graph,
    transaction_id: str,
    depth: int = 2
) -> nx.Graph:
    """
    Implements FR-11: Extracts k-hop ego-neighborhood for a specific transaction
    including connected customers, devices, IPs, merchants, and 2-hop adjacent transactions.
    """
    if transaction_id not in G_nx:
        logger.warning(f"Transaction {transaction_id} not found in graph.")
        return nx.Graph()

    sub_nodes = {transaction_id}
    current_frontier = {transaction_id}
    
    for _ in range(depth):
        next_frontier = set()
        for node in current_frontier:
            for neighbor in G_nx.neighbors(node):
                if neighbor not in sub_nodes:
                    next_frontier.add(neighbor)
                    sub_nodes.add(neighbor)
        current_frontier = next_frontier

    subgraph = G_nx.subgraph(sub_nodes).copy()
    return subgraph

def find_suspicious_clusters(G_nx: nx.Graph, min_fraud_count: int = 2) -> List[Dict[str, Any]]:
    """
    Detects high-risk communities and clusters (shared devices, IPs, and customer syndicates)
    having elevated concentrations of fraud transactions.
    """
    logger.info("Detecting suspicious clusters in the financial network...")
    # Consider projection on devices, IPs, customers
    suspicious_clusters = []

    # Find devices used by multiple fraud transactions
    for node, data in G_nx.nodes(data=True):
        ntype = data.get("node_type")
        if ntype in ["device", "ip_address"]:
            neighbors = list(G_nx.neighbors(node))
            tx_neighbors = [nb for nb in neighbors if G_nx.nodes[nb].get("node_type") == "transaction"]
            fraud_txs = [tx for tx in tx_neighbors if G_nx.nodes[tx].get("is_fraud") == 1]
            
            if len(fraud_txs) >= min_fraud_count:
                total_amt = sum(G_nx.nodes[tx].get("amount", 0.0) for tx in fraud_txs)
                suspicious_clusters.append({
                    "entity_id": node,
                    "entity_type": ntype,
                    "total_connected_transactions": len(tx_neighbors),
                    "fraud_transactions_count": len(fraud_txs),
                    "fraud_rate": len(fraud_txs) / max(1, len(tx_neighbors)),
                    "total_fraud_amount": total_amt,
                    "connected_fraud_tx_ids": fraud_txs
                })

    suspicious_clusters.sort(key=lambda x: x["fraud_transactions_count"], reverse=True)
    logger.info(f"Identified {len(suspicious_clusters)} suspicious entity clusters.")
    return suspicious_clusters

def build_and_save_all_graphs(
    engineered_csv: str = "data/processed/transactions_engineered.csv",
    feature_pipeline_path: str = "models/feature_pipeline.joblib",
    output_dir: str = "data/graph"
) -> Dict[str, Any]:
    """
    Orchestrates end-to-end graph construction, analysis, and serialization.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(engineered_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 1. Build Heterogeneous Graph
    G_nx = build_heterogeneous_networkx_graph(df)
    nx_path = os.path.join(output_dir, "heterogeneous_graph.joblib")
    joblib.dump(G_nx, nx_path)
    logger.info(f"Saved NetworkX graph to {nx_path}")

    # 2. Build PyG Transaction Graph
    from src.feature_engineering import FeaturePipeline
    pipeline = FeaturePipeline.load(feature_pipeline_path)
    X = pipeline.transform(df)
    y = df["is_fraud"].values

    n = len(df)
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    train_mask[:train_end] = True
    val_mask[train_end:val_end] = True
    test_mask[val_end:] = True

    pyg_data = build_pyg_transaction_graph(df, X, y, train_mask, val_mask, test_mask)
    pyg_path = os.path.join(output_dir, "pyg_transaction_graph.pt")
    torch.save(pyg_data, pyg_path)
    logger.info(f"Saved PyG data to {pyg_path}")

    # 3. Analyze Graph
    stats = analyze_graph_structure(G_nx, pyg_data)
    stats_path = os.path.join(output_dir, "graph_statistics.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved graph statistics to {stats_path}")

    # 4. Extract Suspicious Clusters
    clusters = find_suspicious_clusters(G_nx)
    clusters_path = os.path.join(output_dir, "suspicious_clusters.json")
    with open(clusters_path, "w") as f:
        json.dump(clusters, f, indent=2)
    logger.info(f"Saved suspicious clusters to {clusters_path}")

    return {
        "G_nx": G_nx,
        "pyg_data": pyg_data,
        "stats": stats,
        "clusters": clusters
    }

if __name__ == "__main__":
    result = build_and_save_all_graphs()
    print("Graph Construction Finished Successfully!")
    print(f"Nodes: {result['pyg_data'].num_nodes}, Edges: {result['pyg_data'].edge_index.size(1)}")
    print(f"Graph Stats Summary: {json.dumps(result['stats'], indent=2)}")

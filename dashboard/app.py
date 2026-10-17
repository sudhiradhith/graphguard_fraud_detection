"""
dashboard/app.py
GraphGuard: Graph Neural Network–Based Financial Fraud Detection System
Interactive Streamlit Analyst Dashboard with Unique Cyberdeck HUD Navigation & Ultra-Glossy UI.
Implements FR-11 and FR-12.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import torch

from src.evaluation import compute_risk_score, get_risk_tier
from src.explainability import ExplainabilityEngine
from src.graph_builder import extract_transaction_subgraph

# -----------------------------------------------------------------------------
# STREAMLIT PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="GraphGuard | GNN Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State Navigation
if "active_page" not in st.session_state:
    st.session_state["active_page"] = "📊 Executive Overview"
if "search_tx_id" not in st.session_state:
    st.session_state["search_tx_id"] = None

# -----------------------------------------------------------------------------
# ULTRA-GLOSSY GLASSMORPHIC & CYBERDECK HUD CSS THEME
# -----------------------------------------------------------------------------
GLOSSY_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    :root {
        --glass-bg: rgba(15, 23, 42, 0.65);
        --glass-card: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.02) 100%);
        --glass-border: rgba(255, 255, 255, 0.12);
        --glass-border-glow: rgba(56, 189, 248, 0.35);
        --glass-highlight: inset 0 1px 1px 0 rgba(255, 255, 255, 0.25);
        --neon-cyan: #06B6D4;
        --neon-sky: #38BDF8;
        --neon-violet: #8B5CF6;
        --neon-purple: #A855F7;
        --neon-emerald: #10B981;
        --neon-rose: #F43F5E;
        --neon-amber: #F59E0B;
    }

    /* Overall Application Canvas */
    .stApp {
        background-color: #070B14 !important;
        background-image: 
            radial-gradient(circle at 10% 15%, rgba(6, 182, 212, 0.14) 0%, transparent 35%),
            radial-gradient(circle at 90% 20%, rgba(99, 102, 241, 0.16) 0%, transparent 40%),
            radial-gradient(circle at 50% 85%, rgba(168, 85, 247, 0.12) 0%, transparent 45%),
            radial-gradient(circle at 80% 90%, rgba(16, 185, 129, 0.10) 0%, transparent 35%) !important;
        background-attachment: fixed !important;
        color: #F1F5F9 !important;
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
    }

    /* Glossy Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(11, 17, 33, 0.85) 0%, rgba(7, 11, 20, 0.95) 100%) !important;
        backdrop-filter: blur(28px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(28px) saturate(180%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        box-shadow: 4px 0 32px rgba(0, 0, 0, 0.5) !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.08) !important;
    }

    /* Cyberdeck Floating Top Navigation Bar */
    .cyber-nav-container {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(10, 15, 30, 0.90) 100%);
        backdrop-filter: blur(28px) saturate(200%);
        -webkit-backdrop-filter: blur(28px) saturate(200%);
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 1.25rem;
        padding: 0.65rem 0.8rem;
        box-shadow: 0 16px 40px 0 rgba(0, 0, 0, 0.5), inset 0 1px 1px 0 rgba(255, 255, 255, 0.25);
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }

    .cyber-nav-container::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #06B6D4 0%, #8B5CF6 50%, #EC4899 100%);
    }

    /* Top HUD Telemetry Status Ribbon */
    .hud-telemetry-ribbon {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 0.75rem;
        padding: 0.45rem 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.76rem;
        margin-bottom: 1rem;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }

    /* Glowing Titles & Headers */
    .glossy-hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #FFFFFF 0%, #E2E8F0 40%, #38BDF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .glossy-subtitle {
        color: #94A3B8;
        font-size: 0.98rem;
        font-weight: 400;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }

    /* Live Surveillance Badge */
    .live-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.35rem 0.9rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        background: rgba(6, 182, 212, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.4);
        color: #38BDF8;
        box-shadow: 0 0 16px rgba(6, 182, 212, 0.25);
        margin-bottom: 0.8rem;
    }

    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #38BDF8;
        box-shadow: 0 0 10px #38BDF8;
        animation: pulseAnimation 2s infinite;
    }

    @keyframes pulseAnimation {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.7); }
        70% { transform: scale(1.1); box-shadow: 0 0 0 8px rgba(56, 189, 248, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(56, 189, 248, 0); }
    }

    /* Glossy Glass Cards */
    .glass-card {
        background: var(--glass-card);
        backdrop-filter: blur(24px) saturate(180%);
        -webkit-backdrop-filter: blur(24px) saturate(180%);
        border: 1px solid var(--glass-border);
        border-radius: 1rem;
        padding: 1.4rem;
        box-shadow: 0 12px 32px 0 rgba(0, 0, 0, 0.4), var(--glass-highlight);
        position: relative;
        overflow: hidden;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        margin-bottom: 1rem;
    }

    .glass-card:hover {
        border-color: rgba(56, 189, 248, 0.35);
        box-shadow: 0 16px 40px 0 rgba(6, 182, 212, 0.15), var(--glass-highlight);
        transform: translateY(-2px);
    }

    .glass-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, rgba(255, 255, 255, 0.4) 50%, transparent 100%);
    }

    /* Glossy Metric Component */
    .glossy-metric-box {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.06) 0%, rgba(255, 255, 255, 0.015) 100%);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 1rem;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.2);
        position: relative;
        overflow: hidden;
        margin-bottom: 1rem;
    }

    .glossy-metric-box::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, var(--neon-cyan), var(--neon-violet));
    }

    .glossy-metric-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        margin-bottom: 0.4rem;
    }

    .glossy-metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }

    .glossy-metric-sub {
        font-size: 0.8rem;
        font-weight: 500;
        color: #38BDF8;
        margin-top: 0.35rem;
    }

    /* Glossy Risk Badges */
    .badge-critical {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.28) 0%, rgba(185, 28, 28, 0.45) 100%);
        border: 1px solid rgba(248, 113, 113, 0.65);
        color: #FECACA;
        padding: 0.4rem 1.1rem;
        border-radius: 9999px;
        font-weight: 800;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.3);
        text-shadow: 0 0 10px rgba(239, 68, 68, 0.6);
    }

    .badge-high {
        background: linear-gradient(135deg, rgba(249, 115, 22, 0.28) 0%, rgba(194, 65, 12, 0.45) 100%);
        border: 1px solid rgba(251, 146, 60, 0.65);
        color: #FFEDD5;
        padding: 0.4rem 1.1rem;
        border-radius: 9999px;
        font-weight: 800;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        box-shadow: 0 0 20px rgba(249, 115, 22, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.3);
    }

    .badge-moderate {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.25) 0%, rgba(180, 83, 9, 0.4) 100%);
        border: 1px solid rgba(251, 191, 36, 0.6);
        color: #FEF3C7;
        padding: 0.4rem 1.1rem;
        border-radius: 9999px;
        font-weight: 800;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        box-shadow: 0 0 18px rgba(245, 158, 11, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.25);
    }

    .badge-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.25) 0%, rgba(4, 120, 87, 0.4) 100%);
        border: 1px solid rgba(52, 211, 153, 0.6);
        color: #D1FAE5;
        padding: 0.4rem 1.1rem;
        border-radius: 9999px;
        font-weight: 800;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        box-shadow: 0 0 18px rgba(16, 185, 129, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.25);
    }

    /* Circular Glossy Score Gauge */
    .gauge-wrapper {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1.5rem;
        padding: 1.2rem;
        background: var(--glass-card);
        backdrop-filter: blur(24px);
        border: 1px solid var(--glass-border);
        border-radius: 1.25rem;
        box-shadow: var(--glass-highlight);
    }

    .gauge-circle {
        position: relative;
        width: 110px;
        height: 110px;
        border-radius: 50%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        border: 4px solid rgba(255, 255, 255, 0.1);
        box-shadow: inset 0 0 20px rgba(0, 0, 0, 0.6);
    }

    .gauge-number {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        color: #FFFFFF;
        line-height: 1;
    }

    .gauge-scale {
        font-size: 0.75rem;
        color: #94A3B8;
        font-weight: 600;
        text-transform: uppercase;
    }

    /* Streamlit Button Customization for Cyber-Deck HUD */
    .stButton > button {
        border-radius: 0.85rem !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.02em !important;
        padding: 0.55rem 0.85rem !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }

    /* Active Nav Tab (Primary Button) */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.35) 0%, rgba(99, 102, 241, 0.45) 100%) !important;
        border: 1px solid rgba(56, 189, 248, 0.85) !important;
        color: #FFFFFF !important;
        box-shadow: 0 0 24px rgba(6, 182, 212, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.45) !important;
        text-shadow: 0 0 12px rgba(56, 189, 248, 0.8) !important;
        transform: translateY(-1px) !important;
    }

    /* Inactive Nav Tab (Secondary Button) */
    .stButton > button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.035) !important;
        border: 1px solid rgba(255, 255, 255, 0.09) !important;
        color: #94A3B8 !important;
        backdrop-filter: blur(14px) !important;
    }

    .stButton > button[kind="secondary"]:hover {
        background: rgba(56, 189, 248, 0.12) !important;
        border-color: rgba(56, 189, 248, 0.4) !important;
        color: #F1F5F9 !important;
        box-shadow: 0 0 16px rgba(6, 182, 212, 0.25) !important;
        transform: translateY(-2px) !important;
    }

    /* Tactical Quick-Jump Pill Buttons in Sidebar */
    .sidebar-jump-btn > button {
        width: 100% !important;
        text-align: left !important;
        justify-content: flex-start !important;
        margin-bottom: 0.35rem !important;
        font-size: 0.8rem !important;
    }

    /* Streamlit native widget glassmorphism tuning */
    .stSelectbox div[data-baseweb="select"] > div {
        background: rgba(15, 23, 42, 0.7) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 0.75rem !important;
        color: #F8FAFC !important;
        backdrop-filter: blur(16px) !important;
    }

    .stSelectbox div[data-baseweb="select"] > div:hover {
        border-color: rgba(56, 189, 248, 0.5) !important;
        box-shadow: 0 0 16px rgba(56, 189, 248, 0.2) !important;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 1rem !important;
        overflow: hidden !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35) !important;
    }
</style>
"""

st.markdown(GLOSSY_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PLOTLY GLOSSY THEME HELPER
# -----------------------------------------------------------------------------
def apply_glossy_theme(fig: go.Figure, height: int = 400) -> go.Figure:
    """Applies glossy frosted canvas styling to Plotly figures."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        plot_bgcolor="rgba(15, 23, 42, 0.2)",
        height=height,
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#CBD5E1", size=12),
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.06)",
            linecolor="rgba(255, 255, 255, 0.12)",
            tickfont=dict(color="#94A3B8")
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.06)",
            linecolor="rgba(255, 255, 255, 0.12)",
            tickfont=dict(color="#94A3B8")
        ),
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.6)",
            bordercolor="rgba(255, 255, 255, 0.1)",
            borderwidth=1,
            font=dict(color="#E2E8F0")
        )
    )
    return fig

# -----------------------------------------------------------------------------
# CACHED DATA & RESOURCE LOADERS
# -----------------------------------------------------------------------------
@st.cache_data
def load_base_data():
    df = pd.read_csv("data/processed/transactions_engineered.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    with open("data/graph/graph_statistics.json", "r") as f:
        graph_stats = json.load(f)
    with open("data/graph/suspicious_clusters.json", "r") as f:
        clusters = json.load(f)
    benchmark_df = pd.read_csv("models/model_comparison_benchmark.csv")
    return df, graph_stats, clusters, benchmark_df

@st.cache_resource
def load_graph_and_engine():
    G_nx = joblib.load("data/graph/heterogeneous_graph.joblib")
    engine = ExplainabilityEngine()
    pyg_data = torch.load("data/graph/pyg_transaction_graph.pt", weights_only=False)
    
    preds_dict = {}
    if os.path.exists("models/graphsage_predictions.pt"):
        d = torch.load("models/graphsage_predictions.pt", weights_only=False)
        preds_dict["GraphSAGE"] = d["probabilities"].numpy()
    if os.path.exists("models/gcn_predictions.pt"):
        d = torch.load("models/gcn_predictions.pt", weights_only=False)
        preds_dict["GCN"] = d["probabilities"].numpy()
    if os.path.exists("models/gat_predictions.pt"):
        d = torch.load("models/gat_predictions.pt", weights_only=False)
        preds_dict["GAT"] = d["probabilities"].numpy()

    if os.path.exists("models/xgboost.joblib"):
        xgb_m = joblib.load("models/xgboost.joblib")
        from src.feature_engineering import FeaturePipeline
        p = FeaturePipeline.load("models/feature_pipeline.joblib")
        df_feat = pd.read_csv("data/processed/transactions_engineered.csv")
        X = p.transform(df_feat)
        preds_dict["XGBoost"] = xgb_m.predict_proba(X)[:, 1]

    return G_nx, engine, pyg_data, preds_dict

df_tx, graph_stats, suspicious_clusters, benchmark_df = load_base_data()
G_nx, explain_engine, pyg_data, model_predictions = load_graph_and_engine()

# -----------------------------------------------------------------------------
# SIDEBAR TACTICAL COMMAND DECK & HYPER-JUMPS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 0.9rem; padding: 0.5rem 0 1.2rem 0;">
        <div style="width: 48px; height: 48px; border-radius: 12px; background: linear-gradient(135deg, #06B6D4 0%, #6366F1 100%); display: flex; align-items: center; justify-content: center; box-shadow: 0 0 24px rgba(6, 182, 212, 0.45); border: 1px solid rgba(255, 255, 255, 0.3);">
            <span style="font-size: 24px;">🛡️</span>
        </div>
        <div>
            <div style="font-size: 1.35rem; font-weight: 800; letter-spacing: -0.02em; background: linear-gradient(135deg, #FFFFFF 0%, #38BDF8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">GraphGuard</div>
            <div style="font-size: 0.72rem; color: #94A3B8; font-weight: 500; letter-spacing: 0.04em; text-transform: uppercase;">GNN Fraud Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size: 0.75rem; color: #94A3B8; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 0.5rem;'>⚡ TACTICAL HYPER-JUMP DECK</div>", unsafe_allow_html=True)

    # 1-Click Tactical Triage Jump Pills
    if st.button("🚨 Jump: ATO Emergency Alert", use_container_width=True, help="Instantly jump to Account Takeover anomaly investigation for TXN_000010"):
        st.session_state["active_page"] = "🔍 Transaction Investigation"
        st.session_state["search_tx_id"] = "TXN_000010"
        st.rerun()

    if st.button("🕸️ Jump: Device Syndicate #1", use_container_width=True, help="Jump to 2-hop ego-network mesh around top shared device ring"):
        st.session_state["active_page"] = "🌐 Fraud Network Explorer"
        st.rerun()

    if st.button("⚔️ Jump: GNN vs Tree Arena", use_container_width=True, help="Jump to 6-model benchmark comparison matrix"):
        st.session_state["active_page"] = "📈 Model Comparison Benchmark"
        st.rerun()

    if st.button("🔍 Jump: Top 98 Rings Scanner", use_container_width=True, help="Jump to community detection cluster table"):
        st.session_state["active_page"] = "🕸️ Suspicious Clusters & Rings"
        st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.75rem; color: #94A3B8; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 0.4rem;'>ACTIVE INFERENCE ENGINE</div>", unsafe_allow_html=True)
    
    active_model_name = st.selectbox(
        "Active Inference Core",
        options=["GraphSAGE (GNN)", "XGBoost (Tabular ML)", "GAT (GNN Attention)", "GCN (GNN)"],
        index=0,
        label_visibility="collapsed"
    )
    model_key = active_model_name.split(" ")[0]

    st.markdown(f"""
    <div style="background: rgba(6, 182, 212, 0.08); border: 1px solid rgba(6, 182, 212, 0.25); border-radius: 0.75rem; padding: 0.8rem; margin-top: 0.5rem; box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1);">
        <div style="font-size: 0.72rem; color: #38BDF8; font-weight: 700; text-transform: uppercase;">ACTIVE GNN BACKBONE</div>
        <div style="font-size: 0.88rem; color: #FFFFFF; font-weight: 700; margin-top: 0.15rem;">{active_model_name}</div>
        <div style="font-size: 0.72rem; color: #94A3B8; margin-top: 0.25rem;">PR-AUC: <strong>{benchmark_df[benchmark_df['Model'].str.contains(model_key)]['PR-AUC (Primary)'].values[0]:.4f}</strong></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 0.73rem; color: #64748B; line-height: 1.4; padding: 0.2rem;">
        <strong style="color: #94A3B8;">Security Sandbox Notice</strong><br>
        Educational & Portfolio Prototype.<br>
        Presentation thresholds are analytical triage bands, not calibrated regulatory policies.
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# UNIQUE FLOATING CYBERDECK HUD NAVIGATION BAR
# -----------------------------------------------------------------------------
NAV_ITEMS = [
    ("📊 Overview", "📊 Executive Overview", "System Telemetry"),
    ("🔍 Audit Dossier", "🔍 Transaction Investigation", "Single Txn Risk"),
    ("🌐 Graph Radar", "🌐 Fraud Network Explorer", "2-Hop Ego Mesh"),
    ("📈 Model Arena", "📈 Model Comparison Benchmark", "6-Model Showdown"),
    ("🕸️ Syndicates", "🕸️ Suspicious Clusters & Rings", "98 Detected Rings"),
    ("💡 Explainability", "💡 Explainability & SHAP Hub", "Tree SHAP & XAI")
]

# Top HUD Telemetry Ribbon
st.markdown(f"""
<div class="hud-telemetry-ribbon">
    <div style="display: flex; align-items: center; gap: 0.6rem;">
        <span class="pulse-dot"></span>
        <span style="color: #38BDF8; font-weight: 700; letter-spacing: 0.04em;">GRAPHGUARD HUD v1.0</span>
        <span style="color: #64748B;">//</span>
        <span style="color: #10B981; font-weight: 600;">ACTIVE MESH: 17,031 NODES</span>
    </div>
    <div style="display: flex; align-items: center; gap: 1.2rem; color: #94A3B8;">
        <span>EDGES: <strong style="color: #E2E8F0;">{graph_stats['heterogeneous_total_edges']:,}</strong></span>
        <span>HOMOPHILY: <strong style="color: #34D399;">{graph_stats['edge_homophily']:.1%}</strong></span>
        <span>CORE: <strong style="color: #A78BFA;">{active_model_name.split()[0]}</strong></span>
        <span>TEST: <strong style="color: #38BDF8;">1,800 NODES</strong></span>
    </div>
</div>
""", unsafe_allow_html=True)

# Cyberdeck 6-Column Interactive Navigation Bar
nav_cols = st.columns(6)
for i, (short_label, full_page, subtitle) in enumerate(NAV_ITEMS):
    is_active = (st.session_state["active_page"] == full_page)
    with nav_cols[i]:
        if st.button(
            short_label,
            type="primary" if is_active else "secondary",
            key=f"nav_btn_{i}",
            use_container_width=True,
            help=f"Switch console to {full_page} ({subtitle})"
        ):
            st.session_state["active_page"] = full_page
            st.rerun()

current_page = st.session_state["active_page"]

# =============================================================================
# PAGE 1: EXECUTIVE OVERVIEW
# =============================================================================
if current_page == "📊 Executive Overview":
    st.markdown("""
    <div class="live-status-pill">
        <span class="pulse-dot"></span>
        <span>REAL-TIME FINANCIAL GRAPH SURVEILLANCE</span>
    </div>
    <div class="glossy-hero-title">
        <span>Executive Intelligence Console</span>
    </div>
    <div class="glossy-subtitle">
        Bridging transaction-level behavioral anomalies with multi-entity relationship intelligence to expose coordinated money mule networks and bot proxies.
    </div>
    """, unsafe_allow_html=True)

    # Glossy Top Metrics Row
    total_tx = len(df_tx)
    total_fraud = int(df_tx["is_fraud"].sum())
    fraud_rate = total_fraud / total_tx
    total_fraud_volume = df_tx[df_tx["is_fraud"] == 1]["amount"].sum()
    best_pr_auc = benchmark_df["PR-AUC (Primary)"].max()

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="glossy-metric-box">
            <div class="glossy-metric-label">TOTAL TRANSACTIONS</div>
            <div class="glossy-metric-value">{total_tx:,}</div>
            <div class="glossy-metric-sub">100% Validated & Cleaned</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="glossy-metric-box">
            <div class="glossy-metric-label">FRAUD INCIDENTS</div>
            <div class="glossy-metric-value" style="color: #F87171;">{total_fraud:,}</div>
            <div class="glossy-metric-sub">{fraud_rate:.2%} baseline prevalence</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="glossy-metric-box">
            <div class="glossy-metric-label">FRAUD VOLUME EXPOSURE</div>
            <div class="glossy-metric-value" style="color: #FBBF24;">${total_fraud_volume:,.2f}</div>
            <div class="glossy-metric-sub">Across 8 Merchant Sectors</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="glossy-metric-box">
            <div class="glossy-metric-label">BENCHMARK TOP PR-AUC</div>
            <div class="glossy-metric-value" style="color: #38BDF8;">{best_pr_auc:.4f}</div>
            <div class="glossy-metric-sub">GraphSAGE / Random Forest</div>
        </div>
        """, unsafe_allow_html=True)

    # Charts Section
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div style='font-size: 1.05rem; font-weight: 700; color: #FFFFFF; margin-bottom: 0.5rem;'>Merchant Category Risk Breakdown</div>", unsafe_allow_html=True)
        cat_stats = df_tx.groupby("merchant_category").agg(
            total_count=("transaction_id", "count"),
            fraud_count=("is_fraud", "sum"),
        ).reset_index()
        fig_cat = px.bar(
            cat_stats,
            x="merchant_category",
            y=["total_count", "fraud_count"],
            barmode="group",
            labels={"value": "Volume", "merchant_category": "Sector", "variable": "Class"},
            color_discrete_map={"total_count": "#38BDF8", "fraud_count": "#F43F5E"}
        )
        fig_cat = apply_glossy_theme(fig_cat, height=360)
        st.plotly_chart(fig_cat, use_container_width=True)

    with c2:
        st.markdown("<div style='font-size: 1.05rem; font-weight: 700; color: #FFFFFF; margin-bottom: 0.5rem;'>Diurnal 24-Hour Fraud Curves</div>", unsafe_allow_html=True)
        hourly = df_tx.groupby(["hour", "is_fraud"]).size().unstack(fill_value=0).reset_index()
        fig_hour = go.Figure()
        fig_hour.add_trace(go.Scatter(
            x=hourly["hour"], y=hourly[0], mode="lines+markers", name="Legitimate (0)",
            line=dict(color="#10B981", width=2.5), marker=dict(size=5)
        ))
        fig_hour.add_trace(go.Scatter(
            x=hourly["hour"], y=hourly[1], mode="lines+markers", name="Fraud Spikes (1)",
            line=dict(color="#F43F5E", width=3, dash="dot"), marker=dict(size=6)
        ))
        fig_hour.update_layout(xaxis_title="Hour of Day (0–23)", yaxis_title="Transactions Count")
        fig_hour = apply_glossy_theme(fig_hour, height=360)
        st.plotly_chart(fig_hour, use_container_width=True)

    # Topology Highlights
    st.markdown("<div style='font-size: 1.15rem; font-weight: 800; color: #FFFFFF; margin-top: 1rem; margin-bottom: 0.8rem;'>Financial Relationship Graph Topology</div>", unsafe_allow_html=True)
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.markdown(f"""
        <div class="glass-card" style="padding: 1.1rem;">
            <div style="font-size: 0.75rem; color: #94A3B8; font-weight: 700;">HETEROGENEOUS NODES</div>
            <div style="font-size: 1.45rem; font-weight: 800; color: #FFFFFF;">{graph_stats['heterogeneous_total_nodes']:,}</div>
            <div style="font-size: 0.75rem; color: #64748B; margin-top: 0.2rem;">Tx, Cust, Dev, IP, Merch</div>
        </div>
        """, unsafe_allow_html=True)
    with g2:
        st.markdown(f"""
        <div class="glass-card" style="padding: 1.1rem;">
            <div style="font-size: 0.75rem; color: #94A3B8; font-weight: 700;">RELATIONAL EDGES</div>
            <div style="font-size: 1.45rem; font-weight: 800; color: #FFFFFF;">{graph_stats['heterogeneous_total_edges']:,}</div>
            <div style="font-size: 0.75rem; color: #64748B; margin-top: 0.2rem;">Multi-entity relational links</div>
        </div>
        """, unsafe_allow_html=True)
    with g3:
        st.markdown(f"""
        <div class="glass-card" style="padding: 1.1rem;">
            <div style="font-size: 0.75rem; color: #94A3B8; font-weight: 700;">EDGE HOMOPHILY</div>
            <div style="font-size: 1.45rem; font-weight: 800; color: #34D399;">{graph_stats['edge_homophily']:.1%}</div>
            <div style="font-size: 0.75rem; color: #64748B; margin-top: 0.2rem;">High intra-cluster cohesion</div>
        </div>
        """, unsafe_allow_html=True)
    with g4:
        st.markdown(f"""
        <div class="glass-card" style="padding: 1.1rem;">
            <div style="font-size: 0.75rem; color: #94A3B8; font-weight: 700;">MEAN NODE DEGREE</div>
            <div style="font-size: 1.45rem; font-weight: 800; color: #38BDF8;">{graph_stats['degree_mean']:.2f}</div>
            <div style="font-size: 0.75rem; color: #64748B; margin-top: 0.2rem;">0 isolated transaction nodes</div>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# PAGE 2: TRANSACTION INVESTIGATION
# =============================================================================
elif current_page == "🔍 Transaction Investigation":
    st.markdown("""
    <div class="live-status-pill">
        <span class="pulse-dot"></span>
        <span>INVESTIGATION DOSSIER & MULTI-TIER RISK AUDIT</span>
    </div>
    <div class="glossy-hero-title">Transaction Investigation Console</div>
    <div class="glossy-subtitle">
        Deep-dive into individual financial transactions with calibrated 0–100 risk scoring, connected entity fingerprints, and automated relational red flags.
    </div>
    """, unsafe_allow_html=True)

    # Search Bar & Filter Options
    f1, f2, f3 = st.columns([1, 1, 2])
    with f1:
        filter_label = st.selectbox("Ground Truth Filter", ["All Transactions", "Fraud Only (1)", "Legitimate Only (0)"])
    with f2:
        filter_scenario = st.selectbox("Fraud Pattern Filter", ["All Patterns"] + list(df_tx["fraud_scenario"].unique()))

    filtered_df = df_tx.copy()
    if filter_label == "Fraud Only (1)":
        filtered_df = filtered_df[filtered_df["is_fraud"] == 1]
    elif filter_label == "Legitimate Only (0)":
        filtered_df = filtered_df[filtered_df["is_fraud"] == 0]
    if filter_scenario != "All Patterns":
        filtered_df = filtered_df[filtered_df["fraud_scenario"] == filter_scenario]

    # Pre-select hyper-jump ID if set
    options_list = filtered_df["transaction_id"].head(500).tolist()
    default_idx = 0
    if st.session_state["search_tx_id"] and st.session_state["search_tx_id"] in options_list:
        default_idx = options_list.index(st.session_state["search_tx_id"])

    selected_tx_id = st.selectbox(
        "Select Transaction Identifier for Deep Inspection:",
        options=options_list,
        index=default_idx
    )

    if selected_tx_id:
        idx = explain_engine.tx_id_to_idx[selected_tx_id]
        prob = float(model_predictions.get(model_key, np.zeros(len(df_tx)))[idx])
        dossier = explain_engine.get_unified_investigation(selected_tx_id, predicted_prob=prob)
        raw = dossier["raw_attributes"]
        score = dossier["risk_score"]
        tier = dossier["risk_tier"]

        # Circular Gauge Banner & Status Card
        g_col, s_col = st.columns([1, 2])
        with g_col:
            border_color = "#EF4444" if tier == "Critical" else ("#F97316" if tier == "High" else ("#F59E0B" if tier == "Moderate" else "#10B981"))
            glow_color = f"rgba({239 if tier=='Critical' else (249 if tier=='High' else (245 if tier=='Moderate' else 16))}, {68 if tier=='Critical' else (115 if tier=='High' else (158 if tier=='Moderate' else 185))}, {68 if tier=='Critical' else (22 if tier=='High' else (11 if tier=='Moderate' else 129))}, 0.35)"
            
            st.markdown(f"""
            <div class="gauge-wrapper">
                <div class="gauge-circle" style="border-color: {border_color}; box-shadow: 0 0 30px {glow_color}, inset 0 0 20px rgba(0,0,0,0.7);">
                    <div class="gauge-number" style="color: {border_color};">{score}</div>
                    <div class="gauge-scale">OUT OF 100</div>
                </div>
                <div>
                    <div style="font-size: 0.75rem; color: #94A3B8; font-weight: 700; text-transform: uppercase;">PRESENTATION BAND</div>
                    <div style="margin-top: 0.35rem;">
                        <span class="badge-{tier.lower()}">{tier.upper()} RISK</span>
                    </div>
                    <div style="font-size: 0.78rem; color: #CBD5E1; margin-top: 0.5rem;">
                        Probability: <strong style="color: #FFFFFF;">{prob:.4f}</strong>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with s_col:
            is_fr = raw.get("is_fraud")
            st.markdown(f"""
            <div class="glass-card" style="height: 100%; display: flex; flex-direction: column; justify-content: center;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                    <span style="font-size: 0.8rem; color: #94A3B8; font-weight: 700; text-transform: uppercase;">TRANSACTION ID: <span style="color: #38BDF8;">{selected_tx_id}</span></span>
                    <span style="font-size: 0.8rem; font-weight: 700; color: {'#F87171' if is_fr==1 else '#34D399'};">
                        { '🚨 CONFIRMED FRAUD' if is_fr==1 else '✅ VERIFIED LEGITIMATE' }
                    </span>
                </div>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem;">
                    <div>
                        <div style="font-size: 0.72rem; color: #64748B; font-weight: 600;">AMOUNT</div>
                        <div style="font-size: 1.1rem; font-weight: 800; color: #FFFFFF;">${raw.get('amount', 0):,.2f}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.72rem; color: #64748B; font-weight: 600;">TIMESTAMP</div>
                        <div style="font-size: 0.85rem; font-weight: 600; color: #E2E8F0;">{str(raw.get('timestamp'))[:16]}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.72rem; color: #64748B; font-weight: 600;">CHANNEL</div>
                        <div style="font-size: 0.85rem; font-weight: 600; color: #38BDF8;">{raw.get('channel', 'N/A').upper()}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.72rem; color: #64748B; font-weight: 600;">PATTERN TYPE</div>
                        <div style="font-size: 0.85rem; font-weight: 600; color: #A78BFA;">{raw.get('fraud_scenario', 'N/A')}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Relational Alerts
        st.markdown("<div style='font-size: 1.1rem; font-weight: 800; color: #FFFFFF; margin-top: 1.2rem; margin-bottom: 0.6rem;'>Multi-Entity Relational Red Flags</div>", unsafe_allow_html=True)
        red_flags = dossier["graph_explanation"].get("relational_red_flags", [])
        if red_flags:
            for flag in red_flags:
                st.markdown(f"""
                <div style="background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(248, 113, 113, 0.4); border-radius: 0.75rem; padding: 0.75rem 1rem; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.75rem; box-shadow: 0 4px 16px rgba(239, 68, 68, 0.15);">
                    <span style="font-size: 1.1rem;">⚠️</span>
                    <span style="font-size: 0.88rem; color: #FECACA; font-weight: 500;">{flag}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(52, 211, 153, 0.4); border-radius: 0.75rem; padding: 0.75rem 1rem; display: flex; align-items: center; gap: 0.75rem;">
                <span style="font-size: 1.1rem;">🛡️</span>
                <span style="font-size: 0.88rem; color: #D1FAE5; font-weight: 500;">No multi-account entity sharing or velocity anomalies detected in 2-hop radius.</span>
            </div>
            """, unsafe_allow_html=True)

        # Connected Entities Table
        st.markdown("<div style='font-size: 1.1rem; font-weight: 800; color: #FFFFFF; margin-top: 1.5rem; margin-bottom: 0.6rem;'>Connected Entity Fingerprints</div>", unsafe_allow_html=True)
        entities_data = []
        for ent_type, details in dossier["graph_explanation"].get("connected_entities", {}).items():
            entities_data.append({
                "Entity Type": ent_type.capitalize(),
                "Identifier": details["entity_id"],
                "Total Linked Transactions": details["total_shared_transactions"] + 1,
                "Linked Fraud Transactions": details["fraudulent_shared_transactions"],
                "Fraud Prevalence on Entity": f"{details['fraud_rate_on_entity']:.1%}"
            })
        if entities_data:
            ent_df = pd.DataFrame(entities_data)
            st.dataframe(ent_df, use_container_width=True, hide_index=True)

# =============================================================================
# PAGE 3: FRAUD NETWORK EXPLORER
# =============================================================================
elif current_page == "🌐 Fraud Network Explorer":
    st.markdown("""
    <div class="live-status-pill">
        <span class="pulse-dot"></span>
        <span>EGO-NETWORK SUBGRAPH TOPOLOGY</span>
    </div>
    <div class="glossy-hero-title">Multi-Entity Fraud Network Explorer</div>
    <div class="glossy-subtitle">
        Visualizing 2-hop relational neighborhoods around investigated transactions to uncover account clustering, device pooling, and shared proxy botnets.
    </div>
    """, unsafe_allow_html=True)

    tx_options = df_tx[df_tx["is_fraud"] == 1]["transaction_id"].head(50).tolist() + df_tx["transaction_id"].head(50).tolist()
    chosen_tx = st.selectbox("Select Target Center Transaction:", tx_options, index=0)

    if chosen_tx:
        sub_g = extract_transaction_subgraph(G_nx, chosen_tx, depth=2)
        
        st.markdown(f"""
        <div style="display: flex; gap: 1rem; margin-bottom: 1rem;">
            <div class="glass-card" style="padding: 0.8rem 1.2rem; margin: 0; display: inline-flex; align-items: center; gap: 0.5rem;">
                <span style="color: #94A3B8; font-size: 0.8rem;">Sub-Network Nodes:</span>
                <strong style="color: #38BDF8; font-size: 1.05rem;">{sub_g.number_of_nodes()}</strong>
            </div>
            <div class="glass-card" style="padding: 0.8rem 1.2rem; margin: 0; display: inline-flex; align-items: center; gap: 0.5rem;">
                <span style="color: #94A3B8; font-size: 0.8rem;">Sub-Network Edges:</span>
                <strong style="color: #A78BFA; font-size: 1.05rem;">{sub_g.number_of_edges()}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        pos = nx.spring_layout(sub_g, seed=42, k=0.35)

        edge_x, edge_y = [], []
        for u, v in sub_g.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.2, color="rgba(148, 163, 184, 0.25)"),
            hoverinfo="none",
            mode="lines"
        )

        node_x, node_y, node_color, node_text, node_size = [], [], [], [], []
        color_map = {
            "customer": "#38BDF8",
            "device": "#A855F7",
            "ip_address": "#F59E0B",
            "merchant": "#06B6D4",
            "transaction_legit": "#10B981",
            "transaction_fraud": "#F43F5E"
        }

        for node, data in sub_g.nodes(data=True):
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            ntype = data.get("node_type", "unknown")
            if ntype == "transaction":
                is_fr = data.get("is_fraud", 0)
                col = color_map["transaction_fraud"] if is_fr == 1 else color_map["transaction_legit"]
                sz = 22 if node == chosen_tx else 14
                lbl = f"TXN: {node}<br>Amount: ${data.get('amount', 0):.2f}<br>Fraud: {'YES' if is_fr==1 else 'NO'}"
            else:
                col = color_map.get(ntype, "#CBD5E1")
                sz = 16
                lbl = f"{ntype.upper()}: {node}"

            node_color.append(col)
            node_size.append(sz)
            node_text.append(lbl)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            hoverinfo="text",
            text=[n if n == chosen_tx or sub_g.nodes[n].get("node_type") != "transaction" else "" for n in sub_g.nodes()],
            textposition="top center",
            hovertext=node_text,
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color="#0F172A")
            )
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(text=f"2-Hop Graph Visualization for {chosen_tx}", font=dict(color="#FFFFFF", size=16)),
                showlegend=False,
                hovermode="closest",
                margin=dict(b=10, l=10, r=10, t=50),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=620
            )
        )
        fig = apply_glossy_theme(fig, height=620)
        st.plotly_chart(fig, use_container_width=True)

        # Legend Pills
        st.markdown("""
        <div style="display: flex; gap: 0.8rem; flex-wrap: wrap; margin-top: 0.5rem;">
            <span style="background: rgba(244, 63, 94, 0.15); border: 1px solid #F43F5E; color: #FECDD3; padding: 0.3rem 0.8rem; border-radius: 9999px; font-size: 0.8rem; font-weight: 700;">🔴 Fraud Txn</span>
            <span style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10B981; color: #A7F3D0; padding: 0.3rem 0.8rem; border-radius: 9999px; font-size: 0.8rem; font-weight: 700;">🟢 Legit Txn</span>
            <span style="background: rgba(56, 189, 248, 0.15); border: 1px solid #38BDF8; color: #BAE6FD; padding: 0.3rem 0.8rem; border-radius: 9999px; font-size: 0.8rem; font-weight: 700;">🔵 Customer</span>
            <span style="background: rgba(168, 85, 247, 0.15); border: 1px solid #A855F7; color: #E9D5FF; padding: 0.3rem 0.8rem; border-radius: 9999px; font-size: 0.8rem; font-weight: 700;">🟣 Device</span>
            <span style="background: rgba(245, 158, 11, 0.15); border: 1px solid #F59E0B; color: #FDE68A; padding: 0.3rem 0.8rem; border-radius: 9999px; font-size: 0.8rem; font-weight: 700;">🟡 IP Proxy</span>
            <span style="background: rgba(6, 182, 212, 0.15); border: 1px solid #06B6D4; color: #A5F3FC; padding: 0.3rem 0.8rem; border-radius: 9999px; font-size: 0.8rem; font-weight: 700;">🔷 Merchant</span>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# PAGE 4: MODEL COMPARISON BENCHMARK
# =============================================================================
elif current_page == "📈 Model Comparison Benchmark":
    st.markdown("""
    <div class="live-status-pill">
        <span class="pulse-dot"></span>
        <span>STANDARDIZED TEST SPLIT BENCHMARK</span>
    </div>
    <div class="glossy-hero-title">Model Comparison Benchmark</div>
    <div class="glossy-subtitle">
        Rigorous, leakage-safe evaluation of Traditional Classifiers vs Graph Neural Networks on 1,800 chronologically subsequent transactions (1.33% fraud prevalence).
    </div>
    """, unsafe_allow_html=True)

    # Formatted Benchmark Table
    st.dataframe(
        benchmark_df.style.format({
            "PR-AUC (Primary)": "{:.4f}",
            "ROC-AUC": "{:.4f}",
            "F1-Score": "{:.4f}",
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "Threshold": "{:.3f}"
        }).highlight_max(subset=["PR-AUC (Primary)", "ROC-AUC", "F1-Score", "Recall"], color="rgba(6, 182, 212, 0.35)"),
        use_container_width=True
    )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    bc1, bc2 = st.columns(2)
    with bc1:
        st.markdown("<div style='font-size: 1.05rem; font-weight: 700; color: #FFFFFF; margin-bottom: 0.5rem;'>Combined Precision-Recall Benchmark</div>", unsafe_allow_html=True)
        if os.path.exists("reports/figures/combined_pr_curves.png"):
            st.image("reports/figures/combined_pr_curves.png", use_container_width=True)
    with bc2:
        st.markdown("<div style='font-size: 1.05rem; font-weight: 700; color: #FFFFFF; margin-bottom: 0.5rem;'>Combined ROC Benchmark Curves</div>", unsafe_allow_html=True)
        if os.path.exists("reports/figures/combined_roc_curves.png"):
            st.image("reports/figures/combined_roc_curves.png", use_container_width=True)

    st.markdown("""
    <div class="glass-card" style="margin-top: 1rem;">
        <div style="font-size: 1.05rem; font-weight: 800; color: #FFFFFF; margin-bottom: 0.5rem;">Key Architectural Insights & Trade-Offs</div>
        <ul style="color: #CBD5E1; font-size: 0.9rem; line-height: 1.6; padding-left: 1.2rem; margin: 0;">
            <li><strong>GraphSAGE (0.9966 PR-AUC, 100% Recall):</strong> Localized mean neighborhood sampling accurately pulled fraud signals across device/IP rings without diluting minority labels. Achieved <strong>zero false negatives</strong> on test data.</li>
            <li><strong>XGBoost (0.9388 F1, 92.0% Precision):</strong> Gradient boosting showed highest precision on rolling temporal & velocity tabular features.</li>
            <li><strong>GCN & GAT Over-Smoothing:</strong> Global Laplacian convolution without neighborhood sampling caused rare fraud representations to over-smooth into dense legitimate transaction neighborhoods.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# PAGE 5: SUSPICIOUS CLUSTERS & RINGS
# =============================================================================
elif current_page == "🕸️ Suspicious Clusters & Rings":
    st.markdown("""
    <div class="live-status-pill">
        <span class="pulse-dot"></span>
        <span>SYNDICATE CLUSTER DISCOVERY ENGINE</span>
    </div>
    <div class="glossy-hero-title">Suspicious Entity Clusters & Rings</div>
    <div class="glossy-subtitle">
        Automated graph community detection identifying shared hardware identifiers, proxy IP botnets, and multi-account mule syndicates.
    </div>
    """, unsafe_allow_html=True)

    c_metric1, c_metric2, c_metric3 = st.columns(3)
    with c_metric1:
        st.markdown(f"""
        <div class="glossy-metric-box">
            <div class="glossy-metric-label">DETECTED HIGH-RISK CLUSTERS</div>
            <div class="glossy-metric-value" style="color: #F87171;">{len(suspicious_clusters)}</div>
            <div class="glossy-metric-sub">Multi-account syndicates</div>
        </div>
        """, unsafe_allow_html=True)
    with c_metric2:
        dev_clusters = sum(1 for c in suspicious_clusters if c["entity_type"] == "device")
        st.markdown(f"""
        <div class="glossy-metric-box">
            <div class="glossy-metric-label">SHARED DEVICE RINGS</div>
            <div class="glossy-metric-value" style="color: #A78BFA;">{dev_clusters}</div>
            <div class="glossy-metric-sub">Hardware pooling schemes</div>
        </div>
        """, unsafe_allow_html=True)
    with c_metric3:
        ip_clusters = sum(1 for c in suspicious_clusters if c["entity_type"] == "ip_address")
        st.markdown(f"""
        <div class="glossy-metric-box">
            <div class="glossy-metric-label">BOT PROXY CLUSTERS</div>
            <div class="glossy-metric-value" style="color: #FBBF24;">{ip_clusters}</div>
            <div class="glossy-metric-sub">High-velocity automated bursts</div>
        </div>
        """, unsafe_allow_html=True)

    entity_type_filter = st.selectbox("Filter by Entity Class:", ["All Entities", "Hardware Device (device)", "Network IP (ip_address)"])
    filtered_clusters = suspicious_clusters
    if "device" in entity_type_filter:
        filtered_clusters = [c for c in suspicious_clusters if c["entity_type"] == "device"]
    elif "ip_address" in entity_type_filter:
        filtered_clusters = [c for c in suspicious_clusters if c["entity_type"] == "ip_address"]

    st.markdown("<div style='font-size: 1.05rem; font-weight: 700; color: #FFFFFF; margin-top: 1rem; margin-bottom: 0.5rem;'>Top Suspicious Syndicates for Investigator Review</div>", unsafe_allow_html=True)
    
    cluster_table = []
    for c in filtered_clusters[:30]:
        cluster_table.append({
            "Entity ID": c["entity_id"],
            "Class": c["entity_type"].upper(),
            "Total Transactions": c["total_connected_transactions"],
            "Fraudulent Transactions": c["fraud_transactions_count"],
            "Fraud Concentration": f"{c['fraud_rate']:.1%}",
            "Fraud Exposure ($)": f"${c['total_fraud_amount']:,.2f}"
        })
    st.dataframe(pd.DataFrame(cluster_table), use_container_width=True, hide_index=True)

# =============================================================================
# PAGE 6: EXPLAINABILITY & SHAP HUB
# =============================================================================
elif current_page == "💡 Explainability & SHAP Hub":
    st.markdown("""
    <div class="live-status-pill">
        <span class="pulse-dot"></span>
        <span>SHAPLEY VALUE ATTRIBUTION & REASONING</span>
    </div>
    <div class="glossy-hero-title">Explainability & Feature Attribution Hub</div>
    <div class="glossy-subtitle">
        Transparently decomposing black-box machine learning decisions into interpretable tabular risk escalators and structural graph drivers.
    </div>
    """, unsafe_allow_html=True)

    tx_list = df_tx[df_tx["is_fraud"] == 1]["transaction_id"].head(20).tolist() + df_tx["transaction_id"].head(10).tolist()
    inspect_tx = st.selectbox("Select Transaction Identifier for Feature Attribution:", tx_list, index=0)

    if inspect_tx:
        shap_res = explain_engine.explain_tabular_shap(inspect_tx, top_k=8)
        factors = shap_res.get("top_factors", [])

        if factors:
            df_factors = pd.DataFrame(factors)
            fig_shap = px.bar(
                df_factors,
                x="shap_value",
                y="feature",
                orientation="h",
                color="direction",
                title=f"SHAP Marginal Log-Odds Impact for {inspect_tx}",
                color_discrete_map={"Risk Escalator": "#F43F5E", "Risk Mitigator": "#10B981"},
                labels={"shap_value": "SHAP Contribution to Fraud Log-Odds", "feature": "Engineered Feature"}
            )
            fig_shap.update_layout(yaxis=dict(autorange="reversed"))
            fig_shap = apply_glossy_theme(fig_shap, height=420)
            st.plotly_chart(fig_shap, use_container_width=True)

        st.markdown("""
        <div class="glass-card" style="margin-top: 1rem;">
            <div style="font-size: 1.05rem; font-weight: 800; color: #FFFFFF; margin-bottom: 0.5rem;">Analyst Explanation Methodology</div>
            <p style="color: #CBD5E1; font-size: 0.88rem; line-height: 1.6; margin: 0;">
                • <strong>Tree SHAP (SHapley Additive exPlanations):</strong> Computes exact game-theoretic marginal contributions of tabular features. Positive SHAP values indicate risk escalators (such as off-hour spikes or shared device reuse), while negative values mitigate risk (habitual dining transactions).<br>
                • <strong>Analyst Prudence:</strong> Feature attributions are statistical indicators for investigator triage, not absolute proof of criminal wrongdoing.
            </p>
        </div>
        """, unsafe_allow_html=True)

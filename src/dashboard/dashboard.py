"""
Vietnam Fraud Detector — Streamlit Dashboard.

4-page analytics dashboard for fraud analysts:
1. Live Scoring — manual transaction scoring with SHAP waterfall
2. Model Performance — metrics, PR curve, confusion matrix
3. Fraud Analytics — typology breakdown, hourly heatmap, amount distributions
4. System Status — API health, drift monitoring

Usage:
    1. Start the API first:  uvicorn src.api.app:app --port 8000
    2. Start dashboard:      streamlit run src/dashboard/dashboard.py
"""
import json
import os
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import random
import time
from datetime import datetime

# ─── Config ──────────────────────────────────────────────────────────────
API_BASE = os.environ.get("API_URL", "http://127.0.0.1:8000")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

st.set_page_config(
    page_title="Vietnam Fraud Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Sidebar Navigation ─────────────────────────────────────────────────
st.sidebar.title("🔍 Vietnam Fraud Detector")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🎯 Live Scoring", "📊 Model Performance", "📈 Fraud Analytics", "⚙️ System Status"],
)


# ─── Helper: call API ───────────────────────────────────────────────────
def api_get(path):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=5)
        return r.json() if r.ok else None
    except Exception:
        return None


def api_post(path, data):
    try:
        r = requests.post(f"{API_BASE}{path}", json=data, timeout=10)
        return r.json() if r.ok else None
    except Exception:
        return None


# ─── Load model info once ────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_model_info():
    return api_get("/model-info")


@st.cache_data(ttl=60)
def load_health():
    return api_get("/health")


def get_random_tx():
    """Generates a random transaction dictionary for the UI."""
    banks = ["VCB", "TCB", "BIDV", "VPB", "MBB", "ACB", "MOMO", "ZALOPAY", "VNPAY"]
    tx_types = ["P2P_TRANSFER", "E_COMMERCE", "QR_PAYMENT", "BILL_PAYMENT"]
    
    # 30% chance of a 'suspicious' looking case
    is_suspicious = random.random() < 0.3
    
    if is_suspicious:
        return {
            "amount": random.choice([35000, 50000, 9500000, 12000000]),
            "sender": f"VN_{random.randint(100, 500)}",
            "receiver": f"VN_{random.randint(600, 999)}",
            "bank": random.choice(banks),
            "tx_type": "P2P_TRANSFER" if random.random() > 0.5 else random.choice(tx_types),
            "biometric": False if random.random() > 0.2 else True,
            "device": f"dev_{random.randint(1000, 9999)}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    else:
        return {
            "amount": random.randint(100000, 2000000),
            "sender": f"VN_{random.randint(1000, 9999)}",
            "receiver": f"VN_{random.randint(1000, 9999)}",
            "bank": random.choice(banks),
            "tx_type": random.choice(tx_types),
            "biometric": True if random.random() > 0.1 else False,
            "device": f"dev_{random.randint(100, 999)}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }


# ═════════════════════════════════════════════════════════════════════════
# PAGE 1: LIVE SCORING
# ═════════════════════════════════════════════════════════════════════════
if page == "🎯 Live Scoring":
    st.title("🎯 Live Transaction Scoring")
    st.markdown("Submit a raw transaction and get real-time fraud scoring with SHAP explanations.")

    # Initialize session state for transaction details if empty
    if "tx_in" not in st.session_state:
        st.session_state.tx_in = get_random_tx()

    # Buttons for new transaction types
    c_btn1, c_btn2, _ = st.columns([1, 1, 2])
    if c_btn1.button("🎲 New Random Transaction"):
        st.session_state.tx_in = get_random_tx()
        st.rerun()
    if c_btn2.button("🧹 New Custom Transaction"):
        st.session_state.tx_in = {
            "amount": 0, "sender": "", "receiver": "", "bank": "VCB",
            "tx_type": "P2P_TRANSFER", "biometric": False, "device": "", 
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        st.rerun()

    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Transaction Details")
        amount = st.number_input("Amount (VND)", min_value=0, max_value=50_000_000, value=st.session_state.tx_in["amount"], step=1000)
        sender = st.text_input("Sender CIF", value=st.session_state.tx_in["sender"])
        receiver = st.text_input("Receiver CIF", value=st.session_state.tx_in["receiver"])
        
        # Selectbox indices
        banks = ["VCB", "TCB", "BIDV", "VPB", "MBB", "ACB", "MOMO", "ZALOPAY", "VNPAY"]
        tx_types = ["P2P_TRANSFER", "E_COMMERCE", "QR_PAYMENT", "BILL_PAYMENT"]
        
        bank_idx = banks.index(st.session_state.tx_in["bank"]) if st.session_state.tx_in["bank"] in banks else 0
        tx_type_idx = tx_types.index(st.session_state.tx_in["tx_type"]) if st.session_state.tx_in["tx_type"] in tx_types else 0

        bank = st.selectbox("Receiver Bank", banks, index=bank_idx)
        tx_type = st.selectbox("Transaction Type", tx_types, index=tx_type_idx)
        biometric = st.checkbox("Biometric Verified", value=st.session_state.tx_in["biometric"])
        device = st.text_input("Device MAC Hash", value=st.session_state.tx_in["device"])
        timestamp = st.text_input("Timestamp (YYYY-MM-DD HH:MM:SS)", value=st.session_state.tx_in["timestamp"])

        score_btn = st.button("🔍 Score Transaction", type="primary", width="stretch")

    with col2:
        st.subheader("Scoring Result")
        if score_btn:
            payload = {
                "amount_vnd": amount,
                "sender_cif": sender,
                "receiver_cif": receiver,
                "receiver_bank_code": bank,
                "transaction_type": tx_type,
                "is_biometric_verified": biometric,
                "device_mac_hash": device,
                "timestamp": timestamp,
            }

            with st.spinner("Scoring..."):
                result = api_post("/score", payload)

            if result:
                # Decision badge
                decision = result["decision"]
                color = {"approve": "green", "flag": "orange", "block": "red"}[decision]
                st.markdown(
                    f"### Decision: :{color}[{decision.upper()}]"
                )

                # Scores
                m1, m2, m3 = st.columns(3)
                m1.metric("Ensemble Score", f"{result['fraud_score']:.4f}")
                m2.metric("XGBoost Score", f"{result['xgboost_score']:.4f}")
                m3.metric("AE Anomaly", f"{result['ae_anomaly_score']:.6f}")

                st.markdown(f"**Confidence**: {result['confidence']}")
                st.markdown(f"**Transaction ID**: `{result['transaction_id']}`")
 
                # Diagnostic Signals
                st.subheader("Diagnostic Signals")
                diag = result.get("diagnostics", {})
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("1h Tx Count", diag.get("tx_1h", "N/A"))
                d2.metric("24h Tx Count", diag.get("tx_24h", "N/A"))
                d3.metric("24h Receivers", diag.get("receivers_24h", "N/A"))
                d4.metric("24h Devices", diag.get("devices_24h", "N/A"))
                
                d5, d6, d7, d8 = st.columns(4)
                d5.metric("Pair History", diag.get("pair_count", "N/A"))
                d6.metric("Bank Diversity", diag.get("bank_diversity", "N/A"))
                d7.metric("Min Since Last", f"{diag.get('time_since_last', 0)/60:.1f}")
                d8.metric("Is Night", "Yes" if diag.get("is_night") else "No")

                # SHAP waterfall chart
                st.subheader("Feature Contributions (SHAP)")
                features = result["top_features"]
                names = [f["feature"] for f in features]
                values = [f["contribution"] for f in features]
                colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in values]
                descriptions = [f["description"] for f in features]

                fig = go.Figure(go.Bar(
                    x=values,
                    y=names,
                    orientation="h",
                    marker_color=colors,
                    text=[f"{v:+.4f}" for v in values],
                    textposition="auto",
                    hovertext=[f"{n}: {d}" for n, d in zip(names, descriptions)],
                ))
                fig.update_layout(
                    title="Top 5 SHAP Feature Contributions",
                    xaxis_title="SHAP Value (→ fraud / ← legitimate)",
                    yaxis=dict(autorange="reversed"),
                    height=350,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig, width="stretch")

                # Explanation text
                st.markdown("**Detailed Explanations:**")
                for f in features:
                    icon = "🔴" if f["direction"] == "fraud" else "🟢"
                    st.markdown(
                        f"- {icon} **{f['feature']}** = `{f['value']}` → "
                        f"`{f['contribution']:+.4f}` — {f['description']}"
                    )
            else:
                st.error("❌ API unreachable. Make sure the FastAPI server is running on port 8000.")


# ═════════════════════════════════════════════════════════════════════════
# PAGE 2: MODEL PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")

    info = load_model_info()
    if not info:
        st.error("❌ Cannot reach API. Start the server first.")
        st.stop()

    metrics = info["metrics"]

    # Summary cards
    st.subheader("Model Comparison")
    cols = st.columns(4)
    for i, (name, data) in enumerate(metrics.items()):
        with cols[i]:
            st.markdown(f"#### {name.replace('_', ' ').title()}")
            if name == info["champion"].lower():
                st.markdown("👑 **Champion**")
            st.metric("F1", f"{data['f1']:.4f}")
            st.metric("PR-AUC", f"{data['pr_auc']:.4f}")
            st.metric("Precision", f"{data['precision']:.4f}")
            st.metric("Recall", f"{data['recall']:.4f}")

    # Comparison bar chart
    st.subheader("Head-to-Head Comparison")
    model_names = list(metrics.keys())
    metric_names = ["precision", "recall", "f1", "pr_auc"]

    fig = go.Figure()
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
    for i, metric in enumerate(metric_names):
        fig.add_trace(go.Bar(
            name=metric.upper().replace("_", "-"),
            x=model_names,
            y=[metrics[m].get(metric, 0) for m in model_names],
            marker_color=colors[i],
        ))
    fig.update_layout(barmode="group", height=400, yaxis=dict(range=[0, 1.05]))
    st.plotly_chart(fig, width="stretch")

    # Per-typology detection
    st.subheader("Per-Typology Detection (XGBoost)")
    xgb_typo = metrics.get("xgboost", {}).get("typology", {})
    if xgb_typo:
        typo_data = []
        for name, data in xgb_typo.items():
            typo_data.append({
                "Fraud Type": name.replace("_", " ").title(),
                "Total": data["count"],
                "Detected": data["detected"],
                "Recall": f"{data['recall'] * 100:.2f}%",
                "Missed": data["count"] - data["detected"],
            })
        st.dataframe(pd.DataFrame(typo_data), width="stretch", hide_index=True)

    # Ensemble info
    st.subheader("Ensemble Configuration")
    c1, c2 = st.columns(2)
    c1.markdown(f"**Weights**: XGBoost {info['ensemble_weights']['xgboost']*100:.0f}% / "
                f"AE {info['ensemble_weights']['autoencoder']*100:.0f}%")
    c2.markdown(f"**AE Anomaly Threshold**: `{info['ae_threshold']:.6f}` (95th percentile)")


# ═════════════════════════════════════════════════════════════════════════
# PAGE 3: FRAUD ANALYTICS
# ═════════════════════════════════════════════════════════════════════════
elif page == "📈 Fraud Analytics":
    st.title("📈 Fraud Analytics")
    st.markdown("Detailed analysis of the training dataset's fraud patterns.")

    # Load processed features
    features_path = os.path.join(ROOT, "data", "processed", "features_550k.csv")
    if not os.path.exists(features_path):
        st.error(f"Features file not found: {features_path}")
        st.stop()

    @st.cache_data
    def load_features():
        df = pd.read_csv(features_path, usecols=[
            "amount_vnd", "is_fraud", "fraud_type", "hour",
            "is_night", "is_biometric_verified", "transaction_type",
            "receiver_bank_code", "pair_tx_count",
        ])
        return df

    df = load_features()
    fraud = df[df["is_fraud"] == 1]
    legit = df[df["is_fraud"] == 0]

    # Overview metrics
    st.subheader("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{len(df):,}")
    c2.metric("Fraud Transactions", f"{len(fraud):,}")
    c3.metric("Fraud Rate", f"{len(fraud)/len(df)*100:.1f}%")
    c4.metric("Fraud Types", fraud["fraud_type"].nunique())

    # Typology breakdown
    st.subheader("Fraud by Typology")
    col1, col2 = st.columns([1, 1])
    with col1:
        typo_counts = fraud["fraud_type"].value_counts()
        fig = px.pie(
            values=typo_counts.values,
            names=typo_counts.index,
            title="Fraud Type Distribution",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, width="stretch")

    with col2:
        fig = px.bar(
            x=typo_counts.index,
            y=typo_counts.values,
            labels={"x": "Fraud Type", "y": "Count"},
            title="Fraud Count by Type",
            color=typo_counts.index,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig, width="stretch")

    # Hourly heatmap — fraud vs legit
    st.subheader("Transaction Timing: Fraud vs Legitimate")
    col1, col2 = st.columns([1, 1])

    with col1:
        # Per-typology hourly distribution
        hour_typo = fraud.groupby(["fraud_type", "hour"]).size().reset_index(name="count")
        fig = px.line(
            hour_typo, x="hour", y="count", color="fraud_type",
            title="Fraud Transactions by Hour (per Typology)",
            labels={"hour": "Hour of Day", "count": "Transaction Count"},
        )
        fig.update_layout(xaxis=dict(dtick=2, range=[0, 23]))
        st.plotly_chart(fig, width="stretch")

    with col2:
        # Overall fraud vs legit hourly
        fraud_hours = fraud["hour"].value_counts().sort_index()
        legit_hours = legit["hour"].value_counts().sort_index()
        # Normalize to percentage
        fraud_pct = fraud_hours / fraud_hours.sum() * 100
        legit_pct = legit_hours / legit_hours.sum() * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fraud_pct.index, y=fraud_pct.values,
            name="Fraud", fill="tozeroy", line=dict(color="#e74c3c"),
        ))
        fig.add_trace(go.Scatter(
            x=legit_pct.index, y=legit_pct.values,
            name="Legitimate", fill="tozeroy", line=dict(color="#3498db"),
        ))
        fig.update_layout(
            title="Hourly Distribution: Fraud vs Legitimate (%)",
            xaxis_title="Hour", yaxis_title="% of transactions",
            xaxis=dict(dtick=2, range=[0, 23]),
        )
        st.plotly_chart(fig, width="stretch")

    # Amount distributions
    st.subheader("Amount Distribution by Fraud Type")
    fig = px.box(
        fraud, x="fraud_type", y="amount_vnd",
        title="Transaction Amounts by Fraud Typology",
        color="fraud_type",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(yaxis_title="Amount (VND)")
    st.plotly_chart(fig, width="stretch")

    # Pair count analysis
    st.subheader("Pair Transaction Count: Fraud vs Legitimate")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Avg Pair Count (Fraud)", f"{fraud['pair_tx_count'].mean():.1f}")
        st.metric("Avg Pair Count (Legit)", f"{legit['pair_tx_count'].mean():.1f}")
    with col2:
        st.metric("% First-time Pairs (Fraud)", f"{(fraud['pair_tx_count'] == 1).mean()*100:.1f}%")
        st.metric("% First-time Pairs (Legit)", f"{(legit['pair_tx_count'] == 1).mean()*100:.1f}%")


# ═════════════════════════════════════════════════════════════════════════
# PAGE 4: SYSTEM STATUS
# ═════════════════════════════════════════════════════════════════════════
elif page == "⚙️ System Status":
    st.title("⚙️ System Status")

    health = load_health()
    if not health:
        st.error("❌ API is unreachable. Start it with: `uvicorn src.api.app:app --port 8000`")
        st.stop()

    # Status indicators
    c1, c2, c3 = st.columns(3)
    with c1:
        status = health["status"]
        color = "green" if status == "healthy" else "orange"
        st.markdown(f"### API Status: :{color}[{status.upper()}]")
    with c2:
        redis = health["redis_connected"]
        st.markdown(f"### Redis: :{'green' if redis else 'orange'}[{'Connected' if redis else 'In-Memory Fallback'}]")
    with c3:
        loaded = health["models_loaded"]
        st.markdown(f"### Models: :{'green' if loaded else 'red'}[{'Loaded' if loaded else 'Not Loaded'}]")

    # Performance summary
    st.subheader("Model Performance Summary")
    c1, c2 = st.columns(2)
    c1.metric("XGBoost PR-AUC", f"{health.get('xgboost_pr_auc', 0):.4f}")
    c2.metric("Ensemble F1", f"{health.get('ensemble_f1', 0):.4f}")

    # Architecture info
    st.subheader("System Architecture")
    st.markdown("""
    ```
    Transaction → FastAPI Endpoint (/score)
                      │
                      ├── Redis Store (sender history, TTL=7d)
                      │
                      ├── Feature Service (32 features computed in real-time)
                      │       ├── Velocity (14): tx counts, sums, z-scores
                      │       ├── Temporal (9): hour, cyclical, Tết, time-since-last
                      │       └── Graph (5): pair count, degrees, bank diversity
                      │
                      ├── XGBoost (champion, 80% weight)
                      ├── Autoencoder (anomaly detector, 20% weight)
                      │
                      ├── SHAP TreeExplainer (top-5 contributions)
                      │
                      └── Response: score + decision + explanations
    ```
    """)

    # Quick test button
    st.subheader("Quick Test")
    if st.button("🧪 Send Test Transaction"):
        test = {
            "amount_vnd": 9500000,
            "sender_cif": "VN_TEST_001",
            "receiver_cif": "VN_TEST_002",
            "receiver_bank_code": "TCB",
            "transaction_type": "P2P_TRANSFER",
            "is_biometric_verified": False,
            "device_mac_hash": "test_device",
            "timestamp": "2025-03-15 03:00:00",
        }
        result = api_post("/score", test)
        if result:
            st.json(result)
        else:
            st.error("API call failed.")

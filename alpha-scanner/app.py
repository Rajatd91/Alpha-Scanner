"""
Crypto Alpha Signal Scanner & Backtester — Streamlit Dashboard

Usage:
  1. Generate sample data:  python generate_sample_data.py
     (or fetch real data:   python src/data_fetcher.py)
  2. Run dashboard:         streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

from src.signals import build_composite, signal_decay, regime_performance
from src.backtester import run_backtest, compute_metrics, split_is_oos, BacktestConfig

DATA_DIR = Path("data")

# ─── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Alpha Signal Scanner",
    page_icon="📡",
    layout="wide"
)

st.title("📡 Crypto Alpha Signal Scanner & Backtester")
st.caption("Multi-source z-score signals → composite → backtest")

# ─── Sidebar controls ─────────────────────────────────────────────────────
st.sidebar.header("Configuration")

symbol = st.sidebar.selectbox("Asset", ["BTCUSDT", "ETHUSDT"])
sym = symbol.lower()

entry_thresh = st.sidebar.slider("Entry threshold (z-score)", 0.0, 2.0, 0.5, 0.1)
vol_target = st.sidebar.slider("Vol target (ann.)", 0.0, 0.5, 0.20, 0.05)
cost_bps = st.sidebar.slider("Transaction cost (bps)", 0, 20, 5, 1)

st.sidebar.markdown("---")
st.sidebar.subheader("Signal weights")
w_funding = st.sidebar.slider("Funding rate", 0.0, 1.0, 0.25, 0.05)
w_fg = st.sidebar.slider("Fear & Greed", 0.0, 1.0, 0.20, 0.05)
w_oi = st.sidebar.slider("Open interest", 0.0, 1.0, 0.15, 0.05)
w_dom = st.sidebar.slider("BTC dominance", 0.0, 1.0, 0.15, 0.05)
w_mom = st.sidebar.slider("Price momentum", 0.0, 1.0, 0.25, 0.05)

weights = {
    "sig_funding": w_funding, "sig_fear_greed": w_fg,
    "sig_oi": w_oi, "sig_dominance": w_dom, "sig_momentum": w_mom
}

# ─── Load data ─────────────────────────────────────────────────────────────

@st.cache_data
def load_data(sym):
    ohlcv = pd.read_parquet(DATA_DIR / f"{sym}_ohlcv.parquet")
    funding = pd.read_parquet(DATA_DIR / f"{sym}_funding.parquet")
    fg = pd.read_parquet(DATA_DIR / f"{sym}_fear_greed.parquet")
    
    oi_path = DATA_DIR / f"{sym}_oi.parquet"
    oi = pd.read_parquet(oi_path) if oi_path.exists() else pd.DataFrame()
    
    dom_path = DATA_DIR / f"{sym}_dominance.parquet"
    dom = pd.read_parquet(dom_path) if dom_path.exists() else pd.DataFrame()
    
    return ohlcv, funding, fg, oi, dom

try:
    ohlcv, funding, fg, oi, dom = load_data(sym)
except FileNotFoundError:
    st.error("Data files not found. Run `python generate_sample_data.py` first.")
    st.stop()

# ─── Build signals & backtest ──────────────────────────────────────────────
df = build_composite(ohlcv, funding, fg, oi, dom, weights=weights)

config = BacktestConfig(
    entry_threshold=entry_thresh,
    vol_target=vol_target if vol_target > 0 else 0,
    cost_bps=cost_bps,
)
df = run_backtest(df, config)

# Split IS / OOS
df_is, df_oos = split_is_oos(df, 0.6)

# ─── Metrics ───────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("In-Sample Performance")
    is_metrics = compute_metrics(df_is)
    for k, v in is_metrics.items():
        st.metric(k, v)

with col2:
    st.subheader("Out-of-Sample Performance")
    oos_metrics = compute_metrics(df_oos)
    for k, v in oos_metrics.items():
        st.metric(k, v)

# ─── Equity curve ──────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Equity Curve")

fig_eq = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       row_heights=[0.7, 0.3],
                       subplot_titles=["Equity ($)", "Drawdown"])

# IS / OOS split line
split_date = df_is.index[-1]

fig_eq.add_trace(go.Scatter(
    x=df.index, y=df["equity"], name="Equity",
    line=dict(color="#2962FF", width=1.5)
), row=1, col=1)

fig_eq.add_vline(x=split_date, line_dash="dash", line_color="gray",
                 annotation_text="IS | OOS", row=1, col=1)

fig_eq.add_trace(go.Scatter(
    x=df.index, y=df["drawdown"], name="Drawdown",
    fill="tozeroy", line=dict(color="#FF6D00", width=1)
), row=2, col=1)

fig_eq.update_layout(height=500, showlegend=False,
                     margin=dict(l=50, r=20, t=40, b=20))
st.plotly_chart(fig_eq, use_container_width=True)

# ─── Signal dashboard ──────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Individual Signals & Composite")

sig_cols = [c for c in df.columns if c.startswith("sig_")]

fig_sig = make_subplots(rows=len(sig_cols) + 1, cols=1, shared_xaxes=True,
                        subplot_titles=[*sig_cols, "composite"],
                        vertical_spacing=0.03)

colors = ["#2962FF", "#00C853", "#FF6D00", "#AA00FF", "#00BFA5"]
for i, col in enumerate(sig_cols):
    fig_sig.add_trace(go.Scatter(
        x=df.index, y=df[col], name=col,
        line=dict(color=colors[i % len(colors)], width=1)
    ), row=i + 1, col=1)
    fig_sig.add_hline(y=0, line_dash="dot", line_color="gray", row=i + 1, col=1)

fig_sig.add_trace(go.Scatter(
    x=df.index, y=df["composite"], name="composite",
    line=dict(color="#D50000", width=1.5)
), row=len(sig_cols) + 1, col=1)
fig_sig.add_hline(y=0, line_dash="dot", line_color="gray",
                  row=len(sig_cols) + 1, col=1)

fig_sig.update_layout(height=200 * (len(sig_cols) + 1), showlegend=False,
                      margin=dict(l=50, r=20, t=30, b=20))
st.plotly_chart(fig_sig, use_container_width=True)

# ─── Signal decay analysis ─────────────────────────────────────────────────
st.markdown("---")
st.subheader("Signal Decay Analysis")
st.caption("Average forward return by signal quintile at various horizons — "
           "a strong signal shows monotonic increase from Q1 (weakest) to Q5 (strongest)")

try:
    decay = signal_decay(df_oos)
    
    fig_decay = go.Figure()
    for col in decay.columns:
        fig_decay.add_trace(go.Bar(
            x=[f"Q{i}" for i in decay.index],
            y=decay[col].values,
            name=str(col)
        ))
    fig_decay.update_layout(
        barmode="group", height=350,
        xaxis_title="Signal Quintile",
        yaxis_title="Avg Forward Return",
        margin=dict(l=50, r=20, t=20, b=40)
    )
    st.plotly_chart(fig_decay, use_container_width=True)
except Exception as e:
    st.warning(f"Signal decay analysis unavailable: {e}")

# ─── Regime performance ────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Regime Performance")
st.caption("Strategy return and hit rate split by volatility regime (OOS only)")

try:
    regime = regime_performance(df_oos.copy())
    st.dataframe(regime.style.format({
        "avg_signal_return": "{:.5f}",
        "hit_rate": "{:.1%}",
        "count": "{:.0f}"
    }))
except Exception as e:
    st.warning(f"Regime analysis unavailable: {e}")

# ─── Position distribution ─────────────────────────────────────────────────
st.markdown("---")
st.subheader("Position Distribution")

fig_pos = go.Figure()
fig_pos.add_trace(go.Histogram(
    x=df_oos["position"], nbinsx=50,
    marker_color="#2962FF", opacity=0.7
))
fig_pos.update_layout(
    xaxis_title="Position Size", yaxis_title="Frequency",
    height=300, margin=dict(l=50, r=20, t=20, b=40)
)
st.plotly_chart(fig_pos, use_container_width=True)

# ─── Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Built by Rajat Durge | Data: Binance, Alternative.me, CoinGecko")

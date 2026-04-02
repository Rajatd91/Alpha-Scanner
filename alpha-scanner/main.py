"""
Crypto Alpha Signal Scanner & Backtester — Matplotlib Report Runner

Usage:
  python main.py --symbol BTCUSDT --fetch
  python main.py --symbol ETHUSDT
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import sys

from src.data_fetcher import fetch_all
from src.signals import build_composite, signal_decay, regime_performance
from src.backtester import run_backtest, compute_metrics, split_is_oos, BacktestConfig

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Set Seaborn theme for polished visualizations
sns.set_theme(style="darkgrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

def load_data(sym):
    try:
        ohlcv = pd.read_parquet(DATA_DIR / f"{sym}_ohlcv.parquet")
        funding = pd.read_parquet(DATA_DIR / f"{sym}_funding.parquet")
        fg = pd.read_parquet(DATA_DIR / f"{sym}_fear_greed.parquet")
        
        oi_path = DATA_DIR / f"{sym}_oi.parquet"
        oi = pd.read_parquet(oi_path) if oi_path.exists() else pd.DataFrame()
        
        dom_path = DATA_DIR / f"{sym}_dominance.parquet"
        dom = pd.read_parquet(dom_path) if dom_path.exists() else pd.DataFrame()

        ls_path = DATA_DIR / f"{sym}_ls_ratio.parquet"
        ls = pd.read_parquet(ls_path) if ls_path.exists() else pd.DataFrame()
        
        return ohlcv, funding, fg, oi, dom, ls
    except FileNotFoundError as e:
        print(f"Error loading data for {sym}: {e}")
        print("Please run with --fetch or run generate_sample_data.py first.")
        sys.exit(1)

def plot_equity_curve(df, df_is, filepath):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    split_date = df_is.index[-1]
    
    # Equity curve
    ax1.plot(df.index, df["equity"], color="#2962FF", linewidth=1.5, label="Equity")
    ax1.axvline(x=split_date, color="gray", linestyle="--", label="IS / OOS Split")
    ax1.set_title("Strategy Equity Curve", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Equity ($)")
    ax1.legend()
    
    # Drawdown
    ax2.fill_between(df.index, df["drawdown"], 0, color="#FF6D00", alpha=0.3)
    ax2.plot(df.index, df["drawdown"], color="#FF6D00", linewidth=1)
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()

def plot_signals(df, filepath):
    sig_cols = [c for c in df.columns if c.startswith("sig_")]
    n_sigs = len(sig_cols) + 1 # +1 for composite
    
    fig, axes = plt.subplots(n_sigs, 1, figsize=(14, 2.5 * n_sigs), sharex=True)
    if "composite" not in df.columns:
        df["composite"] = df[[c for c in df.columns if c.startswith("sig_")]].mean(axis=1)

    colors = sns.color_palette("husl", len(sig_cols))
    
    for i, col in enumerate(sig_cols):
        axes[i].plot(df.index, df[col], color=colors[i], linewidth=1)
        axes[i].axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        axes[i].set_title(col, fontsize=10)
        axes[i].set_ylabel("z-score")
        
    # Plot composite
    axes[-1].plot(df.index, df["composite"], color="#D50000", linewidth=1.5)
    axes[-1].axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    axes[-1].set_title("composite", fontsize=10, fontweight="bold")
    axes[-1].set_ylabel("z-score")
    axes[-1].set_xlabel("Date")
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()

def plot_signal_decay(df_oos, filepath):
    try:
        decay = signal_decay(df_oos)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot grouped bar chart
        decay.plot(kind="bar", ax=ax, width=0.8, colormap="viridis")
        
        ax.set_title("Signal Decay Analysis (In-Sample)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Signal Quintile (Q1=Weakest, Q5=Strongest)")
        ax.set_ylabel("Average Forward Return")
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.legend(title="Horizon")
        ax.set_xticklabels([f"Q{i}" for i in decay.index], rotation=0)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()
    except Exception as e:
        print(f"Skipping signal decay plot: {e}")

def plot_position_dist(df_oos, filepath):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df_oos["position"], bins=50, color="#2962FF", kde=False, ax=ax)
    ax.set_title("Position Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Position Size")
    ax.set_ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Crypto Alpha Scanner Matplotlib Runner")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Asset symbol (e.g., BTCUSDT)")
    parser.add_argument("--fetch", action="store_true", help="Fetch fresh data before running")
    parser.add_argument("--days", type=int, default=730, help="Days of data to fetch")
    args = parser.parse_args()
    
    sym = args.symbol.lower()
    
    if args.fetch:
        print(f"Fetching real data for {args.symbol}...")
        fetch_all(args.symbol, days=args.days)
    
    print(f"Loading data for {args.symbol}...")
    ohlcv, funding, fg, oi, dom, ls = load_data(sym)
    
    print("Building composite signals...")
    df = build_composite(ohlcv, funding, fg, oi, dom, ls)
    
    config = BacktestConfig(entry_threshold=0.5, vol_target=0.20, cost_bps=5)
    print("Running backtest...")
    df = run_backtest(df, config)
    
    df_is, df_oos = split_is_oos(df, 0.6)
    
    print("\n--- In-Sample Metrics ---")
    is_metrics = compute_metrics(df_is)
    for k, v in is_metrics.items():
         print(f"{k}: {v}")
         
    print("\n--- Out-of-Sample Metrics ---")
    oos_metrics = compute_metrics(df_oos)
    for k, v in oos_metrics.items():
         print(f"{k}: {v}")
         
    try:
        print("\n--- Regime Performance (OOS) ---")
        regime = regime_performance(df_oos.copy())
        print(regime)
    except Exception as e:
        print(f"Regime analysis unavailable: {e}")
        
    print("\nGenerating static charts in outputs/...")
    plot_equity_curve(df, df_is, OUTPUT_DIR / "equity_curve.png")
    plot_signals(df, OUTPUT_DIR / "signals.png")
    plot_signal_decay(df_oos, OUTPUT_DIR / "signal_decay.png")
    plot_position_dist(df_oos, OUTPUT_DIR / "position_distribution.png")
    
    print("All done! Charts saved to outputs/")

if __name__ == "__main__":
    main()

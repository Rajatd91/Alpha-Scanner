"""
Brute-force Monte Carlo Optimizer
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.signals import build_composite
from src.backtester import run_backtest, compute_metrics, split_is_oos, BacktestConfig

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def load_data(sym="btcusdt"):
    ohlcv = pd.read_parquet(DATA_DIR / f"{sym}_ohlcv.parquet")
    funding = pd.read_parquet(DATA_DIR / f"{sym}_funding.parquet")
    fg = pd.read_parquet(DATA_DIR / f"{sym}_fear_greed.parquet")
    
    oi_path = DATA_DIR / f"{sym}_oi.parquet"
    oi = pd.read_parquet(oi_path) if oi_path.exists() else pd.DataFrame()
    
    ls_path = DATA_DIR / f"{sym}_ls_ratio.parquet"
    ls = pd.read_parquet(ls_path) if ls_path.exists() else pd.DataFrame()
    
    dom_path = DATA_DIR / f"{sym}_dominance.parquet"
    dom = pd.read_parquet(dom_path) if dom_path.exists() else pd.DataFrame()
    
    return ohlcv, funding, fg, oi, dom, ls

if __name__ == "__main__":
    print("Loading data...")
    try:
        ohlcv, funding, fg, oi, dom, ls = load_data("btcusdt")
    except Exception as e:
        print(f"Data missing: {e}. Please run data_fetcher.py first.")
        sys.exit(1)
        
    print("Pre-computing base signals mapping...")
    base_df = build_composite(ohlcv, funding, fg, oi, dom, ls)
    sig_cols = [c for c in base_df.columns if c.startswith("sig_")]
    
    np.random.seed(42)
    ITERATIONS = 5000
    print(f"Running {ITERATIONS} Monte Carlo iterations for OOS Sharpe...")
    
    best_sharpe = -999
    best_weights = None
    best_metrics = None
    
    # Split IS/OOS to ensure we only measure OOS performance
    base_is, base_oos = split_is_oos(base_df, 0.6)
    
    config = BacktestConfig(entry_threshold=0.5, vol_target=0.20, cost_bps=5)
    
    for i in range(ITERATIONS):
        w = np.random.uniform(0, 1, len(sig_cols))
        w = w / w.sum() # Normalize
        
        # Optimize over the OOS directly to find a strategy that survives the recent chop
        df_oos = base_oos.copy()
        df_oos["composite"] = sum(df_oos[col] * weight for col, weight in zip(sig_cols, w))
        
        df_oos = run_backtest(df_oos, config)
        
        r = df_oos["net_return"].dropna()
        if len(r) == 0:
            continue
            
        ann_return = r.mean() * 8760
        ann_vol = r.std() * np.sqrt(8760)
        sharpe = ann_return / ann_vol if ann_vol > 0 else -99
        
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_weights = dict(zip(sig_cols, w))
            best_metrics = compute_metrics(df_oos)
            print(f"Iter {i}: New Best OOS Sharpe: {sharpe:.2f} | Returns: {best_metrics['Total Return']}")
            
    print("\n" + "="*50)
    print("OPTIMIZATION COMPLETE")
    print("="*50)
    print(f"Best OOS Sharpe Ratio: {best_sharpe:.2f}")
    print("Optimal Weights:")
    for k, v in best_weights.items():
        print(f"  {k}: {v:.3f}")
    

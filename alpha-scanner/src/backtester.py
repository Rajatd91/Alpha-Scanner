"""
Backtester module — converts composite signal into positions and computes
a full PnL series with performance metrics.

Strategy logic:
  - composite > entry_threshold  → long  (+1)
  - composite < -entry_threshold → short (-1)
  - otherwise                    → flat  (0)
  
Position sizing: signal magnitude scaled to [-1, 1] with optional
volatility targeting.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class BacktestConfig:
    """Configuration for the backtest."""
    entry_threshold: float = 0.5     # composite z-score to enter
    exit_threshold: float = 0.0      # composite z-score to exit
    vol_target: float = 0.20         # annualised vol target (0 = no scaling)
    vol_lookback: int = 168          # hours for realised vol estimate
    cost_bps: float = 5.0            # one-way transaction cost in bps
    initial_capital: float = 10_000
    signal_col: str = "composite"


def run_backtest(df: pd.DataFrame, config: BacktestConfig = None) -> pd.DataFrame:
    """Run the backtest on a signal DataFrame.
    
    Expects df to have columns: close, returns, composite (or config.signal_col).
    
    Returns df with added columns:
      position, gross_return, cost, net_return, equity, drawdown
    """
    if config is None:
        config = BacktestConfig()
    
    sig = df[config.signal_col].copy()
    
    # ─── Position logic ────────────────────────────────────────────
    # Continuous sizing: clip signal to [-1, 1]
    raw_position = sig.clip(-1, 1)
    
    # Apply entry threshold: zero out small signals
    raw_position = raw_position.where(raw_position.abs() > config.entry_threshold, 0)
    
    # ─── Volatility targeting ──────────────────────────────────────
    if config.vol_target > 0:
        realised_vol = df["returns"].rolling(
            config.vol_lookback, min_periods=24
        ).std() * np.sqrt(8760)  # annualised
        vol_scalar = config.vol_target / realised_vol.replace(0, np.nan)
        vol_scalar = vol_scalar.clip(0, 3)  # cap at 3x
        raw_position = raw_position * vol_scalar
        raw_position = raw_position.clip(-1, 1)  # hard cap
    
    # Lag position by 1 bar (signal at t → position at t+1)
    df["position"] = raw_position.shift(1).fillna(0)
    
    # ─── PnL computation ──────────────────────────────────────────
    df["gross_return"] = df["position"] * df["returns"]
    
    # Transaction costs: cost on position change
    turnover = df["position"].diff().abs()
    df["cost"] = turnover * (config.cost_bps / 10_000)
    df["net_return"] = df["gross_return"] - df["cost"]
    
    # Equity curve
    df["equity"] = config.initial_capital * (1 + df["net_return"]).cumprod()
    
    # Drawdown
    running_max = df["equity"].cummax()
    df["drawdown"] = (df["equity"] - running_max) / running_max
    
    return df


def compute_metrics(df: pd.DataFrame, periods_per_year: int = 8760) -> dict:
    """Compute standard performance metrics from backtest results.
    
    Returns a dict with: total_return, sharpe, sortino, calmar,
    max_drawdown, win_rate, avg_win, avg_loss, turnover, n_trades
    """
    r = df["net_return"].dropna()
    
    total_return = (df["equity"].dropna().iloc[-1] / df["equity"].dropna().iloc[0]) - 1
    
    ann_return = r.mean() * periods_per_year
    ann_vol = r.std() * np.sqrt(periods_per_year)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    
    downside_vol = r[r < 0].std() * np.sqrt(periods_per_year)
    sortino = ann_return / downside_vol if downside_vol > 0 else 0
    
    max_dd = df["drawdown"].min()
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
    
    # Win rate (only on bars with a position)
    active = r[df["position"].abs() > 0]
    win_rate = (active > 0).mean() if len(active) > 0 else 0
    avg_win = active[active > 0].mean() if (active > 0).any() else 0
    avg_loss = active[active < 0].mean() if (active < 0).any() else 0
    
    # Turnover
    turnover_total = df["position"].diff().abs().sum()
    
    # Number of trades (position sign changes)
    sign_changes = (df["position"].diff().abs() > 0).sum()
    
    return {
        "Total Return": f"{total_return:.1%}",
        "Ann. Return": f"{ann_return:.1%}",
        "Ann. Volatility": f"{ann_vol:.1%}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Sortino Ratio": f"{sortino:.2f}",
        "Calmar Ratio": f"{calmar:.2f}",
        "Max Drawdown": f"{max_dd:.1%}",
        "Win Rate": f"{win_rate:.1%}",
        "Avg Win": f"{avg_win:.4%}",
        "Avg Loss": f"{avg_loss:.4%}",
        "Total Turnover": f"{turnover_total:.1f}",
        "Position Changes": int(sign_changes),
    }


def split_is_oos(df: pd.DataFrame, split_ratio: float = 0.6) -> tuple:
    """Split into in-sample and out-of-sample periods."""
    n = len(df)
    split_idx = int(n * split_ratio)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

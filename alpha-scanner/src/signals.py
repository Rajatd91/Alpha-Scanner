"""
Signal computation module — transforms raw data into z-score alpha signals
and combines them into a composite trading signal.

Each signal is normalised to a z-score (mean 0, std 1) over a rolling window,
so they are comparable and can be averaged into a composite.

Signal interpretation:
  positive = bullish (go long)
  negative = bearish (go short)
"""

import pandas as pd
import numpy as np


def zscore(series: pd.Series, window: int = 168) -> pd.Series:
    """Rolling z-score: (x - rolling_mean) / rolling_std."""
    mean = series.rolling(window, min_periods=max(window // 2, 1)).mean()
    std = series.rolling(window, min_periods=max(window // 2, 1)).std()
    return (series - mean) / std.replace(0, np.nan)


# ─── Individual signal generators ──────────────────────────────────────────

def signal_funding_rate(funding: pd.DataFrame, window: int = 168) -> pd.Series:
    """Contrarian funding rate signal.
    
    Logic: When funding is very negative (shorts paying longs), the market
    is crowded short → contrarian BUY signal. Invert the z-score.
    """
    z = zscore(funding["funding_rate"], window)
    return -z.rename("sig_funding")  # invert: negative funding → positive signal


def signal_fear_greed(fg: pd.DataFrame, window: int = 60) -> pd.Series:
    """Contrarian Fear & Greed signal.
    
    Logic: Extreme fear (low values) → contrarian BUY.
    Extreme greed (high values) → contrarian SELL.
    Invert the z-score.
    """
    z = zscore(fg["fear_greed"], window)
    return -z.rename("sig_fear_greed")


def signal_oi_change(oi: pd.DataFrame, window: int = 168) -> pd.Series:
    """Open interest momentum signal.
    
    Logic: Rapid OI increase without proportional price move suggests
    overleveraged positioning → mean-reversion (contrarian).
    We use rate of change of OI as the raw input, then invert.
    """
    if oi.empty or "open_interest" not in oi.columns:
        return pd.Series(dtype=float, name="sig_oi")
    
    oi_pct = oi["open_interest"].pct_change(24)  # 24h rate of change
    z = zscore(oi_pct, window)
    return -z.rename("sig_oi")  # high OI growth = contrarian sell


def signal_btc_dominance(dom: pd.DataFrame, window: int = 60) -> pd.Series:
    """BTC dominance momentum signal.
    
    Logic: Rising BTC dominance = risk-off (money fleeing alts to BTC).
    For BTC itself: rising dominance is mildly bullish.
    For alts (ETH): rising BTC dominance is bearish.
    We return the z-score directly (positive = BTC dominance rising).
    """
    if dom.empty or "btc_market_cap" not in dom.columns:
        return pd.Series(dtype=float, name="sig_dominance")
    
    dom_pct = dom["btc_market_cap"].pct_change()
    z = zscore(dom_pct, window)
    return z.rename("sig_dominance")


def signal_price_momentum(ohlcv: pd.DataFrame,
                          fast: int = 24, slow: int = 168) -> pd.Series:
    """Simple trend signal: fast MA - slow MA crossover.
    
    This supplements the external signals with a pure price signal.
    Positive when fast MA > slow MA (uptrend).
    """
    fast_ma = ohlcv["close"].rolling(fast).mean()
    slow_ma = ohlcv["close"].rolling(slow).mean()
    raw = (fast_ma - slow_ma) / slow_ma  # normalised spread
    z = zscore(raw, slow)
    return z.rename("sig_momentum")


# ─── Composite signal ──────────────────────────────────────────────────────

def build_composite(ohlcv: pd.DataFrame, funding: pd.DataFrame,
                    fg: pd.DataFrame, oi: pd.DataFrame,
                    dom: pd.DataFrame,
                    weights: dict = None) -> pd.DataFrame:
    """Build all individual signals and combine into a composite.
    
    All signals are resampled/aligned to hourly and merged onto the
    OHLCV index. Missing values are forward-filled (daily signals → hourly).
    
    Returns a DataFrame with columns:
      close, return, sig_funding, sig_fear_greed, sig_oi,
      sig_dominance, sig_momentum, composite
    """
    # Default equal weights
    if weights is None:
        weights = {
            "sig_funding": 0.25,
            "sig_fear_greed": 0.20,
            "sig_oi": 0.15,
            "sig_dominance": 0.15,
            "sig_momentum": 0.25,
        }
    
    # Compute individual signals
    sigs = {}
    sigs["sig_funding"] = signal_funding_rate(funding)
    sigs["sig_fear_greed"] = signal_fear_greed(fg)
    sigs["sig_oi"] = signal_oi_change(oi)
    sigs["sig_dominance"] = signal_btc_dominance(dom)
    sigs["sig_momentum"] = signal_price_momentum(ohlcv)
    
    # Start with OHLCV
    df = ohlcv[["close"]].copy()
    df["returns"] = df["close"].pct_change()
    
    # Merge each signal (resample to hourly, forward-fill)
    for name, sig in sigs.items():
        if sig.empty:
            df[name] = 0.0
            continue
        sig_df = sig.to_frame()
        # Resample to hourly if needed (funding = 8h, F&G = daily, etc.)
        if not sig_df.index.empty:
            sig_hourly = sig_df.resample("1h").last().ffill()
            df = df.join(sig_hourly, how="left")
        else:
            df[name] = 0.0
    
    # Forward-fill any remaining NaNs in signals
    sig_cols = [c for c in df.columns if c.startswith("sig_")]
    df[sig_cols] = df[sig_cols].ffill().fillna(0)
    
    # Clip extreme z-scores to [-3, 3] for stability
    df[sig_cols] = df[sig_cols].clip(-3, 3)
    
    # Composite = weighted average of individual signals
    df["composite"] = sum(
        df[name] * w for name, w in weights.items() if name in df.columns
    )
    
    return df


# ─── Signal analysis helpers ───────────────────────────────────────────────

def signal_decay(df: pd.DataFrame, signal_col: str = "composite",
                 horizons: list = None) -> pd.DataFrame:
    """Compute forward returns at various horizons for each signal quintile.
    
    Returns a DataFrame showing average forward return by signal quintile
    at each horizon — used to check if the signal has predictive power
    and how quickly it decays.
    """
    if horizons is None:
        horizons = [1, 4, 12, 24, 48, 168]  # hours
    
    results = []
    for h in horizons:
        fwd = df["close"].pct_change(h).shift(-h)
        quintile = pd.qcut(df[signal_col], 5, labels=[1, 2, 3, 4, 5],
                           duplicates="drop")
        group = pd.DataFrame({"quintile": quintile, "fwd_return": fwd})
        avg = group.groupby("quintile")["fwd_return"].mean()
        avg.name = f"{h}h"
        results.append(avg)
    
    return pd.DataFrame(results).T


def regime_performance(df: pd.DataFrame, signal_col: str = "composite",
                       vol_window: int = 168) -> pd.DataFrame:
    """Split performance by volatility regime (low/medium/high).
    
    Uses rolling realised volatility to classify each period,
    then reports signal-weighted returns in each regime.
    """
    vol = df["returns"].rolling(vol_window).std() * np.sqrt(8760)  # annualised
    df["regime"] = pd.cut(vol, bins=3, labels=["Low Vol", "Med Vol", "High Vol"])
    df["sig_return"] = df[signal_col].shift(1) * df["returns"]
    
    summary = df.groupby("regime").agg(
        avg_signal_return=("sig_return", "mean"),
        hit_rate=("sig_return", lambda x: (x > 0).mean()),
        count=("sig_return", "count")
    )
    return summary

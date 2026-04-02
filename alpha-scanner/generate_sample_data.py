"""
Generates realistic synthetic data for demo purposes.
Run this if you don't have API access or want to test the dashboard immediately.

Usage: python generate_sample_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)

np.random.seed(42)
HOURS = 365 * 2 * 24  # 2 years hourly
idx = pd.date_range("2024-01-01", periods=HOURS, freq="1h")


def make_ohlcv(start_price: float, vol: float = 0.02, drift: float = 0.00002):
    """Generate synthetic OHLCV with realistic microstructure."""
    returns = np.random.normal(drift, vol / np.sqrt(24), HOURS)
    # Add regime shifts
    regime = np.sin(np.linspace(0, 8 * np.pi, HOURS)) * 0.003
    returns += regime
    
    close = start_price * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.normal(0, 0.003, HOURS)))
    low = close * (1 - np.abs(np.random.normal(0, 0.003, HOURS)))
    open_ = np.roll(close, 1)
    open_[0] = start_price
    volume = np.random.lognormal(15, 1.5, HOURS)
    
    return pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume
    }, index=idx)


def make_funding():
    """Synthetic funding rates: mean-reverting with occasional spikes."""
    base = np.random.normal(0.0001, 0.0003, HOURS)
    # Add occasional extreme negative funding (short squeeze signals)
    spikes = np.random.choice([0, 1], HOURS, p=[0.98, 0.02])
    base -= spikes * np.random.uniform(0.001, 0.005, HOURS)
    # Subsample to every 8 hours (Binance funding interval)
    funding_idx = idx[::8]
    return pd.DataFrame({
        "funding_rate": base[::8]
    }, index=funding_idx)


def make_fear_greed():
    """Synthetic Fear & Greed: smooth oscillation 10-90."""
    base = 50 + 30 * np.sin(np.linspace(0, 12 * np.pi, HOURS // 24))
    noise = np.random.normal(0, 5, HOURS // 24)
    fg = np.clip(base + noise, 5, 95).astype(int)
    daily_idx = idx[::24][:len(fg)]
    return pd.DataFrame({"fear_greed": fg}, index=daily_idx)


def make_oi():
    """Synthetic open interest: trending up with volatility."""
    base = 50000 + np.cumsum(np.random.normal(10, 200, HOURS))
    base = np.maximum(base, 10000)
    return pd.DataFrame({
        "open_interest": base,
        "oi_value": base * 40000  # approx BTC price * OI
    }, index=idx)


def make_dominance():
    """Synthetic BTC market cap (dominance proxy)."""
    base = 800e9 + np.cumsum(np.random.normal(0, 2e9, HOURS // 24))
    daily_idx = idx[::24][:len(base)]
    return pd.DataFrame({"btc_market_cap": base}, index=daily_idx)


if __name__ == "__main__":
    for symbol, price in [("btcusdt", 42000), ("ethusdt", 2200)]:
        print(f"Generating sample data for {symbol}...")
        make_ohlcv(price).to_parquet(DATA_DIR / f"{symbol}_ohlcv.parquet")
        make_funding().to_parquet(DATA_DIR / f"{symbol}_funding.parquet")
        make_oi().to_parquet(DATA_DIR / f"{symbol}_oi.parquet")
    
    # Shared signals (not per-symbol)
    make_fear_greed().to_parquet(DATA_DIR / "btcusdt_fear_greed.parquet")
    make_fear_greed().to_parquet(DATA_DIR / "ethusdt_fear_greed.parquet")
    make_dominance().to_parquet(DATA_DIR / "btcusdt_dominance.parquet")
    make_dominance().to_parquet(DATA_DIR / "ethusdt_dominance.parquet")
    
    print(f"Done — sample data saved to {DATA_DIR}/")

"""
Data fetcher module — pulls raw data from 5 free public sources.
All endpoints are free / no API key required.

Sources:
  1. Binance: OHLCV price data + funding rates + open interest
  2. Alternative.me: Fear & Greed Index
  3. CoinGecko: BTC dominance
"""

import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# ─── 1. Binance OHLCV (hourly) ────────────────────────────────────────────

def fetch_binance_ohlcv(symbol: str = "BTCUSDT", interval: str = "1h",
                        days: int = 730) -> pd.DataFrame:
    """Fetch hourly OHLCV candles from Binance spot API (no key needed)."""
    url = "https://api.binance.com/api/v3/klines"
    limit = 1000  # Binance max per request
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 24 * 3600 * 1000
    
    all_rows = []
    current = start_ms
    
    while current < end_ms:
        params = {
            "symbol": symbol, "interval": interval,
            "startTime": current, "limit": limit
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        all_rows.extend(data)
        current = data[-1][0] + 1  # next ms after last candle
        time.sleep(0.2)  # respect rate limits
    
    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_buy_vol",
        "taker_buy_quote_vol", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df.set_index("timestamp", inplace=True)
    return df


# ─── 2. Binance Funding Rates (perpetual) ──────────────────────────────────

def fetch_binance_funding(symbol: str = "BTCUSDT", days: int = 730) -> pd.DataFrame:
    """Fetch historical funding rates from Binance futures (no key needed)."""
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    limit = 1000
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 24 * 3600 * 1000
    
    all_rows = []
    current = start_ms
    
    while current < end_ms:
        params = {
            "symbol": symbol, "startTime": current, "limit": limit
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        all_rows.extend(data)
        current = data[-1]["fundingTime"] + 1
        time.sleep(0.2)
    
    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms")
    df["funding_rate"] = df["fundingRate"].astype(float)
    df = df[["timestamp", "funding_rate"]].copy()
    df.set_index("timestamp", inplace=True)
    return df


# ─── 3. Binance Open Interest (futures) ────────────────────────────────────

def fetch_binance_oi(symbol: str = "BTCUSDT", period: str = "1h",
                     days: int = 90) -> pd.DataFrame:
    """Fetch open interest history from Binance futures.
    Note: Binance limits OI history to ~30-90 days depending on period.
    """
    url = "https://fapi.binance.com/futures/data/openInterestHist"
    limit = 500
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 24 * 3600 * 1000
    
    all_rows = []
    current = start_ms
    
    while current < end_ms:
        params = {
            "symbol": symbol, "period": period,
            "startTime": current, "limit": limit
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        all_rows.extend(data)
        current = data[-1]["timestamp"] + 1
        time.sleep(0.5)
    
    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["open_interest"] = df["sumOpenInterest"].astype(float)
    df["oi_value"] = df["sumOpenInterestValue"].astype(float)
    df = df[["timestamp", "open_interest", "oi_value"]].copy()
    df.set_index("timestamp", inplace=True)
    return df


# ─── 4. Fear & Greed Index ─────────────────────────────────────────────────

def fetch_fear_greed(days: int = 730) -> pd.DataFrame:
    """Fetch crypto Fear & Greed index from alternative.me (free, no key)."""
    url = f"https://api.alternative.me/fng/?limit={days}&format=json"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()["data"]
    
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
    df["fear_greed"] = df["value"].astype(int)
    df = df[["timestamp", "fear_greed"]].copy()
    df.sort_values("timestamp", inplace=True)
    df.set_index("timestamp", inplace=True)
    return df


# ─── 5. BTC Dominance (CoinGecko) ─────────────────────────────────────────

def fetch_btc_dominance(days: int = 730) -> pd.DataFrame:
    """Fetch BTC market dominance % from CoinGecko (free, no key for this)."""
    url = "https://api.coingecko.com/api/v3/global"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    current_dom = resp.json()["data"]["market_cap_percentage"]["btc"]
    
    # CoinGecko free tier doesn't give historical dominance easily
    # Use market_chart for total market cap and BTC market cap to derive it
    btc_url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": min(days, 365), "interval": "daily"}
    resp_btc = requests.get(btc_url, params=params, timeout=30)
    resp_btc.raise_for_status()
    btc_caps = resp_btc.json()["market_caps"]
    
    total_url = "https://api.coingecko.com/api/v3/global/market_cap_chart"
    # This endpoint may require a key; fallback to approximation
    # We'll use BTC market cap as a proxy signal (dominance ≈ btc_cap / total_cap)
    
    df = pd.DataFrame(btc_caps, columns=["timestamp_ms", "btc_market_cap"])
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms").dt.normalize()
    df["btc_market_cap"] = df["btc_market_cap"].astype(float)
    df = df[["timestamp", "btc_market_cap"]].copy()
    df.set_index("timestamp", inplace=True)
    
    # Derive dominance change as signal (rising btc_cap relative to its mean)
    return df


# ─── Master fetcher ────────────────────────────────────────────────────────

def fetch_all(symbol: str = "BTCUSDT", days: int = 730,
              save: bool = True) -> dict:
    """Fetch all data sources and optionally save to parquet files."""
    print(f"Fetching OHLCV for {symbol}...")
    ohlcv = fetch_binance_ohlcv(symbol, days=days)
    
    print(f"Fetching funding rates for {symbol}...")
    funding = fetch_binance_funding(symbol, days=days)
    
    print(f"Fetching open interest for {symbol}...")
    try:
        oi = fetch_binance_oi(symbol, days=min(days, 90))
    except Exception as e:
        print(f"  OI fetch failed ({e}), will use empty df")
        oi = pd.DataFrame(columns=["open_interest", "oi_value"])
    
    print("Fetching Fear & Greed index...")
    fg = fetch_fear_greed(days=days)
    
    print("Fetching BTC market cap (dominance proxy)...")
    try:
        dom = fetch_btc_dominance(days=days)
    except Exception as e:
        print(f"  Dominance fetch failed ({e}), will use empty df")
        dom = pd.DataFrame(columns=["btc_market_cap"])
    
    result = {
        "ohlcv": ohlcv, "funding": funding,
        "oi": oi, "fear_greed": fg, "dominance": dom
    }
    
    if save:
        for name, df in result.items():
            if not df.empty:
                path = DATA_DIR / f"{symbol.lower()}_{name}.parquet"
                df.to_parquet(path)
                print(f"  Saved {path.name} ({len(df)} rows)")
    
    return result


if __name__ == "__main__":
    fetch_all("BTCUSDT", days=730)
    fetch_all("ETHUSDT", days=730)
    print("Done — all data saved to data/")

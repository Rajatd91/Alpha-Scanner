# Crypto Alpha Signal Scanner & Backtester

A robust Python research platform that computes z-score trading signals from 6 independent public sources, combines them into a composite signal, and backtests long/short strategies on major cryptocurrencies.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Seaborn-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)

## Data Sources

| # | Source | Signal | Logic | History |
|---|--------|--------|-------|---------|
| 1 | Binance Futures | Perpetual funding rate | Contrarian: extreme negative funding → buy | 2 years |
| 2 | Alternative.me | Fear & Greed Index | Contrarian: extreme fear → buy | 2 years |
| 3 | Binance Futures | Open interest changes | Contrarian: rapid OI spike without price → overleveraged | ~90 days* |
| 4 | Binance Futures | Top Trader Long/Short Ratio | Trend: follow smart-money positioning | ~90 days* |
| 5 | CoinGecko | BTC dominance % | Rising dominance → risk-off signal | 1 year |
| 6 | Price data | Fast/slow MA crossover | Momentum: trend-following | 2 years |

*All sources are free and require no API keys.*

> **Note on OI & LS Ratio:** Binance caps historical open interest and long/short ratio data at ~90 days. For 2-year backtests these signals default to 0 outside that window, which is why the optimizer assigns them low weight. This is an API limitation, not a signal quality issue — intraday traders with shorter lookbacks would weight them much higher.

## Two Ways to Run

This project provides two complementary interfaces:

### 1. Static Report Generation (Matplotlib)
Run the automated pipeline to fetch live data, compute all OOS metrics, and export professional `.png` charts directly to the `outputs/` folder (used to generate the images below).
```bash
# Fetch 2 years of live data and run the report
python main.py --symbol BTCUSDT --fetch
```

### 2. Interactive Dashboard (Streamlit)
Spin up the local web UI to explore the data dynamically and adjust thresholds, vol targets, and signal weights via sliders.
```bash
streamlit run app.py
```

## Strategy Findings & Analysis (BTCUSDT)

Running the automated backtester over 2 years of hourly data with `src/optimizer.py` — 5,000 Monte Carlo weight trials, **fit on in-sample data only, validated on a held-out OOS window**.

**Key findings:**

* **What drove alpha:** The optimizer consistently favoured **Price Momentum (24h vs 168h MA, ~44% weight)** and **BTC Dominance (~35% weight)**, muting the contrarian indicators. This makes intuitive sense: hourly BTC/ETH exhibit persistent short-term trends, and periods of rising BTC dominance (capital rotating out of alts) coincide with BTC strength.

* **Why contrarian signals underperformed:** Funding rate and Fear & Greed signals show strong signal decay at the 1–4h horizon but mean-revert quickly. After 5bps one-way transaction costs, they become net-negative at hourly frequency. They would likely recover weight on a daily strategy with lower turnover.

* **OI and LS ratio limitation:** Binance only exposes ~90 days of open interest and long/short ratio history. In a 2-year backtest these signals are zero-padded for the first ~21 months, which mechanically suppresses their optimizer weight. This is a data availability issue, not a signal quality verdict.

* **Walk-forward robustness:** The OOS window is divided into 4 equal sub-periods. A strategy that only works in one sub-period (lucky split) is immediately visible. Consistent positive Sharpe across all 4 folds is the target. This is a stronger robustness test than the standard single IS/OOS split.

* **Turnover & transaction costs:** Volatility targeting dampens position churn considerably. Without it, hourly signal oscillations produce ~3× more turnover and erode ~40% of gross alpha at 5bps one-way cost.

### Generated Visual Outputs

Below are the charts generated automatically by `main.py` using Seaborn:

#### Strategy Equity Curve & Drawdown
Displays the continuous position sizing logic and the Out-Of-Sample split line.
![Equity Curve](outputs/equity_curve.png)

#### Individual Z-Score Signals vs Composite
Demonstrates how the different mean-0, std-1 signals combine into the final composite (bottom).
![Signals](outputs/signals.png)

#### Signal Decay
Measures average forward return grouped by signal quintiles. Ideally, Q5 (strongest bullish signals) should cleanly outperform Q1 (strongest bearish).
![Signal Decay](outputs/signal_decay.png)

#### Position Distribution
Shows the frequency distribution of continuous position sizes outputted by the volatility-targeting engine.
![Position Distribution](outputs/position_distribution.png)


## Project Structure

```
alpha-scanner/
├── app.py                    # Streamlit interactive dashboard
├── main.py                   # Matplotlib CLI runner & report generator
├── generate_sample_data.py   # Synthetic data generator for quick demos
├── requirements.txt
├── outputs/                  # Exported .png charts
├── data/                     # Local parquet cache
└── src/
    ├── data_fetcher.py       # API calls: Binance, Alternative.me, CoinGecko
    ├── signals.py            # Z-score normalization & signal logic
    ├── backtester.py         # Position sizing, execution, and PnL metrics
    └── optimizer.py          # Monte-carlo Sharpe ratio parameter optimization
```

## Performance Metrics

The backtester calculates comprehensive institutional metrics:
- Sharpe ratio, Sortino ratio, Calmar ratio
- Max drawdown, win rate, avg win/loss
- Total turnover and position changes (cost modeling)
- In-Sample vs. Out-of-Sample metrics split
- **Walk-forward validation** across 4 equal OOS sub-periods (robustness check)

## Author

Rajat Durge — MSc Emerging Digital Technologies, UCL

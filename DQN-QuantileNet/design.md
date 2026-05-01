# DQN-QuantileNet — Dashboard Design Brief

## Overview

Build a web dashboard for analyzing the DQN-QuantileNet trading model — a Temporal Fusion Transformer trained to predict cryptocurrency price movements. The app has three main sections:

1. **Backtest Explorer** — analyze results of walk-forward out-of-sample backtests
2. **Strategy Simulator** — browse and filter top strategy combinations (EV / SQN ranking)
3. **Live Predictions** — display real-time model inference output as new candles arrive

---

## What This System Does

The model (QuantileNet) ingests multi-timeframe OHLCV candle data for crypto pairs (BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT) and outputs two types of predictions for each window:

- **Quantile predictions**: predicted return levels at fixed probability quantiles (10th, 25th, 50th, 75th, 90th percentile) — tells you "there's a 75% chance price moves at least X% in the next 24h"
- **Threshold predictions**: probability of exceeding fixed return thresholds (1%, 1.5%, 2%, 3%, 5%, 8%) — tells you "there's a 34% chance of a +3% move in the next 8h"

Both are produced for **long** and **short** directions, across **5 horizons**: 8h, 12h, 24h, 48h, 72h.

---

## Section 1 — Backtest Explorer

### What it shows
Walk-forward test results: model was trained on rolling 8-month windows, validated on 2 months, and tested on a held-out 6-month block.

### Data source
JSON file written after each backtest: `python/checkpoints/<run_id>/calibration.json`

### Data structure (calibration.json)
```json
{
  "generated_at": "2026-04-25T03:36:49.155000+00:00",
  "threshold_calibration": [
    {
      "horizon_h": 8,
      "direction": "long",
      "threshold_pct": 1.0,
      "mean_pred_pct": 19.4,
      "actual_pct": 17.6,
      "error_pct": 1.8,
      "n": 17280
    }
  ],
  "quantile_coverage": [
    {
      "horizon_h": 8,
      "direction": "long",
      "quantile": 0.10,
      "expected_pct": 10.0,
      "actual_pct": 7.0,
      "error_pct": -3.0
    }
  ]
}
```

### Visualizations needed

**Threshold Calibration Chart**
- X axis: predicted probability (mean_pred_pct)
- Y axis: actual hit rate (actual_pct)
- Perfect calibration = diagonal line y = x
- One chart per direction (long / short), color-coded by horizon (8h, 12h, 24h, 48h, 72h)
- Show all threshold levels as dots on each line

**Quantile Coverage Chart**
- Bar chart: for each quantile (10th, 25th, 50th, 75th, 90th), show expected% vs actual%
- Error bars showing deviation from ideal
- Separate panels for long and short

**Summary Table**
- Rows: horizon × direction × threshold combinations
- Columns: Mean Pred%, Actual%, Error%, N
- Color-code Error% column: green if |error| < 3%, yellow < 8%, red >= 8%

---

## Section 2 — Strategy Simulator (Top Combos)

### What it shows
Results of brute-force simulation across all parameter combinations: horizon × threshold × entry_prob × symbol. The best 100 combinations are ranked by SQN.

### Data source
This data is **currently only printed to stdout**. It needs to be exported to JSON. The Python code should write a file: `python/checkpoints/<run_id>/strategy_top100.json`

The data already exists in the `all_rows` list in `brute_force_tp_sl_top20()` in `backtest.py`. It just needs to be serialized.

### Data structure (strategy_top100.json)
```json
{
  "generated_at": "2026-04-25T03:36:49+00:00",
  "test_range": {
    "start": "2025-10-19T08:29:00Z",
    "end": "2026-04-20T08:14:00Z"
  },
  "symbols": ["BNBUSDT", "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"],
  "entry_prob_grid": [0.34, 0.39, 0.46, 0.51, 0.56],
  "combos": [
    {
      "rank": 1,
      "symbol": "BTCUSDT",
      "horizon": 72,
      "threshold": 3.0,
      "direction": "both",
      "entry_prob": 0.46,
      "tp_pct": 8.3,
      "sl_pct": 6.1,
      "n_trades": 47,
      "p_win": 0.532,
      "avg_win": 7.4,
      "avg_loss": 6.1,
      "total_return": 12.8,
      "sharpe": 1.21,
      "sqn": 2.31,
      "ev": 0.60,
      "kelly": 0.154,
      "tp_rate": 0.532,
      "sl_rate": 0.340,
      "hz_rate": 0.128,
      "max_dd": -0.12,
      "trades": [
        {
          "symbol": "BTCUSDT",
          "entry_time": 1729331369000,
          "exit_time": 1729590569000,
          "return": 0.067,
          "exit_type": "tp",
          "tp_pct": 8.3,
          "sl_pct": 6.1
        }
      ]
    }
  ]
}
```

### Field descriptions
| Field | Type | Meaning |
|-------|------|---------|
| symbol | string | Crypto pair (BTCUSDT etc.) |
| horizon | int | Prediction horizon in hours (8 / 12 / 24 / 48 / 72) |
| threshold | float | Entry threshold % (1.0, 1.5, 2.0, 3.0, 5.0, 8.0) |
| direction | string | "long", "short", or "both" |
| entry_prob | float | Minimum model probability required to enter trade (0–1) |
| tp_pct | float | Average take-profit % (dynamic, from model's p90 quantile) |
| sl_pct | float | Average stop-loss % (dynamic, from model's p10 quantile) |
| n_trades | int | Number of trades executed |
| p_win | float | Win rate (0–1) |
| avg_win | float | Average winning trade return % |
| avg_loss | float | Average losing trade return % (positive number = loss magnitude) |
| total_return | float | Sum of all trade returns % |
| sharpe | float | Annualized Sharpe ratio |
| sqn | float | System Quality Number = sqrt(N) × mean_return / std_return |
| ev | float | Expected value per trade % |
| kelly | float | Kelly fraction (clamped 0–0.25) |
| tp_rate | float | Fraction of trades exiting at TP |
| sl_rate | float | Fraction of trades exiting at SL |
| hz_rate | float | Fraction of trades exiting at horizon (time expiry) |
| max_dd | float | Maximum drawdown (negative number) |
| trades | array | Individual trade log (entry/exit timestamps as Unix ms) |

### Trade exit types
- `tp` — price hit take-profit level before horizon
- `sl` — price hit stop-loss level before horizon
- `hz` — horizon expired without hitting TP or SL

### Visualizations needed

**Ranked Table (main view)**
- Filterable by: symbol, horizon, threshold, direction, min_trades
- Sortable by any column (default: SQN descending)
- Columns: Rank, Symbol, Horizon, Threshold%, Entry%, N, P(Win)%, Avg+%, Avg-%, Total%, Sharpe, SQN, EV%, Kelly, TP%, SL%
- Color coding: SQN column green > 2.0, yellow 1.0–2.0, red < 1.0
- Asterisk (*) on rows with n_trades < 30 (low sample size warning)

**Equity Curve**
- When user clicks a row, show the cumulative return curve for that combo's trade list
- X axis: trade entry date
- Y axis: cumulative return %
- Mark TP exits as green dots, SL as red dots, HZ as gray dots

**Trade Distribution**
- Histogram of individual trade returns for the selected combo
- Show mean and median lines

**Exit Type Breakdown**
- Donut/pie chart: TP% vs SL% vs HZ% for selected combo

---

## Section 3 — Live Predictions

### What it shows
The model's current predictions for each symbol, updated every time a new 15-minute candle closes. Shows what the model thinks will happen over the next 8h–72h.

### Data source
The Python system currently only does training + backtest. Live prediction output needs to be added. The model should write predictions to a JSON file periodically, e.g. `python/live_predictions.json`, which the dashboard polls or watches for changes.

### Data structure (live_predictions.json)
```json
{
  "generated_at": "2026-04-25T04:00:00Z",
  "model_checkpoint": "run_20260425_031514/ckpt_epoch9_best_val0.1645.pt",
  "predictions": [
    {
      "symbol": "BTCUSDT",
      "current_price": 94250.0,
      "current_time": "2026-04-25T04:00:00Z",
      "horizons": [
        {
          "horizon_h": 8,
          "target_time": "2026-04-25T12:00:00Z",
          "long": {
            "quantiles": {
              "p10": 0.4,
              "p25": 1.1,
              "p50": 2.3,
              "p75": 4.8,
              "p90": 8.2
            },
            "threshold_probs": {
              "1.0pct": 0.71,
              "1.5pct": 0.58,
              "2.0pct": 0.44,
              "3.0pct": 0.28,
              "5.0pct": 0.11,
              "8.0pct": 0.04
            }
          },
          "short": {
            "quantiles": {
              "p10": 0.3,
              "p25": 0.9,
              "p50": 1.8,
              "p75": 3.7,
              "p90": 6.9
            },
            "threshold_probs": {
              "1.0pct": 0.62,
              "1.5pct": 0.49,
              "2.0pct": 0.37,
              "3.0pct": 0.21,
              "5.0pct": 0.08,
              "8.0pct": 0.02
            }
          }
        }
      ]
    }
  ]
}
```

### Field descriptions for live predictions
- `quantiles.p10` through `p90` — predicted return in %, e.g. `p90: 8.2` means model predicts 90% chance price moves less than +8.2% upward
- `threshold_probs.1.0pct` — model's estimated probability of a +1% move occurring within the horizon

### Visualizations needed

**Symbol Cards**
- One card per symbol (BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT)
- Shows current price and time of last update

**Prediction Fan Chart**
- For each symbol and horizon, show a fan/cone chart:
  - Center line = p50 predicted return
  - Shaded bands: p25–p75 (darker) and p10–p90 (lighter)
  - Both long (green) and short (red) cones
  - X axis: time (8h, 12h, 24h, 48h, 72h)
  - Y axis: return %

**Probability Bar Chart**
- For selected symbol, show bar chart of threshold probabilities
- X axis: threshold levels (1%, 1.5%, 2%, 3%, 5%, 8%)
- Y axis: probability (0–1)
- Blue bars = long, Orange bars = short
- Horizontal dashed line at current entry_prob from best strategy combo

**Signal Status**
- Based on the best strategy combo from Section 2, show whether any symbol is currently generating an entry signal
- Green badge "LONG SIGNAL" or red badge "SHORT SIGNAL" or gray "NO SIGNAL"
- Shows which horizon+threshold combination triggered it and at what probability

---

## Architecture Notes

### File paths (relative to project root `DQN-QuantileNet/`)
```
python/checkpoints/<run_id>/calibration.json         # Section 1 data
python/checkpoints/<run_id>/strategy_top100.json     # Section 2 data (needs to be added)
python/live_predictions.json                          # Section 3 data (needs to be added)
```

### What still needs to be implemented in Python

1. **Export strategy_top100.json** — in `backtest.py`, after `_print_top100_ev()` call, serialize `all_rows` (the full ranked list, not just top 100 for display) to JSON with timestamp and metadata
2. **Live prediction writer** — after each new 15m candle, run model inference on latest window and write `live_predictions.json`

### Technology stack suggestion
- Framework: Next.js or plain React with Vite
- Charts: Recharts or Plotly.js (good for financial data / fan charts)
- Styling: Tailwind CSS
- Data: poll JSON files every 30 seconds (or use file watcher via WebSocket)
- No backend needed — static file serving from the `DQN-QuantileNet/` directory

### Color conventions
- Long / bullish: green (`#22c55e`)
- Short / bearish: red (`#ef4444`)
- Neutral / HZ exit: gray (`#6b7280`)
- Warning (low n_trades): amber (`#f59e0b`)
- Good SQN (>2): green; mediocre (1–2): yellow; bad (<1): red

---

## Data Freshness

| Data | Written when | Staleness acceptable |
|------|-------------|----------------------|
| calibration.json | After each backtest run (every N epochs during training, or manual backtest.py run) | Hours |
| strategy_top100.json | Same as calibration | Hours |
| live_predictions.json | After each 15m candle close | 15 minutes max |

---

## Example Numbers (from real run, epoch 9)

- Test period: 2025-10-19 to 2026-04-20 (6 months)
- Test windows: 17,280 (one per 15m candle × 5 symbols, filtered to test block)
- Best combo: BTCUSDT, 72h horizon, 3% threshold, entry_prob=0.46 → SQN=2.31, EV=+0.60%, P(Win)=53.2%, 47 trades
- Calibration quality: at 1% threshold the model is well-calibrated (error ~2–6%), at higher thresholds it overestimates probability (error up to +15%)
- Short-direction model tends to underestimate probabilities at low thresholds (negative errors)

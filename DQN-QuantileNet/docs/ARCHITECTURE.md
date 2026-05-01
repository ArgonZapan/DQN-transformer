# DQN-QuantileNet — Architecture

## Overview

DQN-QuantileNet is a standalone probabilistic forecasting network that predicts future price movement distributions. It operates independently from DQN-OPUS — separate training, separate weights, separate pipeline — but shares the same Node.js data infrastructure and ZMQ communication pattern.

**Core question answered:** "What is the probability distribution of price returns over the next N hours?"

---

## Output Modes

Configurable via `config.toml`. Three modes available, can run simultaneously.

### Mode A — `quantile`
Fixed quantile level, variable price threshold.

> "At the 10th percentile, price will move by **Y%** in the next 4h."

Output shape: `[n_symbols, n_timeframes, n_horizons, n_quantiles, 2]`
- Last dim: `[long_threshold, short_threshold]` (when direction = `both`)

### Mode B — `threshold`
Fixed price threshold, variable probability.

> "There is **X%** chance price moves by 1% or more in the next 4h."

Output shape: `[n_symbols, n_timeframes, n_horizons, n_thresholds, 2]`

### Mode C — `both`
Both heads active simultaneously. The model learns one shared TFT backbone, two output heads.

```
TFT Backbone
     │
  ┌──┴──┐
  │     │
Head A  Head B
(quantile) (threshold)
```

---

## Direction Modes

```toml
direction = "both"   # "long" | "short" | "both"
```

- `long`  — predictions for upward price movement only
- `short` — predictions for downward price movement only
- `both`  — symmetric outputs for both directions (doubles output width)

---

## Architecture — Temporal Fusion Transformer (TFT)

TFT is chosen over a plain Transformer encoder because it was designed specifically for probabilistic time series forecasting. Key advantages:

- **Variable Selection Network (VSN)** — learns which features matter per timestep
- **LSTM encoder** before attention — captures local temporal dependencies
- **Gated Residual Networks (GRN)** — filters irrelevant signals, prevents gradient issues
- **Interpretable attention** — attention weights show which historical timesteps drove each prediction

### Data flow

```
Multi-timeframe candles (1m + 15m + 1h + ...)
           │
    [Feature normalization]
           │
    [Variable Selection Network]
           │
    [LSTM Encoder per timeframe]
           │
    [Temporal attention fusion]         ← fuses across timeframes
           │
    [Static enrichment + GRN]
           │
    ┌──────┴──────┐
    │             │
[Quantile head]  [Threshold head]
    │             │
 Y values       X probabilities
```

---

## Multi-Timeframe Input (Option B)

All configured timeframes are fed as simultaneous inputs to a single model.

```toml
[timeframes]
candles_1m  = 60   # 60 candles of 1m data
candles_15m = 32   # 32 candles of 15m data
candles_1h  = 24   # 24 candles of 1h data
candles_4h  = 0    # disabled
```

Setting a timeframe to `0` disables it entirely — no code changes required.

The Variable Selection Network learns independently which timeframe is informative for each horizon. Short horizons (1h) lean on 1m candles; longer horizons (8h, 24h) lean on 1h candles.

---

## Input Features

Same 11 features as DQN-OPUS, computed by the same Node.js pipeline:

| Index | Feature       | Description                        |
|-------|---------------|------------------------------------|
| 0     | open_norm     | Normalized open price              |
| 1     | high_norm     | Normalized high                    |
| 2     | low_norm      | Normalized low                     |
| 3     | close_norm    | Normalized close                   |
| 4     | volume_norm   | Normalized volume (clipped at 3σ)  |
| 5     | rsi           | RSI(14) / 100                      |
| 6     | macd_hist     | MACD histogram / close             |
| 7     | sma20_dist    | (close − SMA20) / close            |
| 8     | bb_width      | Bollinger band width               |
| 9     | stoch_k       | Stochastic %K(14)                  |
| 10    | pos_feature   | Position/context feature           |

---

## Prediction Targets

### Horizons
Configurable list of forecast horizons in hours. Any number of horizons supported simultaneously.

```toml
[prediction]
horizons_hours = [1, 2, 4, 8, 12, 24]
```

Each horizon maps to a fixed number of candles at the base timeframe (1m → multiply by 60).

### Quantile levels (mode: quantile)
```toml
quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
```

### Threshold levels (mode: threshold)
```toml
thresholds_pct = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
```

Values are percentage price movements (±8% max). The model predicts the probability of exceeding each threshold within the forecast horizon.

---

## Training Pipeline

Mirrors DQN-OPUS infrastructure:

| Component     | Reused from DQN-OPUS         | Notes                            |
|---------------|------------------------------|----------------------------------|
| Data download | `node/data/` (same scripts)  | Same Binance CSV format          |
| ZMQ transport | Same ZMQ PUSH/PULL pattern   | Different ports                  |
| Config loader | Same TOML structure          | Separate `config.toml`           |
| TensorBoard   | Same `runs/` convention      | No dashboard, no monitor         |
| Checkpoints   | Same `.pt` format            | Separate `checkpoints/` folder   |

**No dashboard. No monitor. TensorBoard only.**

### Loss functions

- **Quantile head** — Pinball loss (quantile regression loss):
  `L(τ) = τ · max(y−ŷ, 0) + (1−τ) · max(ŷ−y, 0)`

- **Threshold head** — Binary cross-entropy per threshold level

When `mode = "both"`, total loss = `pinball_loss + bce_loss` (weighted, configurable).

---

## Project Structure (planned)

```
DQN-QuantileNet/
├── config.toml              # all configuration
├── train.bat                # start training
├── tensorboard.bat          # launch TensorBoard
├── docs/
│   ├── ARCHITECTURE.md      # this file
│   └── CONFIG.md            # config reference
├── node/                    # data pipeline (mirrors DQN-OPUS/node)
│   ├── index.js             # entry point
│   ├── actors/              # data actors per symbol
│   ├── data/                # historical data + indicators
│   └── config.js
└── python/
    ├── main.py              # training entry point
    ├── config.py            # config loader
    ├── model/
    │   ├── tft.py           # Temporal Fusion Transformer
    │   ├── vsn.py           # Variable Selection Network
    │   ├── grn.py           # Gated Residual Network
    │   └── heads.py         # Quantile + Threshold output heads
    ├── training/
    │   ├── trainer.py       # training loop
    │   ├── dataset.py       # supervised dataset builder
    │   └── losses.py        # pinball loss + BCE
    ├── server/
    │   └── zmq_server.py    # ZMQ data receiver
    └── utils/
        └── tensorboard.py   # TensorBoard logging
```

---

## Key Differences from DQN-OPUS

| Aspect            | DQN-OPUS                     | DQN-QuantileNet              |
|-------------------|------------------------------|------------------------------|
| Learning paradigm | Reinforcement Learning (DQN) | Supervised learning          |
| Output            | Q-values per action          | Price return distribution    |
| Architecture      | Transformer encoder          | Temporal Fusion Transformer  |
| Training signal   | TD error / reward            | Realized future returns      |
| Dashboard         | Yes                          | No                           |
| Monitor           | Yes                          | No                           |
| TensorBoard       | Yes                          | Yes                          |
| Multi-timeframe   | Input features               | Input + output horizons      |

# DQN-QuantileNet — Config Reference

Full reference for `config.toml`. All sections are required unless marked optional.

---

## Symbols

```toml
[[actors]]
symbol = "BTCUSDT"
exchange = "binance"

[[actors]]
symbol = "ETHUSDT"
exchange = "binance"
```

Each `[[actors]]` block adds one symbol to the training data. Predictions are generated per symbol independently.

---

## Timeframes

```toml
[timeframes]
candles_1m  = 60   # number of 1m candles in input window (0 = disabled)
candles_15m = 32   # number of 15m candles
candles_1h  = 24   # number of 1h candles
candles_4h  = 0    # disabled
candles_1d  = 0    # disabled
```

- Set any timeframe to `0` to disable it with no other changes needed.
- Multiple active timeframes are fused inside the TFT model.
- At least one timeframe must be active.

---

## Prediction

```toml
[prediction]
mode      = "both"       # "quantile" | "threshold" | "both"
direction = "both"       # "long" | "short" | "both"

horizons_hours = [1, 2, 4, 8, 12, 24]   # forecast horizons (any count, in hours)

# Mode: quantile — predict price threshold at fixed quantile levels
quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

# Mode: threshold — predict probability of exceeding fixed price thresholds
thresholds_pct = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

# Weight of each loss when mode = "both"
quantile_loss_weight  = 0.5
threshold_loss_weight = 0.5
```

### horizons_hours
Each value generates a separate prediction target. For a 1m base timeframe, `4` hours = 240 candles look-ahead.

### quantiles
Values must be in (0, 1). Interpreted as: "at quantile τ, price will move by Y%."

### thresholds_pct
Values are absolute percentage price changes. Max recommended: 8.0%. Interpreted as: "probability that price moves by X% or more."

---

## Model

```toml
[model]
# TFT backbone
d_model          = 64     # hidden dimension throughout the model
n_attention_heads = 4     # multi-head attention heads
lstm_layers      = 2      # LSTM encoder depth
lstm_hidden      = 64     # LSTM hidden state size
grn_dropout      = 0.1    # dropout inside Gated Residual Networks
vsn_dropout      = 0.1    # dropout inside Variable Selection Network
ff_dim           = 128    # feed-forward layer dimension
n_transformer_blocks = 2  # number of TFT attention blocks
```

---

## Training

```toml
[training]
lr            = 1.0e-4
batch_size    = 256
epochs        = 100                    # max training epochs
patience      = 10                     # early stopping: stop after N epochs without improvement
seed          = 42
device        = "cuda"                 # "cuda" | "cpu" | "auto"
num_threads   = 12

validation_days   = 30                 # last N days reserved as out-of-sample validation
training_months   = 48                 # months of training data (0 = full history)

checkpoint_interval = 5               # save checkpoint every N epochs
keep_last_n_checkpoints = 10
resume_from_checkpoint = ""           # path to .pt file, or "" for fresh training

compile_model = false                  # torch.compile (PyTorch 2+)

lr_scheduler = "cosine"               # "none" | "cosine"
lr_min       = 1.0e-5                 # cosine scheduler floor
```

---

## ZMQ

```toml
[zmq]
host        = "tcp://127.0.0.1"
port        = 6555              # ZMQ PULL — receives data batches from Node.js actors
metrics_port = 6556             # ZMQ PUB — sends metrics back (optional)
```

Ports are offset from DQN-OPUS defaults (5555/5556) to allow both networks to run simultaneously.

---

## Features

```toml
[features]
num_features      = 11
normalization_window = 60   # rolling window for mean/std normalization
```

Must match the Node.js `buildFeatures()` output. Do not change `num_features` without updating the data pipeline.

---

## Data

```toml
[data]
source  = "file"                    # "file" | "api"
path    = "node/data/historical/"
cache_path = "node/data/cache/"
binance_rate_limit    = 1000        # ms between API requests
allow_partial_history = false       # true = run even if some TF data is missing
api_retry_interval_sec = 60
```

---

## Logging

```toml
[logging]
level          = "INFO"    # DEBUG | INFO | WARNING | ERROR
max_file_size_mb = 10
max_files      = 5
log_dir        = "logs"
```

---

## TensorBoard

```toml
[tensorboard]
log_dir                = "runs"
log_interval_sec       = 300   # how often to write scalar metrics
histogram_interval_sec = 600   # how often to write weight histograms
```

Metrics logged per training step:
- `loss/total`, `loss/quantile`, `loss/threshold`
- `metrics/pinball_loss_mean`, `metrics/bce_loss_mean`
- `metrics/quantile_coverage` — fraction of true values within predicted quantile bounds
- `metrics/calibration_error` — expected vs actual quantile coverage
- `lr`, `epoch`, `grad_norm`
- Weight and gradient histograms per layer

---

## Alerts (optional)

```toml
[alerts]
enabled          = false
telegram_token   = ""
telegram_chat_id = ""
cooldown_sec     = 300
grad_explode_threshold = 5.0
loss_plateau_steps     = 2000
```

---

## Minimal working config

```toml
[[actors]]
symbol = "BTCUSDT"
exchange = "binance"

[timeframes]
candles_1m  = 60
candles_15m = 32
candles_1h  = 24

[prediction]
mode      = "both"
direction = "both"
horizons_hours    = [1, 4, 8, 24]
quantiles         = [0.10, 0.25, 0.50, 0.75, 0.90]
thresholds_pct    = [0.5, 1.0, 2.0, 4.0, 8.0]

[model]
d_model           = 64
n_attention_heads = 4
lstm_layers       = 2
lstm_hidden       = 64
n_transformer_blocks = 2

[training]
lr         = 1.0e-4
batch_size = 256
device     = "cuda"

[zmq]
host = "tcp://127.0.0.1"
port = 6555

[features]
num_features         = 11
normalization_window = 60

[data]
source     = "file"
path       = "node/data/historical/"
cache_path = "node/data/cache/"

[logging]
level   = "INFO"
log_dir = "logs"

[tensorboard]
log_dir = "runs"
```

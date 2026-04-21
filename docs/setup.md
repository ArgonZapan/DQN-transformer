# Setup i konfiguracja

## Przegląd

Konfiguracja systemu opiera się na **jednym pliku TOML** — zero hardcoded zmiennych w kodzie. Każdy moduł ładuje swoją sekcję.

## Konfiguracja config.toml

### Pełny przykład (aktualna konfiguracja)

Poniższy fragment odzwierciedla kluczowe sekcje aktualnego `config.toml`. Pełna konfiguracja w pliku źródłowym.

```toml
[learner]
host = "tcp://127.0.0.1"
port = 5555
metrics_port = 5556
device = "cuda"   # "cuda" | "cpu" | "auto"
num_threads = 12

[[actors]]
symbol = "BTCUSDT"
exchange = "binance"
leverage = 1

# ... (ETH, SOL, BNB, XRP analogicznie)

[timeframes]
candles_1m  = 60
candles_15m = 32
candles_1h  = 48
candles_1d  = 14
candles_1w  = 0    # 0 = wyłączony

[features]
num_features = 11

[monitoring]
host = "tcp://127.0.0.1"
port = 3001
metrics_pull_port = 3002
metrics_push_interval_sec = 5
dashboard_poll_interval_sec = 60

[training]
gamma             = 0.97
lr                = 1.0e-4
batch_size        = 512
buffer_capacity   = 500000
min_buffer_size   = 100000
target_update_interval = 1000
epsilon_start     = 0.90
epsilon_end       = 0.10
epsilon_decay_fraction = 0.20
dropout           = 0.1
n_step            = 25
return_mode       = "nstep"
seed              = 42
resume_from_checkpoint = ""
keep_last_n_checkpoints = 5
checkpoint_interval = 5000
evaluation_interval = 10000
train_data_fraction = 0.8
min_episode_length = 100

[per]
alpha = 0.6
beta_start = 0.4
beta_end = 1.0
epsilon = 0.0001

[model]
num_actions = 4
n_transformer_blocks = 3
n_attention_heads = 8
key_dim = 64
ff_dim = 512
conv_kernel_size = 3
conv1d_filters = 128

[features]
num_features = 8
use_ohlcv = true
use_rsi = true
use_macd = true
use_sma20 = true

[reward]
commission_open = 0.001
commission_close = 0.001
trade_penalty = 0.001
clip_min = -1.0
clip_max = 1.0
intermediate_reward_max = 0.1
drawdown_penalty = 0.5
time_decay_hours = 1.0
confidence_scale = 1.0

[data]
source = "file"
path = "node/data/historical/"
cache_path = "node/data/cache/"
binance_rate_limit = 1000
normalization_window = 20
allow_partial_history = true
api_retry_interval_sec = 60

[logging]
level = "INFO"
max_file_size_mb = 10
max_files = 5
log_dir = "logs"

[backtesting]
enabled = false
model_path = "python/checkpoints/best_model.pt"
start_date = "2023-01-01"
end_date = "2023-12-31"
results_dir = "results"

[api]
# Klucz API ładowany ze zmiennej środowiskowej BINANCE_API_KEY
# NIE wpisuj go tutaj
simulation_mode = true
```

### Sekcje konfiguracji

| Sekcja | Opis |
|---|---|
| `[learner]` | Konfiguracja Python Learnera (host, porty ZMQ) |
| `[[actors]]` | Lista aktorów (wielokrotne wpisy) |
| `[timeframes]` | Globalne ustawienia timeframe'ów |
| `[monitoring]` | Monitoring Service i Dashboard |
| `[training]` | Parametry treningu, checkpointing, seed |
| `[per]` | Parametry Prioritized Experience Replay |
| `[model]` | Architektura modelu |
| `[features]` | Cechy danych — modularne wskaźniki |
| `[reward]` | System nagradzania i prowizje |
| `[data]` | Źródła danych, rate limiting, normalizacja |
| `[logging]` | Logowanie i rotacja plików |
| `[backtesting]` | Tryb ewaluacji out-of-sample |
| `[api]` | Ustawienia API (klucz przez zmienną środowiskową) |

### Parametry learnera

| Parametr | Opis | Domyślnie |
|---|---|---|
| host | Adres hosta ZMQ | tcp://127.0.0.1 |
| port | Port ZMQ REQ/REP (step/predict) | 5555 |
| metrics_port | Port ZMQ PUSH (metryki do Monitoring) | 5556 |
| device | Urządzenie do obliczeń | cuda |

### Parametry treningu

| Parametr | Opis | Domyślnie |
|---|---|---|
| gamma | Dyskontowanie przyszłych nagród | 0.999 |
| lr | Learning rate | 0.0001 |
| batch_size | Rozmiar batcha | 256 |
| buffer_capacity | Pojemność replay buffer | 500000 |
| min_buffer_size | Minimalne wypełnienie przed treningiem | 10000 |
| target_update_interval | Interwał update target network | 1000 |
| epsilon_start | Początkowe epsilon | 1.0 |
| epsilon_end | Końcowe epsilon | 0.05 |
| epsilon_decay_fraction | Frakcja kroków do decay | 0.3 |
| dropout | Dropout rate | 0.1 |
| n_step | N-step returns (1 = standardowy TD) | 1 |
| seed | Seed dla reprodukowalności (-1 = losowy) | 42 |
| resume_from_checkpoint | Ścieżka checkpointu do wznowienia | "" |
| keep_last_n_checkpoints | Liczba przechowywanych checkpointów | 5 |
| checkpoint_interval | Zapis co N kroków | 5000 |
| evaluation_interval | Ewaluacja OOS co N kroków | 10000 |
| train_data_fraction | Frakcja danych treningowych | 0.8 |
| min_episode_length | Min. długość epizodu w krokach | 100 |

### Parametry PER

| Parametr | Opis | Domyślnie |
|---|---|---|
| alpha | Stopień priorytetyzacji (0=losowe, 1=pełne) | 0.6 |
| beta_start | Początkowe IS weights | 0.4 |
| beta_end | Końcowe IS weights (rośnie do 1.0) | 1.0 |
| epsilon | Minimalna stała priorytetu (nie-zero) | 0.0001 |

### Parametry modelu

| Parametr | Opis | Domyślnie |
|---|---|---|
| num_actions | Liczba akcji (LONG/SHORT/HOLD/CLOSE) | 4 |
| n_transformer_blocks | Bloki Transformer | 3 |
| n_attention_heads | Głowice attention | 8 |
| key_dim | Wymiar klucza | 64 |
| ff_dim | Wymiar feed-forward | 512 |
| conv_kernel_size | Rozmiar kernela Conv1D | 3 |
| conv1d_filters | Liczba filtrów Conv1D | 128 |

## Uruchomienie

### run.bat (Windows)

Jednym poleceniem — plik .bat startuje wszystkie procesy:

```bash
run.bat
```

Uruchamia równolegle:
1. **Python Learner** — ZeroMQ, trening, predykcje
2. **Monitoring Service** — Node.js, agregacja metryk
3. **Actorzy** — Node.js (N instancji wg config.toml)
4. **Dashboard** — Vite + React w dev mode

### Ręczne uruchamianie

```bash
# Terminal 1 — Python Learner
cd python && python main.py

# Terminal 2 — Monitoring Service
cd monitoring && node server.js

# Terminal 3 — Actorzy (zarządza N instancjami)
cd node && node index.js

# Terminal 4 — Dashboard
cd dashboard && npm run dev
```

## Wymagania sprzętowe

Wymagania są **w pełni konfigurowalne** — można dostosować rozmiar batcha, pojemność bufora i wielkość modelu do posiadanego sprzętu.

### Profile sprzętowe

| Komponent | Profil Budget | Profil Mid | Profil High |
|---|---|---|---|
| GPU | RTX 3060 (12GB) | RTX 3080 (10GB) | RTX 3090 (24GB) |
| RAM | 16GB | 32GB | 64GB |
| Buffer capacity | 50k | 500k | 2M |
| Batch size | 64 | 256 | 512 |
| VRAM usage | ~4GB | ~8GB | ~12GB |

### Dostosowanie konfiguracji

Domyślna konfiguracja w `config.toml` zakłada **profil Mid** — wystarczy zmienić wartości w sekcjach `[training]` i `[model]` aby dostosować do swojego sprzętu.

**Budżet (RTX 3060, 16GB RAM):**
```toml
[training]
batch_size = 64
buffer_capacity = 50000

[model]
n_transformer_blocks = 1
n_attention_heads = 4
ff_dim = 256
```

**Wysoki (RTX 3090, 64GB RAM):**
```toml
[training]
batch_size = 512
buffer_capacity = 2000000

[model]
n_transformer_blocks = 6
n_attention_heads = 16
ff_dim = 1024
```

## Instalacja zależności

### Python

```bash
cd python
pip install -r requirements.txt
```

**Wymagane pakiety:**
- torch
- numpy
- pyzmq
- msgpack
- toml (loader config.toml)

> **Uwaga:** FastAPI NIE jest używany — cała komunikacja Python ↔ Node.js odbywa się przez ZeroMQ. Jeśli Monitoring Service wystawia REST API, robi to Node.js/Express, nie Python.

### Monitoring Service (Node.js)

```bash
cd monitoring
npm install
```

**Wymagane pakiety:**
- zeromq (PUSH/PULL dla metryk)
- express (HTTP REST API dla Dashboardu)
- cors (CORS dla dev mode)
- toml (loader config.toml)

### Node.js (Actors)

```bash
cd node
npm install
```

**Wymagane pakiety:**
- zeromq (REQ/REP + PUSH)
- msgpack5
- binance-api-node
- toml

### Dashboard

```bash
cd dashboard
npm install
```

**Wymagane pakiety:**
- vite
- react
- react-dom
- recharts (wykresy)

## Docker

```bash
docker-compose up -d
```

### docker-compose.yml

```yaml
version: '3'
services:
  learner:
    build:
      context: .
      dockerfile: docker/Dockerfile.python
    ports:
      - "5555:5555"
    volumes:
      - ./python:/app
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  monitoring:
    build:
      context: .
      dockerfile: docker/Dockerfile.node
    ports:
      - "3001:3001"
      - "3002:3002"
    volumes:
      - ./monitoring:/app
    depends_on:
      - learner

  actors:
    build:
      context: .
      dockerfile: docker/Dockerfile.node
    volumes:
      - ./node:/app
    depends_on:
      - learner
      - monitoring
```

## Troubleshooting

### Problem: ModuleNotFoundError

```bash
pip install -r requirements.txt
```

### Problem: Port already in use

Zmień porty w `config.toml` lub sprawdź co używa portu:

```bash
netstat -ano | findstr :5555
```

### Problem: CUDA out of memory

Zmniejsz batch size i buffer capacity:

```toml
[training]
batch_size = 64
buffer_capacity = 50000
```

### Problem: Actor nie może się połączyć

Sprawdź czy Learner jest uruchomiony:

```bash
telnet 127.0.0.1 5555
```

### Problem: Dashboard nie wyświetla danych

1. Sprawdź czy Monitoring Service działa
2. Sprawdź adres w `dashboard/src/config.js`
3. Otwórz konsolę przeglądarki i sprawdź błędy
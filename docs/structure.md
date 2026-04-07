# Struktura plików projektu

## Przegląd

```
trading-dqn/
│
├── config.toml                    ← jedna konfiguracja dla całego systemu
├── run.bat                        ← jedno uruchomienie (Windows)
│
├── python/
│   ├── main.py
│   ├── config.py                  ← loader config.toml (sekcja [learner])
│   ├── model/
│   │   ├── network.py
│   │   ├── regime_detector.py
│   │   └── noisy_linear.py
│   ├── training/
│   │   ├── replay_buffer.py
│   │   ├── trainer.py
│   │   └── prioritized_buffer.py
│   ├── server/
│   │   ├── zmq_server.py
│   │   └── schemas.py
│   ├── utils/
│   │   ├── normalizer.py
│   │   └── metrics.py
│   ├── monitoring/
│   │   └── monitor_client.py    ← ZeroMQ PUSH metryk do Monitoring Svc
│   └── checkpoints/
│
├── node/
│   ├── index.js
│   ├── config.js                  ← loader config.toml (sekcje [actors], [timeframes])
│   ├── env/
│   │   ├── tradingEnv.js
│   │   ├── reward.js
│   │   ├── state.js
│   │   └── episode.js
│   ├── data/
│   │   ├── binance.js
│   │   ├── indicators.js
│   │   ├── normalizer.js
│   │   ├── cache/                  ← cache bieżących danych
│   │   └── historical/             ← pobrane dane historyczne (CSV/Parquet)
│   ├── actors/
│   │   ├── actor.js
│   │   └── actorManager.js
│   ├── client/
│   │   └── pythonClient.js       ← ZeroMQ REQ/REP
│   └── monitoring/
│       └── monitor_client.js      ← ZeroMQ PUSH metryk do Monitoring Svc
│
├── monitoring/
│   ├── server.js                  ← Monitoring Service (Node.js)
│   └── config.js                  ← loader config.toml (sekcja [monitoring])
│
├── dashboard/
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx
│       └── components/
│
├── shared/
│   └── stateSchema.js
│
├── scripts/
│   ├── download_data.js
│   ├── evaluate.py
│   └── visualize.py
│
├── tests/
│   ├── python/
│   └── node/
│
├── debug/                         ← smoke-testy integracyjne
│   └── ...
│
├── docker/
│   ├── Dockerfile.python
│   ├── Dockerfile.node
│   └── docker-compose.yml
│
├── docs/                           ← dokumentacja projektu
│   ├── README.md                   ← główny plik nawigacyjny
│   ├── architecture.md             ← architektura systemu
│   ├── model.md                    ← model sieci neuronowej
│   ├── algorithm.md                ← algorytm DQN i Rainbow
│   ├── training.md                 ← proces treningu
│   ├── reward.md                   ← system nagradzania
│   ├── data.md                     ← dane wejściowe
│   ├── actors.md                   ← dokumentacja aktorów
│   ├── communication.md            ← komunikacja ZeroMQ
│   ├── monitoring.md               ← monitoring i dashboard
│   ├── backtesting.md              ← tryb backtesting i ewaluacja
│   ├── shutdown.md                 ← graceful shutdown systemu
│   ├── setup.md                    ← konfiguracja i uruchomienie
│   └── structure.md                ← struktura plików
│
├── datasets/                       ← dataset'y treningowe/eval (opcjonalnie)
│   ├── train/                      ← dane treningowe
│   └── eval/                       ← dane ewaluacyjne (out-of-sample)
│
├── results/                        ← wyniki backtestingu (JSON)
│   └── backtest_<timestamp>.json
│
├── logs/                           ← logi wszystkich modułów (auto-rotacja)
│   ├── learner.log
│   ├── actors.log
│   ├── monitoring.log
│   └── dashboard.log
│
├── .env.example                    ← przykładowe zmienne środowiskowe (bez wartości)
├── .gitignore                      ← wyklucza .env, checkpoints, dane historyczne
├── run.bat                         ← uruchomienie systemu (Windows)
├── stop.bat                        ← graceful shutdown (Windows)
└── Readme.md
```

## Opisy modułów

### Python (Learner)

| Plik | Opis |
|---|---|
| `main.py` | Entry point — inicjalizacja wszystkiego |
| `config.py` | Loader config.toml dla sekcji learner/training/model |
| `model/network.py` | Główna sieć Conv1D + Transformer + Dueling |
| `model/regime_detector.py` | Wykrywanie reżimu rynkowego |
| `model/noisy_linear.py` | Warstwy NoisyLinear dla eksploracji |
| `training/replay_buffer.py` | Pre-alokowany replay buffer z pinned memory |
| `training/trainer.py` | Logika treningu (loss, target update) |
| `training/prioritized_buffer.py` | Prioritized Experience Replay |
| `server/zmq_server.py` | Obsługa ZMQ requestów od Actorów |
| `server/schemas.py` | Schematy wiadomości MessagePack |
| `utils/normalizer.py` | Normalizacja danych wejściowych |
| `utils/metrics.py` | Obliczanie metryk (Sharpe, Drawdown, itp.) |
| `monitoring/monitor_client.py` | Wysyłanie metryk do Monitoring Service |
| `checkpoints/` | Zapisane wagi modelu |

### Node.js (Actors)

| Plik | Opis |
|---|---|
| `index.js` | Entry point — odpalenie ActorManager |
| `config.js` | Loader config.toml dla sekcji actors/timeframes |
| `env/tradingEnv.js` | Środowisko tradingowe (pozycje, stany) |
| `env/reward.js` | System nagradzania |
| `env/state.js` | Budowanie stanu z 5 timeframe'ów |
| `env/episode.js` | Zarządzanie epizodami |
| `data/binance.js` | Pobieranie danych z Binance API |
| `data/indicators.js` | Obliczanie wskaźników (RSI, MACD, SMA) |
| `data/normalizer.js` | Normalizacja danych w Node.js |
| `data/cache/` | Cache danych kryptowalutowych |
| `actors/actor.js` | Pojedyncza instancja aktora |
| `actors/actorManager.js` | Zarządzanie wieloma aktorami, batchowanie |
| `client/pythonClient.js` | Komunikacja ZMQ z Python Learner |
| `monitoring/monitor_client.js` | Wysyłanie metryk do Monitoring Service |

### Monitoring

| Plik | Opis |
|---|---|
| `server.js` | Serwis do agregacji metryk |
| `config.js` | Loader config.toml dla sekcji monitoring |

### Dashboard

| Plik | Opis |
|---|---|
| `package.json` | Zależności Node.js |
| `vite.config.js` | Konfiguracja Vite |
| `src/App.jsx` | Główny komponent React |
| `src/components/` | Komponenty wykresów i statystyk |

### Shared

| Plik | Opis |
|---|---|
| `stateSchema.js` | Wspólny schemat stanu dla Python i Node.js |

### Scripts

| Plik | Opis |
|---|---|
| `download_data.js` | Pobieranie danych historycznych z Binance |
| `evaluate.py` | Entry point backtestingu — ewaluacja out-of-sample |
| `visualize.py` | Wizualizacja wyników backtestingu |

### Tests

| Kierunek | Opis |
|---|---|
| `python/` | Testy jednostkowe dla kodu Python |
| `node/` | Testy jednostkowe dla kodu Node.js |

### Debug

Zawiera smoke-testy integracyjne do szybkiej weryfikacji czy komponenty działają razem.

### Docker

| Plik | Opis |
|---|---|
| `Dockerfile.python` | Image dla Python Learner |
| `Dockerfile.node` | Image dla Node.js components |
| `docker-compose.yml` | Orchestracja wszystkich kontenerów |

## Flow danych przez strukturę

```
1. Node.js/data/binance.js    — pobiera dane
2. Node.js/data/indicators.js — oblicza wskaźniki
3. Node.js/env/state.js       — buduje stan
4. Node.js/actors/actor.js    — wybiera akcję
5. Node.js/client/pythonClient.js — wysyła ZMQ do Pythona
6. Python/server/zmq_server.py  — odbiera request
7. Python/model/network.py      — predykcja
8. Python/training/trainer.py   — krok treningu
9. Python/training/replay_buffer.py — zapis doświadczenia
10. Python/monitoring/monitor_client.py — push metryk
11. Node.js/monitoring/monitor_client.js — push metryk
12. Monitoring/server.js        — agregacja
13. Dashboard/src/App.jsx       — wizualizacja
```

## Ładowanie konfiguracji

Każdy moduł ładuje tylko swoją sekcję:

```python
# Python (config.py)
import toml

config = toml.load('../config.toml')
learner_config = config['learner']
training_config = config['training']
model_config = config['model']
```

```javascript
// Node.js (config.js)
const toml = require('toml');
const fs = require('fs');

const configContent = fs.readFileSync('../config.toml', 'utf-8');
const config = toml.parse(configContent);

const actorsConfig = config.actors;  // Array
const timeframesConfig = config.timeframes;
```

## Zasady struktury

1. **Jedna konfiguracja** — `config.toml` w głównym katalogu
2. **Zero hardcoded zmiennych** — wszystko przez konfig
3. **Separacja odpowiedzialności** — każdy moduł ma jasny cel
4. **Testowalność** — `tests/` i `debug/` dla weryfikacji
5. **Skalowalność** — dodaj aktora przez konfig, zero zmian w kodzie
6. **Bezpieczeństwo** — klucz API przez zmienne środowiskowe, nigdy w `config.toml`
7. **Graceful shutdown** — `stop.bat` zamyka procesy w odpowiedniej kolejności
8. **Obserwowalność** — logi per moduł z rotacją, metryki w Monitoring Service
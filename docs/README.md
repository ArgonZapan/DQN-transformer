# Dokumentacja projektu Trading DQN

Dokumentacja projektu Trading DQN — architektura Actor-Learner Ape-X z modelem Conv1D + Transformer do tradingu algorytmicznego na parach kryptowalut (BTC/ETH/SOL).

## Spis treści

| Rozdział | Opis |
|---|---|
| [1. Architektura systemu](architecture.md) | Diagram architektury, podział odpowiedzialności, separacja Actor/Learner |
| [2. Model sieci neuronowej](model.md) | Architektura Conv1D + Transformer, Dueling DQN, parametry modelu |
| [3. Algorytm DQN i Rainbow](algorithm.md) | Double DQN, Prioritized Experience Replay, wzór Bellmana, target network |
| [4. Proces treningu](training.md) | Fazy treningu, warm-up, checkpointing, seed, wersjonowanie modelu |
| [5. System nagradzania](reward.md) | Formuły nagród, prowizje, clipping, typowe problemy |
| [6. Dane wejściowe](data.md) | Multi-scale temporal representation, cechy, normalizacja, rate limiting |
| [7. Aktorzy](actors.md) | Konfiguracja aktorów, batchowanie, epsilon-greedy, normalizacja per para |
| [8. Komunikacja](communication.md) | ZeroMQ protokoły, typy wiadomości, MessagePack |
| [9. Monitoring](monitoring.md) | Monitoring Service, Dashboard, metryki tradingowe |
| [10. Backtesting](backtesting.md) | Tryb ewaluacji, metryki, interpretacja wyników |
| [11. Graceful Shutdown](shutdown.md) | stop.bat, kolejność zamykania, obsługa SIGTERM |
| [12. Setup i konfiguracja](setup.md) | Instalacja, config.toml, bezpieczeństwo API, logowanie, walidacja |
| [13. Struktura projektu](structure.md) | Struktura plików i folderów |

## Szybki start

```bash
# Uruchomienie
run.bat

# Zatrzymanie
stop.bat
```

Pełna konfiguracja w pliku `config.toml` — zero hardcoded zmiennych w kodzie. Klucz API Binance przez zmienną środowiskową `BINANCE_API_KEY`.

## Architektura w skrócie

```
Node.js (Actor)                    Python (Learner)
├── Symulacja środowiska           ├── Replay Buffer
├── Logika tradingowa              ├── Trening modelu
└── Zarządzanie epizodami          └── Predykcje
        │                                  │
        └──────── ZeroMQ ──────────────────┘
```

## Stack technologiczny

| Moduł | Technologia |
|---|---|
| Learner | Python + PyTorch + ZeroMQ |
| Actor (N instancji) | Node.js + ZeroMQ |
| Monitoring Service | Node.js + Express + ZeroMQ |
| Dashboard | Vite + React (dev mode) |
| Komunikacja | ZeroMQ + MessagePack |
| Konfiguracja | TOML (jeden plik, zero hardcoded zmiennych) |
| Logowanie | Python logging + Winston (Node.js) |
| Bezpieczeństwo | Zmienne środowiskowe / .env (klucz API) |
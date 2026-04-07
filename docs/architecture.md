# Architektura systemu

## Przegląd

Projekt implementuje architekturę **Actor-Learner Ape-X** z modelem Conv1D + Transformer do tradingu algorytmicznego. System składa się z trzech głównych komponentów działających w osobnych procesach.

## Diagram architektury

```
┌─────────────────────────────────────────────────────────┐
│                        Node.js                          │
│                                                         │
│  Actor 1 ──┐                                            │
│  Actor 2 ──┼──► actorManager ──► ZMQ REQ/REP ──┐        │
│  Actor N ──┘         ▲                           │        │
│                        │                           ▼        │
│              ◄── nextAction ◄──── ZMQ REQ/REP             │
│                                                         │
│  Każdy moduł ──► PUSH metrics ──► Monitoring Svc         │
└─────────────────────────────────────────────────────────┘
     │                                    │
     ▼                                    ▼
┌─────────────┐              ┌─────────────────────────┐
│   Python    │              │   Node.js Monitoring     │
│             │              │   (agregacja metryk)      │
│  ZeroMQ ───►│              ├─────────────────────────┤
│  Replay Buf │              │   Dashboard (Vite+React) │
│  Trainer ──►│              │   pull co 1 min, GET     │
│  Model      │              │   tylko wyświetla        │
└─────────────┘              └─────────────────────────┘
```

## Podział odpowiedzialności

### Node.js (Actor)
Node.js odpowiada wyłącznie za **symulację środowiska rynkowego**:
- Pobieranie danych z giełdy (Binance API)
- Budowanie stanów (state) z wieloma timeframe'ami
- Obliczanie nagród za zamknięte pozycje
- Zarządzanie epizodami (start, krok, koniec)
- Wysyłanie doświadczeń do Python Learnera

### Python (Learner)
Python odpowiada wyłącznie za **wszystko związane z ML**:
- Przechowywanie doświadczeń w replay buforze
- Trening modelu sieci neuronowej
- Predykcje akcji dla Actorów
- Wysyłanie metryk treningowych

### Monitoring Service (Node.js)
Dedykowany serwis do **agregacji i udostępniania metryk**:
- Odbiera metryki od Actorów i Learnera (PUSH/PULL)
- Udostępnia dane dla Dashboardu (REQ/REP)
- Jest pasywny — nie modyfikuje danych

### Dashboard (Vite + React)
Interfejs użytkownika do **wizualizacji metryk**:
- Odpytuje Monitoring Service co 1 minutę
- Wyświetla: loss curve, epsilon decay, equity curve, Sharpe ratio
- Tylko tryb GET — nie modyfikuje danych

## Dlaczego taki podział

| Aspekt | Uzasadnienie |
|---|---|
| **Node.js dla tradingu** | Dojrzałe biblioteki do Binance API, naturalne dla logiki tradingowej |
| **Python dla ML** | Natywna obsługa CUDA, standard dla RL (Stable Baselines, RLlib, CleanRL) |
| **ZeroMQ** | Czysta separacja odpowiedzialności, niskie opóźnienia (~0.1ms), automatyczny reconnect |
| **Testowalność** | Każdą część można testować niezależnie |
| **Wymienialność** | Model można wymienić bez dotykania logiki tradingowej i odwrotnie |

## Wzorzec Actor-Learner

Implementacja opiera się na wzorcu **Actor-Learner**:

- **Actor (Node.js)** — działa w środowisku, zbiera doświadczenia
- **Learner (Python)** — uczy się z doświadczeń, dostarcza politykę (model)

Korzyści:
- Skalowalność — można dodać więcej Actorów
- Równoległość — Actors zbierają doświadczenia równocześnie
- Separacja — każdy komponent ma jasną odpowiedzialność

## Uruchomienie i zamykanie

System uruchamiany i zatrzymywany jednym poleceniem:

```bat
run.bat   :: Uruchomienie wszystkich procesów
stop.bat  :: Graceful shutdown w odpowiedniej kolejności
```

Kolejność startu: Monitoring Service → Learner → Actorzy → Dashboard.
Kolejność zamykania: Actorzy → Learner (zapisuje checkpoint) → Monitoring → Dashboard.

Pełna dokumentacja: [Graceful Shutdown](shutdown.md).

## Warstwy komunikacji

System używa trzech oddzielnych kanałów komunikacji, dopasowanych do charakteru danych:

### 1. Actor ↔ Learner — ZeroMQ REQ/REP

**Szybka, dwukierunkowa komunikacja** — dużo requestów na sekundę, niskie opóźnienia kluczowe.

```
Actor ──REQ (state, action, reward, done)──► Learner
Actor ◄──REP (nextAction)────────────────── Learner
```

- Duża częstotliwość (każdy krok tradingowy)
- Batchowanie requestów dla efektywności
- Automatyczny reconnect przy padzie

### 2. Monitoring — ZeroMQ PUSH/PULL

**Jednostronne wysyłanie metryk** — każdy moduł push-uje dane do Monitoring Service.

```
Actor         ──PUSH──► Monitoring Svc (PULL)
Learner       ──PUSH──► Monitoring Svc (PULL)
```

- Bez zwrotnej odpowiedzi
- Nie blokuje nadawcy
- Asynchroniczne, nie wpływa na performance

### 3. Dashboard ↔ Monitoring — HTTP REST

**Przeglądarka odpytuje Monitoring Service** — fetch jest naturalny dla webu.

```
Dashboard (fetch/HTTP GET) ──► Monitoring Service (REST API)
Dashboard ◄── JSON response ── Monitoring Service
```

- Dashboard działa w przeglądarce — HTTP jest natywny
- Polling co 60 sekund (niska częstotliwość)
- Tylko GET — brak modyfikacji danych
- CORS-enabled dla dev mode

### Podsumowanie

| Kanał | Protokół | Pattern | Częstotliwość |
|---|---|---|---|
| Actor ↔ Learner | ZeroMQ | REQ/REP | Wysoka (~10-100/s) |
| Monitoring | ZeroMQ | PUSH/PULL | Średnia (~1/5s) |
| Dashboard | HTTP REST | GET | Niska (~1/60s) |

Pełna dokumentacja komunikacji: [Komunikacja ZeroMQ](communication.md)

# Komunikacja ZeroMQ

## Przegląd

Moduły komunikują się przez **ZeroMQ** (REQ/REP + PUSH/PULL pattern), co zapewnia niskie opóźnienia i lepszą skalowalność niż REST/HTTP.

## Topologia komunikacji

```
Actor ←→ REQ/REP (step + action + experience) ←→ Learner
Learner ──► PUSH (metrics) ──► Monitoring Service (PULL)
Actor  ──► PUSH (metrics) ──► Monitoring Service (PULL)
Monitoring Service ◄── HTTP REST (GET) ◄── Dashboard
```

> **Doświadczenia (experiences)** są wysyłane przez **ten sam kanał REQ/REP** co request o akcję. Step request zawiera `(state, action, reward, nextState, done)` a w odpowiedzi oprócz `nextAction` Learner zapisuje doświadczenie do replay buffer.

## Patterny ZeroMQ

### REQ/REP (Request/Reply)

**Synchroniczna** komunikacja dwukierunkowa.

```
Client (REQ) ──► request ──► Server (REP)
Client (REQ) ◄── response ◄─ Server (REP)
```

**Zastosowanie:**
- Step — Actor wysyła `(state, action, reward, nextState, done)` i dostaje `nextAction`
- Step request **zawiera doświadczenie** — Learner zapisuje je do replay buffer
- Predict — Actor pyta o Q-values
- Status — HTTP endpoint Monitoring Service

### PUSH/PULL (Push/Pull)

**Asynchroniczna** komunikacja jednokierunkowa.

```
Sender (PUSH) ──► message ──► Receiver (PULL)
```

**Zastosowanie:**
- Metrics — Actor i Learner wysyłają metryki do Monitoring Service

## Typy wiadomości

### Batch Predict (REQ/REP) — główny kanał

**Kierunek:** ActorManager → Learner → ActorManager

ActorManager zbiera requesty od wszystkich aktorów i wysyła jeden **batch** do Learnera:

**Request (lista, jeden element per aktor):**
```json
[
    {
        "actorId": "BTCUSDT",
        "symbol": "BTCUSDT",
        "state": {
            "candles_1m":  [[...11 cech...], ...60 świec...],
            "candles_15m": [[...], ...32 świece...],
            "candles_1h":  [[...], ...48 świec...],
            "candles_1d":  [[...], ...14 świec...]
        },
        "positionFeatures": [0, 1, -0.02, 0.12, 0.01, 0.5, 0.87, ...],
        "actionMask": [0, 0, 1, 1],
        "metrics": {
            "transactions": 5,
            "episode_pnl": 0.023,
            "win": true
        }
    },
    {
        "actorId": "ETHUSDT",
        ...
    }
]
```

**Response (lista, jeden element per aktor):**
```json
[
    {
        "action": 3,
        "qValues": [−0.12, −0.08, 0.05, 0.21],
        "epsilon": 0.72
    },
    {...}
]
```

### Wysyłanie doświadczeń — osobny batch

Na końcu epizodu ActorManager wysyła **osobny batch doświadczeń** (nie w predict request):

```json
[
    {
        "actorId": "BTCUSDT",
        "experiences": [
            {
                "state": {...},
                "action": 0,
                "reward": −0.00075,
                "nextState": {...},
                "done": false,
                "actionMask": [0, 0, 1, 1],
                "returnG": −0.023,
                "gammaToN": 0.489
            },
            ...
        ]
    }
]
```

### Status (Monitoring Service HTTP GET)

**Kierunek:** Dashboard → Monitoring Service → Dashboard

**Response:**
```json
{
    "bufferSize": 125000,
    "trainSteps": 50000,
    "epsilon": 0.65,
    "lastLoss": 0.032
}
```

## Serializacja: MessagePack

### Opis

Stany i akcje są serializowane przez **MessagePack** — binarny format kilkukrotnie zmniejszający payload względem JSON.

### Porównanie

| Format | Rozmiar (batch 3 aktorów) |
|---|---|
| JSON | ~15 KB |
| MessagePack | ~4 KB |

### Przykład użycia

```python
import msgpack
import zmq

# Python (Learner)
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

# Odbierz i zdeserializuj
message = socket.recv()
data = msgpack.unpackb(message, raw=False)

# Zserializuj i wyślij
response = msgpack.packb({"action": 2, "qValues": [0.1, 0.8, -0.2]})
socket.send(response)
```

```javascript
const zeromq = require('zeromq');
const msgpack = require('msgpack5');

// Node.js (Actor)
const socket = new zeromq.Request();
socket.connect('tcp://127.0.0.1:5555');

// Wyślij
const data = msgpack().encode({ state: stateData, action: 0, reward: 0.05 });
socket.send(data);

// Odbierz
const reply = await socket.recv();
const response = msgpack().decode(reply);
```

## Dlaczego ZeroMQ

### Porównanie z HTTP/REST

| Cecha | HTTP/REST | ZeroMQ |
|---|---|---|
| Opóźnienie | ~1-5ms | ~0.1ms |
| Reconnect | Manualny | Automatyczny |
| Pattern | Req/Res tylko | Req/Res, Push/Pull, Pub/Sub |
| Throughput | Ograniczony | Liniowa skalowalność |
| Overhead | Duży (headers) | Minimalny (binary) |

### Korzyści

- **Niskie opóźnienia** — ~0.1ms vs ~1-5ms dla HTTP
- **Automatyczny reconnect** — przy padzie któregokolwiek modułu
- **Wiele patternów** — REQ/REP, PUSH/PULL, PUB/SUB
- **Binarny transport** — mniejszy overhead niż JSON over HTTP
- **Asynchroniczność** — PUSH/PULL nie blokuje nadawcy

## Endpoints

### Konfiguracja

```toml
[learner]
host = "tcp://127.0.0.1"
port = 5555

[monitoring]
host = "tcp://127.0.0.1"
port = 3001
```

### Domyślne porty

| Moduł | Port | Pattern |
|---|---|---|
| Learner (step/predict/experience) | 5555 | REQ/REP |
| Learner (metrics) | 5556 | PUSH |
| Monitoring Service (HTTP REST) | 3001 | HTTP GET |
| Monitoring Service (incoming metrics) | 3002 | PULL (metrics) |

## Obsługa błędów

### Reconnect

ZeroMQ automatycznie próbuje reconnect przy zerwaniu połączenia.

```python
# Python (Learner)
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

# Jeśli Actor padnie i się połączy ponownie,
# ZeroMQ automatycznie obsłuży reconnect
while True:
    message = socket.recv()  # blokuje aż do połączenia
    response = process(message)
    socket.send(response)
```

### Timeout

```python
# Ustaw timeout
socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 sekund
socket.setsockopt(zmq.SNDTIMEO, 5000)

try:
    message = socket.recv()
except zmq.Again:
    print("Timeout - no message received")
```

### Heartbeat (opcjonalny)

```python
# Actor wysyła heartbeat co N sekund
heartbeat_socket = context.socket(zmq.PUSH)
heartbeat_socket.connect("tcp://learner:5558")

import threading

def send_heartbeat():
    while True:
        heartbeat_socket.send(b"alive")
        time.sleep(5)

threading.Thread(target=send_heartbeat, daemon=True).start()
```

## Batchowanie requestów

### Opis

`actorManager.js` zbiera requesty od wszystkich Actorów przez kilka milisekund i wysyła jeden zbiorczy batch do Pythona.

### Przykład batcha

```javascript
// Zbierz requesty
let batch = [];
let resolveBatch = null;

function addToBatch(request) {
    return new Promise((resolve) => {
        batch.push(request);
        if (!resolveBatch) {
            resolveBatch = resolve;
            setTimeout(flushBatch, 10);  // 10ms window
        }
    });
}

async function flushBatch() {
    const currentBatch = batch;
    batch = [];
    resolveBatch = null;
    
    if (currentBatch.length > 0) {
        const response = await sendToPython(currentBatch);
        currentBatch.forEach((req, i) => {
            req.resolve(response[i]);
        });
    }
}
```

### Format batcha

```json
[
    {"actorId": "BTCUSDT", "state": {...}, "action": 0, "reward": 0.05, "nextState": {...}, "done": false},
    {"actorId": "ETHUSDT", "state": {...}, "action": 2, "reward": 0.00, "nextState": {...}, "done": false},
    {"actorId": "SOLUSDT", "state": {...}, "action": 1, "reward": -0.02, "nextState": {...}, "done": false}
]
```

### Odpowiedź batchowa

```json
[
    {"actorId": "BTCUSDT", "action": 2, "qValues": [0.1, -0.3, 0.8]},
    {"actorId": "ETHUSDT", "action": 0, "qValues": [0.6, -0.1, 0.2]},
    {"actorId": "SOLUSDT", "action": 2, "qValues": [-0.2, 0.4, 0.5]}
]
```

## Config

```toml
[learner]
host = "tcp://127.0.0.1"
port = 5555

[monitoring]
host = "tcp://127.0.0.1"
port = 3001
metrics_push_interval_sec = 5
dashboard_poll_interval_sec = 60
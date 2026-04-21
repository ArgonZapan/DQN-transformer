# Aktorzy

## Przegląd

Aktorzy to instancje **środowiska tradingowego** — każdy Actor symuluje handel dla jednej pary kryptowalutowej. Wszyscy aktorzy wysyłają doświadczenia do jednego Python Learnera.

## Aktualna konfiguracja (5 par)

```toml
[[actors]]
symbol = "BTCUSDT"
exchange = "binance"
leverage = 1

[[actors]]
symbol = "ETHUSDT"
exchange = "binance"
leverage = 1

[[actors]]
symbol = "SOLUSDT"
exchange = "binance"
leverage = 1

[[actors]]
symbol = "BNBUSDT"
exchange = "binance"
leverage = 1

[[actors]]
symbol = "XRPUSDT"
exchange = "binance"
leverage = 1
```

Każdy `[[actors]]` uruchamia osobny wątek Node.js. Dodanie nowej pary = jeden wpis w TOML, zero zmian w kodzie.

## Dlaczego wielu aktorów?

| Korzyść | Opis |
|---|---|
| **Generalizacja** | Sieć uczy się wzorców niezależnych od konkretnej pary |
| **Różnorodność** | Bufor zawiera mix: trend / bessa / konsolidacja |
| **Dekorrelacja** | Naturalne rozwiązanie korelacji doświadczeń w DQN |
| **Skalowalność** | Dodaj parę przez TOML |

## Inferencja — ONNX local vs RPC

Actor wybiera akcję w dwuetapowej hierarchii:

### 1. Lokalna inferencja ONNX (preferowana)

```javascript
const onnxResult = await this._inferOnnx(state, actionMask);
if (onnxResult) {
    action  = onnxResult.action;
    qValues = onnxResult.qValues;
}
```

Gdy dostępny plik `model.onnx` — aktor ładuje go lokalnie przez `onnxruntime-node` i wykonuje inferencję **bez latencji TCP**. Plik jest monitorowany co 5 sekund — gdy Learner zapisze nową wersję, Actor ją automatycznie ładuje.

### 2. Fallback: RPC do Python przez ZMQ

```javascript
const response = await this.pythonClient.predict(state, actionMask);
action  = response.action;
qValues = response.qValues;
if (response.epsilon != null) this.epsilon = response.epsilon;
```

Używany gdy brak ONNX lub błąd onnxruntime. Learner zwraca też bieżące epsilon.

### 3. Fallback po błędzie RPC

```javascript
// Silne faworyzowanie HOLD po błędzie sieci — bezpieczna domyślna akcja
action = this._sampleWeightedAction(actionMask, [2, 2, 94, 2]);
```

## Eksploracja Epsilon-Greedy

```javascript
if (Math.random() < this.epsilon) {
    // Losowa akcja z wagami (nie uniform!)
    action = this._sampleWeightedAction(actionMask, [13, 13, 60, 13]);
} else {
    // Model (ONNX lub RPC)
    action = model.predict(state);
}
```

Ważona losowość: `HOLD=60%`, `LONG=13%`, `SHORT=13%`, `CLOSE=13%`. Zapobiega dominacji OPEN akcji w early exploration (odpowiada za `hold_bias=0.5` w config).

## Cykl życia aktora

### 1. Inicjalizacja

- Załaduj konfigurację (symbol, timeframes, reward)
- Załaduj dane historyczne ze wszystkich TF
- Obetnij dane do okna: warmup + training_months + validation_days

### 2. Pętla epizodów

```javascript
while (this.running) {
    await this.runEpisode();
    this.totalEpisodes++;
}
```

### 3. Krok epizodu

```javascript
// Epsilon-greedy: ONNX → RPC → fallback
action = await this.selectAction(state, actionMask);

// Krok w środowisku
const result = this.env.step(action);
done = result.done;
state = result.nextState;
actionMask = result.actionMask;

// Throttle jeśli Learner sygnalizuje pełny bufor
if (this.throttleMs > 0) await sleep(this.throttleMs);
```

### 4. Koniec epizodu — wysłanie doświadczeń

```javascript
// Oblicz zwroty zgodnie z return_mode
const experiences = getExperiences(returnMode, nStep);

// Episode mirror — zdubluj odwrócony LONG↔SHORT
if (episodeMirror) mirrorExperiences = buildMirror(experiences);

// Wyślij do Python Learnera przez PythonClient (ZMQ)
await pythonClient.sendBatch(experiences);
```

## Okno danych treningowych

```toml
[training]
training_months  = 48   # ostatnie 48 miesięcy danych
validation_days  = 30   # ostatnie 30 dni = OOS (nigdy nie trenujesz na nich)
```

```
|← warmup (14 dni) →|← trening (48 mies.) →|← OOS (30 dni) →|
       bufor sieci          losowe starty         nigdy tutaj
```

## Dostępne akcje

| Akcja | ID | Opis |
|---|---|---|
| LONG | 0 | Otwórz pozycję LONG |
| SHORT | 1 | Otwórz pozycję SHORT |
| HOLD | 2 | Trzymaj / czekaj |
| CLOSE | 3 | Zamknij otwartą pozycję |

Zawsze **jedna pozycja jednocześnie**. Flip LONG→SHORT wymaga CLOSE + LONG.

## Logika pozycji (action masking)

| Stan | LONG | SHORT | HOLD | CLOSE |
|---|---|---|---|---|
| Brak pozycji | ✓ | ✓ | ✓ | ✗ |
| Otwarta LONG | ✗ | ✗ | ✓ | ✓ |
| Otwarta SHORT | ✗ | ✗ | ✓ | ✓ |
| < min_hold_steps | ✗ | ✗ | ✓ | ✗ |

`min_hold_steps = 4` — CLOSE jest blokowane przez pierwsze 4 kroki od otwarcia.

## Stan środowiska

```javascript
state = {
    candles_1m:  Float32Array([60, 11]),
    candles_15m: Float32Array([32, 11]),
    candles_1h:  Float32Array([48, 11]),
    candles_1d:  Float32Array([14, 11]),
    // candles_1w = 0 → pominięty
}

positionFeatures = Float32Array([10])  // is_long, is_short, uPnL, hold_6h, ...
```

## Actor Manager

`actorManager.js` zarządza wszystkimi aktorami:

```
Actor BTC ──┐
Actor ETH ──┼──► actorManager ──► batch ──► Python Learner
Actor SOL ──┤
Actor BNB ──┤
Actor XRP ──┘
               ◄─────────── Q-values + epsilon ──────────────
```

Batchuje requesty od wszystkich aktorów w jednym oknie czasowym, wysyła jeden zbiorczy request do Pythona.

## Normalizacja per para

Każda para ma inną skalę cen — BTC ~60k, XRP ~0.5. Normalizacja jest liczona przez **rolling window per para**:

```toml
[data]
normalization_window = 60   # okno rolling mean/std
```

Każdy Actor utrzymuje własne `PerPairNormalizer` — sieć widzi te same wzorce niezależnie od bezwzględnej ceny.

## Episode Mirror

```toml
[training]
episode_mirror = false   # true = generuj lustrzany epizod po każdym
```

Gdy `true`, po każdym epizodzie Actor generuje "lustrzany":
- Wszystkie LONG → SHORT i odwrotnie
- P&L odwrócony: zysk LONG = strata odpowiadającego SHORT
- Balansuje bufor 50/50 między kierunkami pozycji

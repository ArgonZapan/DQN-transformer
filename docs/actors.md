# Aktorzy

## Przegląd

Aktorzy to instancje **środowiska tradingowego** — każdy Actor symuluje handel dla jednej pary kryptowalutowej. Wszyscy aktorzy wysyłają doświadczenia do jednego Python Learnera.

## Konfiguracja aktorów

Aktorzy są zdefiniowani jako **lista obiektów** w pliku TOML. Każdy wpis zawiera symbol, exchange i indywidualne parametry.

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
```

### Dodawanie nowej pary

Dodanie nowej pary to **jeden wpis w konfigu**, zero zmian w kodzie:

```toml
[[actors]]
symbol = "ADAUSDT"
exchange = "binance"
leverage = 1
```

## Globalne timeframe'y

Timeframe'y są zdefiniowane **globalnie** — jeden zestaw dla wszystkich aktorów, jeden model.

```toml
[timeframes]
candles_1m  = 15
candles_15m = 15
candles_1h  = 20
candles_1d  = 30
candles_1w  = 54
```

Wszyscy Actorzy używają:
- Tej samej struktury stanu
- Tego samego schematu nagród
- Tych samych timeframe'ów

## Dlaczego wielu aktorów?

### Korzyści

| Korzyść | Opis |
|---|---|
| **Generalizacja** | Sieć uczy się ogólnych wzorców niezależnych od pary |
| **Różnorodność** | Replay buffer zawiera mix z różnych reżimów rynkowych |
| **Dekorrelacja** | Naturalne rozwiązanie problemu korelacji doświadczeń |
| **Skalowalność** | Dodaj kolejnego aktora przez wpis w TOML |

### Przykład

```
BTC Actor ──► doświadczenia z trendu wzrostowego ──┐
ETH Actor ──► doświadczenia z bessy ───────────────┼──► Python Learner
SOL Actor ──► doświadczenia z konsolidacji ────────┘
```

Model uczy się że formacja X jest bullish niezależnie od tego czy występuje na BTC czy ETH.

## Actor Manager

`actorManager.js` zarządza wszystkimi aktorami:
- Uruchamia instancje według konfiguracji
- Zbiera requesty od aktorów
- Batchuje requesty do Pythona
- Rozdziela odpowiedzi do właściwych aktorów

### Schemat działania

```
Actor 1 ──┐
Actor 2 ──┼──► actorManager ──► batch ──► Python Learner
Actor 3 ──┘
               ◄───────────────────────── Q-values
```

## Batchowanie requestów

### Opis

`actorManager.js` zbiera requesty od wszystkich Actorów przez **kilka milisekund** i wysyła jeden zbiorczy batch do Pythona.

### Korzyści

- **Jedno wywołanie modelu** obsługuje wielu aktorów
- **Lepszy utilisation GPU** — batche są bardziej efektywne
- **Mniejszy narzut komunikacji** — mniej requestów ZMQ

### Implementacja

```javascript
// Zbieraj requesty przez 10ms
const BATCH_WINDOW_MS = 10;

let pendingRequests = [];

setTimeout(() => {
    if (pendingRequests.length > 0) {
        // Wyślej batch
        const batch = pendingRequests.splice(0);
        sendToPython(batch);
    }
}, BATCH_WINDOW_MS);
```

## Epsilon-greedy faza

### Opis

Przy wysokim `epsilon` Actor wybiera **losową akcję** bez pytania modelu — znacząco przyspiesza zapełnienie bufora w początkowej fazie.

### Parametry

```
epsilon start: 1.0    (pełna eksploracja)
epsilon end:   0.05   (głównie eksploatacja)
decay: liniowy przez pierwsze 30% kroków
```

### Decyzja o akcji

```python
if random() < epsilon:
    action = random_action()  # bez pytania modelu
else:
    action = model.predict(state)  # pytanie Pythona
```

### Optymalizacja

Gdy `epsilon > 0.8`, Actorzy **nie pytają Pythona** o akcję — generują losowe działania lokalnie. Python dostaje tylko doświadczenia do bufora.

## Cykl życia aktora

### 1. Inicjalizacja
- Załaduj konfigurację (symbol, exchange, leverage)
- Pobierz dane historyczne
- Zainicjalizuj środowisko tradingowe

### 2. Start epizodu
- Losowy punkt startowy w danych historycznych
- Reset stanu pozycji (brak otwartej pozycji)
- Buduj pierwszy stan

### 3. Krok
- Pobierz aktualny stan
- Wybierz akcję (epsilon-greedy lub model)
- Wykonaj akcję w środowisku
- Oblicz nagrodę
- Wyślij doświadczenie do Learnera

### 4. Koniec epizodu
- Zamknij otwartą pozycję (jeśli istnieje)
- Oblicz Monte Carlo Returns
- Wyślij doświadczenia z return_G
- Reset środowiska

## Stan środowiska

Actor buduje stan z 5 timeframe'ów:

```javascript
state = {
    '1m':  candles_1m,   // [15, 8]
    '15m': candles_15m,  // [15, 8]
    '1h':  candles_1h,   // [20, 8]
    '1d':  candles_1d,   // [30, 8]
    '1w':  candles_1w,   // [54, 8]
}
```

## Dostępne akcje

System używa **4 akcji** — HOLD służy do trzymania pozycji lub czekania, a CLOSE wyłącznie do jej zamknięcia.

| Akcja | ID | Opis |
|---|---|---|
| LONG | 0 | Otwórz pozycję LONG |
| SHORT | 1 | Otwórz pozycję SHORT |
| HOLD | 2 | Trzymaj pozycję / Czekaj (bez pozycji) |
| CLOSE | 3 | Zamknij otwartą pozycję |

> **Uwaga:** Może istnieć tylko **jedna pozycja jednocześnie** — albo LONG, albo SHORT. Nie można mieć obu na raz.

## Logika pozycji

### Pozycja LONG otwarta
- `LONG` → ignoruj (pozycja już otwarta)
- `SHORT` → ignoruj (nie można flip pozycji)
- `HOLD` → trzymaj pozycję
- `CLOSE` → **zamknij pozycję**

### Pozycja SHORT otwarta
- `LONG` → ignoruj (nie można flip pozycji)
- `SHORT` → ignoruj (pozycja już otwarta)
- `HOLD` → trzymaj pozycję
- `CLOSE` → **zamknij pozycję**

### Brak pozycji
- `LONG` → otwórz pozycję LONG
- `SHORT` → otwórz pozycję SHORT
- `HOLD` → czekaj (nic nie rób)
- `CLOSE` → ignoruj (brak pozycji do zamknięcia)

## Config

```toml
[[actors]]
symbol = "BTCUSDT"
exchange = "binance"
leverage = 1

[training]
epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay_fraction = 0.3
## Normalizacja per para

Każda para ma inną skalę cen i wolumenu — BTC kosztuje kilkadziesiąt tysięcy, SOL kilkaset. Normalizacja jest liczona **osobno dla każdej pary**.

### Zasada

Każdy Actor utrzymuje własne statystyki rolling (mean, std) dla swojej pary. Dzięki temu sieć widzi te same wzorce niezależnie od bezwzględnej ceny.

### Konfiguracja

```toml
[data]
normalization_window = 20   # Okno rolling dla mean/std
```

### Implementacja

```javascript
class PerPairNormalizer {
    constructor(window = 20) {
        this.window = window;
        this.history = [];  // ostatnie N wartości close
    }

    update(close) {
        this.history.push(close);
        if (this.history.length > this.window) {
            this.history.shift();
        }
    }

    normalize(value) {
        if (this.history.length < 2) return 0;  // cold start — zwróć 0

        const mean = this.history.reduce((a, b) => a + b) / this.history.length;
        const std = Math.sqrt(
            this.history.map(x => (x - mean) ** 2).reduce((a, b) => a + b) / this.history.length
        );

        if (std === 0) return 0;
        return (value - mean) / std;
    }
}
```

### Cold start

Przy zimnym starcie (pierwsze N świec) historia jest niekompletna. Actor wypełnia brakujące wartości zerem — sieć traktuje to jako brak sygnału.

### Zapis stanu normalizatora

Stan normalizatora (historia rolling) jest zapisywany razem z checkpointem żeby uniknąć ponownego cold startu po restarcie.

## Obsługa końca danych historycznych

Podczas treningu Actor dojdzie do końca danych historycznych. Musi wtedy zresetować epizod bez nakładania się na dane out-of-sample.

### Zasada podziału danych

```
|←────── 80% dane treningowe ──────→|←── 20% OOS ──→|
         Actor trenuje tutaj              NIGDY tutaj
```

### Zachowanie przy końcu danych

Gdy Actor dojdzie do końca danych treningowych (80%):

1. Zamknij otwartą pozycję (jeśli istnieje) — oblicz nagrodę
2. Oblicz Monte Carlo Returns dla epizodu
3. Wyślij doświadczenia do Learnera
4. **Reset do losowego punktu startowego** w przedziale treningowym (0-80%)

### Losowy punkt startowy

```javascript
function getRandomStartIndex(data, trainFraction = 0.8) {
    const trainEnd = Math.floor(data.length * trainFraction);
    // Zostaw miejsce na minimalną długość epizodu (np. 100 kroków)
    const maxStart = trainEnd - config.training.min_episode_length;
    return Math.floor(Math.random() * maxStart);
}
```

### Konfiguracja

```toml
[training]
train_data_fraction = 0.8    # 80% danych na trening, 20% OOS
min_episode_length = 100     # Minimalna długość epizodu w krokach
```

### Obsługa brakującej historii

Gdy para nie ma wystarczającej historii dla danego timeframe'a (np. nowa para bez roku świec tygodniowych):

```toml
[data]
allow_partial_history = true   # false = odrzuć parę jeśli brakuje historii
```

Gdy `allow_partial_history = true`, brakujące świece są wypełniane zerami od lewej (zero-padding). Actor loguje ostrzeżenie przy starcie:

```
[WARNING] [actor:SOLUSDT] Niekompletna historia dla 1w: 30/54 świec. Zero-padding zastosowany.
```

Gdy `allow_partial_history = false`, Actor odrzuca parę i loguje błąd — system nie startuje dla tej pary.

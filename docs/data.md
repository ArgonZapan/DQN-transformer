# Dane wejściowe

## Przegląd

Model używa **multi-scale temporal representation** — danych z wielu timeframe'ów pokrywających tę samą historię z różną rozdzielczością. Im bliżej teraźniejszości, tym więcej szczegółów.

## Timeframe'y (aktualna konfiguracja)

```toml
[timeframes]
candles_1m  = 60   # świece 1-minutowe
candles_15m = 32   # świece 15-minutowe
candles_1h  = 48   # świece 1-godzinne
candles_1d  = 14   # świece dzienne
candles_1w  = 0    # wyłączony (0 = pominięty)
```

### Pokrycie czasowe

| Timeframe | Świece | Pokryty okres |
|---|---|---|
| 1m  | 60 | 1 godzina |
| 15m | 32 | 8 godzin |
| 1h  | 48 | 2 dni |
| 1d  | 14 | 2 tygodnie |
| 1w  | 0  | wyłączony |

Łączna długość sekwencji wejściowej modelu: **154 tokeny** (świece).

### Wyłączanie timeframe'ów

Timeframe z `candles_X = 0` jest **pomijany** zarówno przez Node.js actora jak i przez model Python. Liczba Conv1D bloków dostosowuje się automatycznie.

### Konfiguracja alternatywna (z 1w)

```toml
[timeframes]
candles_1m  = 15
candles_15m = 15
candles_1h  = 20
candles_1d  = 30
candles_1w  = 54
```

## 11 cech na świecę (v1+v2)

Aktualna implementacja łączy cechy z obu wersji feature engineering w jednym zestawie.

### Lista cech

```javascript
// node/env/state.js — buildFeatures()
features[i] = [
    (close - mean) / std,                          // 0. normalizedClose
    (high - low) / close,                          // 1. relativeRange
    (close - open) / close,                        // 2. candleDirection
    Math.min(volume / meanVolume, 3.0),            // 3. volumeClipped (cap 3×)
    rsi / 100,                                      // 4. rsiNorm
    (close - min14) / (max14 - min14),             // 5. stochasticK
    macdLine / close,                               // 6. macdNorm
    (macdLine - signalLine) / close,               // 7. macdHistNorm
    (close - prevClose) / prevClose,               // 8. pctChange
    (4 * std20) / sma20,                           // 9. bollingerWidth
    (close - sma20) / close,                       // 10. smaDistance
]
```

### Opis cech

| # | Nazwa | Formuła | Zakres | Opis |
|---|---|---|---|---|
| 0 | normalizedClose | `(close - mean) / std` | ~[-3, 3] | Z-score ceny |
| 1 | relativeRange | `(H - L) / close` | [0, ∞) | Volatilność świecy |
| 2 | candleDirection | `(close - open) / close` | [-1, 1] | Kierunek świecy |
| 3 | volumeClipped | `min(vol / meanVol, 3.0)` | [0, 3] | Wolumen z capem |
| 4 | rsiNorm | `RSI(14) / 100` | [0, 1] | RSI znormalizowany |
| 5 | stochasticK | `(close - min14) / (max14 - min14)` | [0, 1] | Stochastic K |
| 6 | macdNorm | `macdLine / close` | ~[-0.01, 0.01] | MACD znorm. przez cenę |
| 7 | macdHistNorm | `(macdLine - signal) / close` | ~[-0.005, 0.005] | Histogram MACD |
| 8 | pctChange | `(close - prevClose) / prevClose` | [-1, 1] | Zmiana % |
| 9 | bollingerWidth | `4 * std20 / sma20` | [0, ∞) | Szerokość Bollingera |
| 10 | smaDistance | `(close - sma20) / close` | ~[-0.1, 0.1] | Odległość od SMA20 |

### Dlaczego 11 cech?

- **Cena znormalizowana** — sieć widzi te same wzorce przy BTC=1000 i BTC=100000
- **Relative range** — informacja o volatilności świecy
- **Kierunek** — bull/bear candle
- **Wolumen clipped** — unika dominacji outlier'ów (skoki 100×)
- **RSI** — klasyczny momentum (wykupienie/wyprzedanie)
- **Stochastic K** — alternatywny oscylator momentum (14-period lookback)
- **MACD znorm** — trend, crossover sygnałowy
- **MACD histogram** — siła i kierunek trendu MACD
- **Zmiana %** — krótkoterminowe momentum (1-bar return)
- **Bollinger Width** — volatilność względem SMA20
- **SMA Distance** — pozycja ceny względem SMA20 (ciągła, nie binarna)

## Wskaźniki techniczne

### RSI (14-period)

```javascript
// Relative Strength Index
delta = close[i] - close[i-1]
gain = rolling_mean(max(delta, 0), 14)
loss = rolling_mean(max(-delta, 0), 14)
rs = gain / loss
rsi = 100 - (100 / (1 + rs))
```

### Stochastic K (14-period)

```javascript
// Oscylator stochastyczny — pozycja close względem zakresu 14-period
min14 = min(low[i-13..i])
max14 = max(high[i-13..i])
stochK = (close - min14) / (max14 - min14)
```

### MACD

```javascript
ema12 = EMA(close, 12)
ema26 = EMA(close, 26)
macdLine = ema12 - ema26
signal = EMA(macdLine, 9)

// Normalizacja przez cenę (nie surowe wartości)
macdNorm = macdLine / close
macdHistNorm = (macdLine - signal) / close
```

### Bollinger Width

```javascript
sma20 = SMA(close, 20)
std20 = rolling_std(close, 20)
bollingerWidth = 4 * std20 / sma20   // = (upper - lower) / middle
```

Wąskie pasma → konsolidacja. Szerokie → wysoka volatilność.

## Normalizacja

### Zasada

**Nigdy surowe ceny** — wszystkie cechy są znormalizowane relative lub przez Z-score. Sieć musi widzieć te same wzorce bez względu na to czy BTC kosztuje 1000 czy 100000.

### Rolling window

```toml
[data]
normalization_window = 60   # okno rolling mean/std dla Z-score
```

```javascript
mean = rollingMean(closes, normWindow)   // średnia krocząca 60 świec
std  = rollingStd(closes, normWindow)    // odch. std krocząca 60 świec
normalizedClose = (close - mean) / std
```

## Synchronizacja czasowa timeframe'ów

Każdy krok Actor buduje stan używając świec które są **zamknięte** w momencie bieżącej świecy 1m. Zapobiega look-ahead bias.

### Algorytm (binary search O(log N))

```javascript
function getAlignedCandles(allCandles, currentTime, numCandles) {
    // Binary search: znajdź pierwszą świecę z close_time > currentTime
    let lo = 0, hi = allCandles.length;
    while (lo < hi) {
        const mid = (lo + hi) >>> 1;
        if (allCandles[mid].close_time <= currentTime) lo = mid + 1;
        else hi = mid;
    }
    const start = Math.max(0, lo - numCandles);
    return allCandles.slice(start, lo);
}
```

### Przykład

```
Bieżąca świeca 1m: close_time = 2024-01-15 14:30:00

Świece 15m: ostatnia zamknięta to 14:15-14:30 ✓
Świece 1h:  ostatnia zamknięta to 13:00-14:00 ✓ (14:00-15:00 jest otwarta — pomijamy)
Świece 1d:  ostatnia zamknięta to 2024-01-14 ✓
```

## Struktura stanu

```javascript
state = {
    candles_1m:  Float32Array([60, 11]),  // 60 świec × 11 cech
    candles_15m: Float32Array([32, 11]),
    candles_1h:  Float32Array([48, 11]),
    candles_1d:  Float32Array([14, 11]),
    // candles_1w pomijane (0 w config)
}
```

Łączny rozmiar: `(60 + 32 + 48 + 14) × 11 = 1694 float32`.

## Źródła danych

### Tryb file (domyślny)

```toml
[data]
source = "file"
path = "node/data/historical/"
```

Actor ładuje pliki CSV z danymi historycznymi. Pliki pobiera się raz przez `scripts/download_data.js`.

### Tryb API (real-time)

```toml
[data]
source = "api"
binance_rate_limit = 1000   # ms między żądaniami
api_retry_interval_sec = 60
```

Dane pobierane na bieżąco z Binance REST API. Centralny `BinanceClient` kolejkuje requesty i pilnuje rate-limitów.

## Pobieranie danych historycznych

```bash
cd scripts
node download_data.js --symbol BTCUSDT --interval 1m --days 30
node download_data.js --symbol BTCUSDT --interval 15m --days 90
node download_data.js --symbol BTCUSDT --interval 1h --days 365
node download_data.js --symbol BTCUSDT --interval 1d --days 1095
```

### Format plików

```
node/data/historical/
├── BTCUSDT_1m.csv
├── BTCUSDT_15m.csv
├── BTCUSDT_1h.csv
├── BTCUSDT_1d.csv
├── ETHUSDT_1h.csv
└── ...
```

Format CSV: `timestamp,open,high,low,close,volume,close_time,...`

## Obsługa brakującej historii

```toml
[data]
allow_partial_history = false  # true = zero-padding dla brakujących świec
```

Gdy `false` — Actor nie startuje jeśli brakuje danych dla jakiegoś timeframe'a.
Gdy `true` — brakujące świece wypełniane zerami od lewej (cold start).

## Config

```toml
[features]
num_features = 11

[timeframes]
candles_1m  = 60
candles_15m = 32
candles_1h  = 48
candles_1d  = 14
candles_1w  = 0

[data]
source = "file"
path = "node/data/historical/"
cache_path = "node/data/cache/"
normalization_window = 60
allow_partial_history = false
binance_rate_limit = 1000
api_retry_interval_sec = 60
```

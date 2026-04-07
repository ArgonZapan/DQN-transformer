# Dane wejściowe

## Przegląd

Model używa **multi-scale temporal representation** — danych z wielu timeframe'ów pokrywających tę samą historię z różną rozdzielczością. Im bliżej teraźniejszości, tym więcej szczegółów.

## Timeframe'y

### Konfiguracja

Wszystkie timeframe'y są zdefiniowane globalnie — jeden zestaw dla wszystkich aktorów, jeden model.

```toml
[timeframes]
candles_1m  = 15
candles_15m = 15
candles_1h  = 20
candles_1d  = 30
candles_1w  = 54
```

### Pokrycie czasowe

| Timeframe | Świece | Pokryty okres |
|---|---|---|
| 1m  | 15 | 15 minut |
| 15m | 15 | 3.75 godziny |
| 1h  | 20 | 20 godzin |
| 1d  | 30 | 30 dni |
| 1w  | 54 | ~1 rok |

### Dlaczego taka konfiguracja?

- **Żadna świeca nie pokrywa innej** — czysta logarytmiczna skala rozdzielczości
- **1m** — mikro trendy, szybkie reakcje
- **15m** — krótkoterminowe momentum
- **1h** — średnioterminowe trendy
- **1d** — dzienne trendy, sentiment rynkowy
- **1w** — długoterminowy kontekst

## 8 cech na świecę

### Lista cech

```python
features = [
    (close - mean) / std,           # 1. znormalizowana cena close
    (high - low) / close,           # 2. relative range świecy
    (close - open) / close,         # 3. kierunek świecy
    volume / mean_volume,           # 4. znormalizowany wolumen
    rsi / 100,                      # 5. RSI w zakresie [0, 1]
    macd_normalized,                # 6. MACD znormalizowany
    (close - prev_close) / prev_close,  # 7. zmiana % względem poprzedniej
    float(close > sma20),           # 8. czy powyżej SMA20 (0 lub 1)
]
```

### Opis cech

| # | Cecha | Opis | Zakres |
|---|---|---|---|
| 1 | Znormalizowana cena | `(close - mean) / std` | ~[-3, 3] |
| 2 | Relative range | `(high - low) / close` | [0, ∞) |
| 3 | Kierunek świecy | `(close - open) / close` | [-1, 1] |
| 4 | Wolumen | `volume / mean_volume` | [0, ∞) |
| 5 | RSI | `rsi / 100` | [0, 1] |
| 6 | MACD | `macd_normalized` | ~[-1, 1] |
| 7 | Zmiana % | `(close - prev_close) / prev_close` | [-1, 1] |
| 8 | SMA20 | `float(close > sma20)` | {0, 1} |

### Dlaczego 8 cech?

- **Cena znormalizowana** — sieć widzi te same wzorce przy BTC=1000 i BTC=100000
- **Range świecy** — informacja o zmienności/volatilności
- **Kierunek** — bull/bear candle
- **Wolumen** — potwierdzenie siły ruchu
- **RSI** — momentum, wykupienie/wyprzedanie
- **MACD** — trend, konwergencja/dywergencja
- **Zmiana %** — momentum krótkoterminowe
- **SMA20** — pozycja względem trendu

## Normalizacja

### Zasada

**Nigdy surowe ceny** — zawsze zmiany procentowe lub znormalizowane wartości.

### Dlaczego?

Sieć musi widzieć te same wzorce bez względu na to czy BTC kosztuje 1000 czy 100000. Surowe ceny zmieniają się o rzędy wielkości — wzorce są te same, ale wartości inne.

### Metody normalizacji

#### Z-score normalizacja

```python
mean = prices.rolling(window=20).mean()
std = prices.rolling(window=20).std()
normalized = (close - mean) / std
```

#### Min-Max normalizacja

```python
min_val = prices.rolling(window=20).min()
max_val = prices.rolling(window=20).max()
normalized = (close - min_val) / (max_val - min_val)
```

#### Percentage change

```python
pct_change = (close - prev_close) / prev_close
```

## Wskaźniki techniczne

### RSI (Relative Strength Index)

```python
# 14-period RSI
delta = close.diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = -delta.where(delta < 0, 0).rolling(14).mean()
rs = gain / loss
rsi = 100 - (100 / (1 + rs))
```

**Interpretacja:**
- RSI > 70 — wykupienie (możliwa korekta)
- RSI < 30 — wyprzedanie (możliwe odbicie)

### MACD (Moving Average Convergence Divergence)

```python
ema_12 = close.ewm(span=12).mean()
ema_26 = close.ewm(span=26).mean()
macd = ema_12 - ema_26
signal = macd.ewm(span=9).mean()
histogram = macd - signal
```

**Normalizacja MACD:**
```python
macd_normalized = macd / close
```

### SMA (Simple Moving Average)

```python
sma_20 = close.rolling(20).mean()
```

**Sygnał binarny:**
```python
above_sma = float(close > sma_20)  # 1 lub 0
```

## Struktura danych wejściowych

### Dla jednego timeframe'a

```python
# Shape: [num_candles, num_features]
state_1m = torch.zeros(15, 8)   # 15 świec × 8 cech
state_15m = torch.zeros(15, 8)  # 15 świec × 8 cech
state_1h = torch.zeros(20, 8)   # 20 świec × 8 cech
state_1d = torch.zeros(30, 8)   # 30 świec × 8 cech
state_1w = torch.zeros(54, 8)   # 54 świece × 8 cech
```

### Łączny rozmiar stanu

```python
# Łączna liczba elementów
total = 15*8 + 15*8 + 20*8 + 30*8 + 54*8 = 1072 floatów
```

## Źródła danych

### Binance API

```python
import requests

def fetch_klines(symbol, interval, limit=1000):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    response = requests.get(url, params=params)
    return response.json()
```

### Interwały Binance

| Interwał | Wartość |
|---|---|
| 1 minuta | 1m |
| 15 minut | 15m |
| 1 godzina | 1h |
| 1 dzień | 1d |
| 1 tydzień | 1w |

### Cache danych

Dane są cachowane lokalnie aby uniknąć zbędnych zapytań API i przyspieszyć trening.

```
node/data/cache/
├── BTCUSDT_1m.csv
├── BTCUSDT_15m.csv
├── BTCUSDT_1h.csv
├── ...
```

## Pobieranie danych historycznych — download_data.js

### Opis

Skrypt `scripts/download_data.js` służy do pobrania danych historycznych z Binance API i zapisania ich lokalnie. Dzięki temu trening może działać offline i nie zależy od rate-limitów API.

### Uruchomienie

```bash
cd scripts
node download_data.js --symbol BTCUSDT --interval 1m --days 30
```

### Parametry

| Parametr | Opis | Domyślnie |
|---|---|---|
| `--symbol` | Para kryptowalutowa | BTCUSDT |
| `--interval` | Timeframe (1m, 15m, 1h, 1d, 1w) | 1h |
| `--days` | Liczba dni wstecz | 30 |
| `--output` | Folder zapisu (relative do projektu) | node/data/historical/ |

### Format zapisu

Dane są zapisywane jako **CSV** z kolumnami:

```csv
timestamp,open,high,low,close,volume,close_time,quote_volume,trades,taker_buy_base,taker_buy_quote,ignore
1697000000000,34500.5,34600.0,34400.0,34550.0,1234.5,...
```

### Lokalizacja plików

```
node/data/historical/
├── BTCUSDT_1m.csv
├── BTCUSDT_15m.csv
├── BTCUSDT_1h.csv
├── BTCUSDT_1d.csv
├── BTCUSDT_1w.csv
├── ETHUSDT_1h.csv
└── SOLUSDT_1h.csv
```

### Jak Actor wie gdzie szukać danych?

Actor ładuje ścieżkę z `config.toml`:

```toml
[data]
source = "file"  # "api" lub "file"
path = "node/data/historical/"
cache_path = "node/data/cache/"
```

- **source = "file"** — ładuj z plików historycznych
- **source = "api"** — pobieraj z Binance API w czasie rzeczywistym

### Zasięg danych

| Timeframe | Zalecany okres | Min. okres |
|---|---|---|
| 1m | 30 dni | 7 dni |
| 15m | 90 dni | 30 dni |
| 1h | 1 rok | 90 dni |
| 1d | 3 lata | 1 rok |
| 1w | 5+ lat | 2 lata |

### Przykład pobrania pełnego zestawu

```bash
# BTC - wszystkie timeframe'y
node download_data.js --symbol BTCUSDT --interval 1m --days 30
node download_data.js --symbol BTCUSDT --interval 15m --days 90
node download_data.js --symbol BTCUSDT --interval 1h --days 365
node download_data.js --symbol BTCUSDT --interval 1d --days 1095
node download_data.js --symbol BTCUSDT --interval 1w --days 1825

# ETH, SOL - 1h (wystarczy dla Faz 2+)
node download_data.js --symbol ETHUSDT --interval 1h --days 365
node download_data.js --symbol SOLUSDT --interval 1h --days 365
```

## Przygotowanie danych

### Pipeline

```python
def prepare_data(raw_klines):
    # 1. Ekstrakcja OHLCV
    df = extract_ohlcv(raw_klines)
    
    # 2. Oblicz wskaźniki
    df['rsi'] = calculate_rsi(df['close'])
    df['macd'] = calculate_macd(df['close'])
    df['sma20'] = calculate_sma(df['close'], 20)
    
    # 3. Normalizacja
    df['close_norm'] = z_score_normalize(df['close'])
    df['volume_norm'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # 4. Buduj features
    features = build_features(df)
    
    # 5. Podziel na sekwencje
    sequences = create_sequences(features)
    
    return sequences
```

## Config

Wszystkie parametry danych są konfigurowalne:

```toml
[features]
num_features = 8

[timeframes]
candles_1m  = 15
candles_15m = 15
candles_1h  = 20
candles_1d  = 30
candles_1w  = 54
## Synchronizacja czasowa timeframe'ów

Każdy krok Actor buduje stan używając świec które są **zamknięte** w momencie bieżącej świecy 1m. Nie można używać świec które jeszcze nie są zamknięte — to byłoby patrzenie w przyszłość (look-ahead bias).

### Zasada

Świeca wyższego timeframe'a jest aktualna jeśli jej `close_time <= current_1m_close_time`.

```javascript
function getAlignedCandles(allCandles, currentTime, timeframe) {
    // Znajdź ostatnią świecę której close_time nie przekracza currentTime
    return allCandles[timeframe]
        .filter(c => c.close_time <= currentTime)
        .slice(-config.timeframes[`candles_${timeframe}`]);
}
```

### Przykład z konkretnym timestamp

```
Bieżąca świeca 1m: close_time = 2024-01-15 14:30:00

Świece 15m: ostatnia zamknięta to 14:15-14:30 (close_time = 14:30:00) ✓
Świece 1h:  ostatnia zamknięta to 13:00-14:00 (close_time = 14:00:00) ✓
            świeca 14:00-15:00 jest otwarta — pomijamy ✗
Świece 1d:  ostatnia zamknięta to 2024-01-14 (close_time = 2024-01-15 00:00:00) ✓
Świece 1w:  ostatnia zamknięta to tydz. kończący 2024-01-14 ✓
```

### Dlaczego to ważne?

Bez synchronizacji Actor mógłby patrzeć na niezamkniętą świecę 1h która zawiera informacje o przyszłości względem bieżącej świecy 1m — model nauczyłby się cheatingować na danych historycznych i zawodziłby w produkcji.

## Rate limiting Binance API

Binance ma limity zapytań które przy wielu Aktorach można łatwo przekroczyć.

### Limity Binance REST API

| Endpoint | Limit |
|---|---|
| GET /api/v3/klines | 1200 requestów/minutę (waga 1-10 per request) |
| Ogólny limit | 6000 wag/minutę |

### Centralny manager zapytań

`data/binance.js` kolejkuje wszystkie requesty i pilnuje limitów — Actorzy nie odpytują API bezpośrednio.

```javascript
class BinanceRateLimiter {
    constructor(config) {
        this.maxRequestsPerMinute = config.data.binance_rate_limit;
        this.queue = [];
        this.requestsThisMinute = 0;

        // Reset licznika co minutę
        setInterval(() => { this.requestsThisMinute = 0; }, 60000);
    }

    async fetch(symbol, interval, limit) {
        // Czekaj jeśli limit osiągnięty
        while (this.requestsThisMinute >= this.maxRequestsPerMinute) {
            await sleep(1000);
        }

        this.requestsThisMinute++;
        return await fetchFromBinance(symbol, interval, limit);
    }
}
```

### Konfiguracja

```toml
[data]
source = "file"                 # "file" lub "api"
path = "node/data/historical/"
cache_path = "node/data/cache/"
binance_rate_limit = 1000       # Max requestów/minutę (margines bezpieczeństwa)
normalization_window = 20
allow_partial_history = true
```

### Retry logic przy błędach API

```javascript
async function fetchWithRetry(symbol, interval, limit, maxRetries = 3) {
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
            return await rateLimiter.fetch(symbol, interval, limit);
        } catch (err) {
            if (attempt === maxRetries) {
                logger.error(`Fetch failed after ${maxRetries} retries: ${err.message}`);
                throw err;
            }
            // Exponential backoff: 1s, 2s, 4s
            const delay = Math.pow(2, attempt - 1) * 1000;
            logger.warn(`Binance API error (attempt ${attempt}/${maxRetries}), retry in ${delay}ms`);
            await sleep(delay);
        }
    }
}
```

### Zachowanie gdy API niedostępne

Gdy Binance API jest niedostępne przez dłuższy czas (wszystkie retry wyczerpane):
- Actor loguje błąd i **pauzuje trening** dla tej pary
- Pozostałe Actory działają normalnie
- Monitoring Service wyświetla alert o niedostępności pary
- Actor próbuje ponownie co `api_retry_interval_sec` sekund

```toml
[data]
api_retry_interval_sec = 60    # Próbuj ponownie co minutę przy długiej niedostępności
```

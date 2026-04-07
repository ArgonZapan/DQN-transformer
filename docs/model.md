# Model sieci neuronowej

## Architektura ogólna

Model łączy **konwolucyjne bloki wydobywające wzorce** z **transformerem do analizy zależności między timeframe'ami**. Architektura typu **Dueling DQN** oddziela ocenę stanu od oceny akcji.

> **Ważne:** Wszystkie parametry (liczba timeframe'ów, liczba świec, cechy, akcje) są **konfigurowalne przez `config.toml`** — wartości w przykładach są ilustracyjne.

```
N wejść (timeframe'y z konfiguracji)
     │
     ├── Conv1D Block (tf_1)   ─┐
     ├── Conv1D Block (tf_2)   ─┤
     ├── Conv1D Block (tf_3)   ─┼──► Konkatenacja
     ├── ...                   ─┤         │
     └── Conv1D Block (tf_N)   ─┘         │
                                   Projekcja → Reshape (N, feature_dim)
                                        │
                               M× Transformer Encoder Block
                               MultiHeadAttention(H heads, key_dim=K)
                               LayerNorm + Residual
                               FeedForward(F) + LayerNorm
                                        │
                               GlobalAveragePooling1D
                                        │
                               Dense(256)
                                        │
                           ┌────────────┴────────────┐
                      Value(1)               Advantage(A)
                           └────────────┬────────────┘
                                Q = V + (A - mean(A))
```

### Przykład dla domyślnej konfiguracji

```toml
[timeframes]
candles_1m  = 15
candles_15m = 15
candles_1h  = 20
candles_1d  = 30
candles_1w  = 54

[features]
num_features = 8  # OHLCV + RSI + MACD + SMA20
```

```
5 wejść (timeframe'y)
     │
     ├── Conv1D Block (1m)   ─┐
     ├── Conv1D Block (15m)  ─┤
     ├── Conv1D Block (1h)   ─┼──► Konkatenacja (640)
     ├── Conv1D Block (1d)   ─┤         │
     └── Conv1D Block (1w)   ─┘         │
                                   Projekcja → Reshape (5, 128)
                                        │
                               3× Transformer Encoder Block
                               MultiHeadAttention(8 heads, key_dim=64)
                               LayerNorm + Residual
                               FeedForward(512) + LayerNorm
                                        │
                               GlobalAveragePooling1D
                                        │
                               Dense(512) → Dense(256)
                                        │
                           ┌────────────┴────────────┐
                      Value(1)               Advantage(4)
                           └────────────┬────────────┘
                                Q = V + (A - mean(A))
```

## Conv1D Block

### Opis

Każdy timeframe jest przetwarzany przez **osobny blok konwolucyjny**. Conv1D przesuwa małe okienko po sekwencji świec i wykrywa **lokalne wzorce**:
- Formacje cenowe
- Skoki wolumenu
- Lokalne momentum

### Struktura bloku

```python
# num_features z config.toml - modularne cechy
Conv1D(filters=128, kernel_size=3, input_shape=(None, num_features))
BatchNormalization
ReLU
GlobalAveragePooling1D
```

### Dlaczego Conv1D?
- **Lokalność** — wykrywa wzorce w małym oknie czasowym
- **Niezmienniczość** — ten sam wzorzec jest rozpoznawany w dowolnym miejscu sekwencji
- **BatchNormalization** — stabilizuje trening, przyspiesza konwergencję

## Transformer Encoder

### Opis

Po konkatenacji wektorów z Conv1D, dane są reshapowane do **sekwencji N tokenów** (jeden per timeframe) i przetwarzane przez bloki Transformer Encoder.

### Mechanizm Multi-Head Attention

Dla każdego tokenu (timeframe'u) sieć oblicza jak bardzo powinien patrzeć na każdy inny token. Sieć sama odkrywa zależności między timeframe'ami — np. że sygnał z 1h potwierdza sygnał z 15m.

### Struktura bloku

```python
# Parametry z config.toml
MultiHeadAttention(num_heads=n_attention_heads, key_dim=key_dim)  # np. 8 heads, key_dim=64
LayerNormalization
Add (residual connection)

# Feed-Forward
Dense(ff_dim, activation='relu')  # np. 512
LayerNormalization
Add (residual connection)
```

### Dlaczego Residual Connections?

Residual connections są kluczowe — **pozwalają gradientom płynąć przez głęboką sieć** bez zanikania. Bez nich głębokie sieci nie mogą się efektywnie uczyć.

### Parametry Transformer

| Parametr | Wartość | Config field |
|---|---|---|
| Bloki Transformer | 3 | `[model].n_transformer_blocks` |
| Głowice attention | 8 | `[model].n_attention_heads` |
| Key dimension | 64 | `[model].key_dim` |
| Feed-Forward dim | 512 | `[model].ff_dim` |

## Dueling Architecture

### Opis

Trunk sieci rozdziela się na dwa strumienie:

**Value stream** — ocenia jak dobra jest ogólnie sytuacja rynkowa niezależnie od akcji. "Rynek jest teraz niebezpieczny" to informacja niezależna od tego czy kupujesz czy sprzedajesz.

**Advantage stream** — ocenia o ile lepsza jest każda akcja od średniej w tej sytuacji.

### Wzór końcowy

```
Q(s, a) = V(s) + (A(s, a) - mean(A(s, ·)))
```

### Dlaczego Dueling?

Podział przyspiesza uczenie bo sieć może nauczyć się wartości stanu bez konieczności testowania każdej akcji osobno. W tradingu to kluczowe — sieć szybko uczy się że "rynek jest zły" bez przechodzenia przez wszystkie akcje.

### Action Masking — maskowanie niedozwolonych akcji

Nie wszystkie akcje są zawsze dozwolone — akcja **CLOSE** ma sens tylko gdy pozycja jest otwarta, a otwieranie pozycji (LONG/SHORT) tylko gdy brak otwartej.

```python
def forward(self, states, action_mask=None):
    """
    states: stany z 5 timeframe'ów
    action_mask: [batch_size, num_actions] binary mask
                 1 = dozwolona akcja, 0 = zablokowana
    
    Przykładowe maski:
    - brak pozycji: [1, 1, 1, 0]  # LONG, SHORT, HOLD dozwolone, CLOSE zablokowane
    - pozycja LONG: [0, 0, 1, 1]   # HOLD, CLOSE dozwolone
    - pozycja SHORT:[0, 0, 1, 1]   # HOLD, CLOSE dozwolone
    """
    q_values = self.network(states)
    
    if action_mask is not None:
        # Zablokuj niedozwolone akcje (ustaw Q na -inf)
        q_values = q_values.masked_fill(action_mask == 0, float('-inf'))
    
    return q_values
```

### Dlaczego action masking?

- Sieć nie marnuje kroków na niemożliwe akcje
- Szybsza konwergencja — mniej złych wyborów do odkrycia
- Eliminacja edge cases (np. CLOSE gdy brak pozycji)

## Confidence Score — pewność modelu

### Opis

Oprócz Q-values, model może obliczać **confidence score** — miarę pewności co do wyboru najlepszej akcji. Confidence jest używany m.in. do modyfikacji nagród ([więcej w reward.md](reward.md)).

### Implementacja (placeholder)

```python
def get_confidence(self, q_values):
    """
    Oblicz pewność modelu na podstawie rozrzutu Q-values.
    
    Metody:
    1. Entropia rozkładu softmax z Q-values
    2. Różnica między najlepszą a drugą akcją
    3. Wariancja Q-values
    """
    # Przykład: różnica między najlepszą a drugą akcją
    sorted_q = torch.sort(q_values, dim=1, descending=True)
    confidence = sorted_q.values[:, 0] - sorted_q.values[:, 1]
    return torch.sigmoid(confidence)  # normalizacja do [0, 1]
```

> **Szczegóły implementacji** — dokładna formuła confidence score jest zależna od eksperymentów. Placeholder do uzupełnienia po weryfikacji która metoda daje najlepsze wyniki.

## Parametry modelu

| Parametr | Wartość | Config field |
|---|---|---|
| Cechy na świecę | 8 (modularne!) | `[features].num_features` |
| Liczba akcji | 4 (LONG/SHORT/HOLD/CLOSE) | `[model].num_actions` |
| Conv1D kernel size | 3 | `[model].conv_kernel_size` |
| Conv1D filters | 128 | `[model].conv1d_filters` |
| Dropout | 0.1 | `[training].dropout` |
| Łączne parametry | ~5-8M | — |

### Modularne cechy

Liczba cech `num_features = 8` to **domyślna konfiguracja** wynikająca z listy wskaźników. Cechy są modularne — można dodać lub usunąć wskaźnik:

```toml
[features]
# Cechy bazowe
use_ohlcv = true          # 5 cech: open, high, low, close, volume

# Wskaźniki techniczne (modularne)
use_rsi = true            # +1 cecha
use_macd = true           # +1 cecha
use_sma20 = true          # +1 cecha
# use_bollinger_bands = false  # +2 cechy (gdy włączone)
# use_atr = false              # +1 cecha (gdy włączone)

# Wynik: 5 + 1 + 1 + 1 = 8
num_features = 8
```

## Przykład przepływu danych

### Konfiguracja

```python
# Config z config.toml, nie hardcoded!
config = load_config()

timeframes = config['timeframes']       # {'candles_1m': 15, 'candles_15m': 15, ...}
num_features = config['features']['num_features']  # 8
num_actions = config['model']['num_actions']       # 4
```

### Input

```python
# Stany dla timeframe'ów - wymiary z configu
# 1m:  [candles_1m, num_features]
# 15m: [candles_15m, num_features]
# itd.
state_1m  = torch.zeros(15, num_features)   # wartość z config['timeframes']['candles_1m']
state_15m = torch.zeros(15, num_features)   # wartość z config['timeframes']['candles_15m']
state_1h  = torch.zeros(20, num_features)   # wartość z config['timeframes']['candles_1h']
state_1d  = torch.zeros(30, num_features)   # wartość z config['timeframes']['candles_1d']
state_1w  = torch.zeros(54, num_features)   # wartość z config['timeframes']['candles_1w']
```

### Process

```python
# Conv1D extraction
vec_1m  = conv_block(input_1m)   # -> [128]
vec_15m = conv_block(input_15m)  # -> [128]
vec_1h  = conv_block(input_1h)   # -> [128]
vec_1d  = conv_block(input_1d)   # -> [128]
vec_1w  = conv_block(input_1w)   # -> [128]

# Konkatenacja
concat = [vec_1m, vec_15m, vec_1h, vec_1d, vec_1w]  # -> [640]

# Reshape do sekwencji tokenów (5 timeframe'ów)
tokens = reshape(concat)  # -> [5, 128]

# Transformer
x = transformer_blocks(tokens)  # -> [5, 128]
x = global_average_pooling(x)   # -> [128]

# Head
x = dense(256)(x)

# Dueling
V = dense(1)(x)           # -> [1]
A = dense(num_actions)(x) # -> [4]
Q = V + (A - mean(A))     # -> [4]  Q-values dla LONG/SHORT/HOLD/CLOSE

# Action masking (opcjonalne)
action_mask = get_action_mask(position)  # np. [0, 0, 1, 1] dla pozycji LONG
Q = Q.masked_fill(action_mask == 0, float('-inf'))

# Confidence score (opcjonalne)
confidence = get_confidence(Q)  # -> [0, 1]
```

### Output

```python
return {
    'q_values': Q,         # [4] - LONG, SHORT, HOLD, CLOSE
    'action': Q.argmax(),  # najlepsza akcja
    'confidence': confidence,  # pewność modelu
}
```

## Regime Detector (opcjonalny)

Moduł wykrywający reżim rynkowy (trend/boczny/ volatilny). Pozwala modelowi dostosować strategię do aktualnych warunków rynkowych.

## Noisy Linear (opcjonalny)

Zamiast epsilon-greedy, warstwy NoisyLinear dodają szum do wag sieci — automatyczna eksploracja która maleje w miarę treningu.

### Zalety NoisyLinear
- Eliminuje ręczne strojenie epsilon
- Bardziej naturalna eksploracja
- Lepsze wyniki w środowiskach o wysokiej wariancji
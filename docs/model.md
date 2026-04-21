# Model sieci neuronowej

## Architektura ogólna

Model łączy **konwolucyjne bloki wydobywające wzorce** z **transformerem analizującym zależności między timeframe'ami**. Architektura typu **Dueling DQN** oddziela ocenę stanu od oceny akcji.

> **Ważne:** Wszystkie parametry (filtry, liczba bloków, cechy) są **konfigurowalne przez `config.toml`** — wartości w przykładach odpowiadają aktualnej konfiguracji.

```
N wejść (timeframe'y z konfiguracji)
     │
     ├── LayerNorm(num_features)          ← normalizacja wejścia per-sample
     │
     ├── Conv1D Block (tf_1) → [batch, seq_1, conv_filters]  ─┐
     ├── Conv1D Block (tf_2) → [batch, seq_2, conv_filters]  ─┤
     ├── Conv1D Block (tf_3) → [batch, seq_3, conv_filters]  ─┼──► Konkatenacja po osi seq
     └── Conv1D Block (tf_N) → [batch, seq_N, conv_filters]  ─┘         │
                                                              [batch, Σseq, conv_filters]
                                                                         │
                                                    M× Transformer Encoder Block
                                                    MultiHeadAttention(H heads)
                                                    LayerNorm + Residual
                                                    FeedForward(ff_dim) + LayerNorm
                                                                         │
                                                    GlobalAveragePooling (po osi seq)
                                                                         │
                                                    Trunk: Dense(conv_filters×4) + Dense(conv_filters×2)
                                                                         │
                                  ┌──────────────────────────────────────┤
                                  │                              Position Features [10]
                            [batch, trunk_h2]                           │
                                  │                              pos_fc Dense(32)
                                  └─────────── concat ──────────────────┘
                                                    │
                                        [batch, trunk_h2 + 32]
                                                    │
                                   ┌────────────────┴────────────────┐
                              Value(1)                        Advantage(num_actions)
                                   └────────────────┬────────────────┘
                                              Q = V + (A - mean(A))
```

### Aktualna konfiguracja (config.toml)

```toml
[timeframes]
candles_1m  = 60   # 60 świec 1-minutowych
candles_15m = 32   # 32 świece 15-minutowe
candles_1h  = 48   # 48 świec 1-godzinnych
candles_1d  = 14   # 14 świec dziennych
candles_1w  = 0    # wyłączony

[features]
num_features = 11

[model]
conv1d_filters      = 64
conv_kernel_size    = 3
n_transformer_blocks = 1
n_attention_heads   = 4
key_dim             = 16
ff_dim              = 128
```

Łączna długość sekwencji po konkatenacji: `60 + 32 + 48 + 14 = 154` tokenów.

## Conv1D Block

### Opis

Każdy timeframe jest przetwarzany przez **osobny blok konwolucyjny**. Blok zachowuje **pełną sekwencję** świec — nie redukuje jej do jednego wektora (GAP jest stosowany dopiero po Transformerze).

Używa **kauzalnego paddingu** — pad tylko po lewej stronie (przeszłość), bez patrzenia w przyszłość.

### Struktura bloku

```python
# Wejście: [batch, seq_len, num_features]
F.pad(x, (kernel_size - 1, 0))   # kauzalny pad po lewej
Conv1D(num_features → conv_filters, kernel_size=3, padding=0)
LayerNorm(conv_filters)           # per-sample, brak trybu train/eval
GELU()
Dropout(dropout)
# Wyjście: [batch, seq_len, conv_filters]  — pełna sekwencja zachowana
```

### Dlaczego LayerNorm zamiast BatchNorm?

BatchNorm w trybie `eval()` używa statystyk z całego treningu zamiast bieżącego batcha — to powoduje **rozbieżność w Double DQN** (main.eval() do selekcji akcji vs main.train() do backpropu). LayerNorm normalizuje per-sample, więc działa identycznie w obu trybach.

### Dlaczego brak GAP po Conv1D?

Global Average Pooling po każdym TF redukuje N świec do 1 tokenu — Transformer wtedy operuje na 4 tokenach (jednym per TF), co jest zbyteczne (4-tokenowy self-attention to w zasadzie ważona suma). Zamiast tego zwracamy pełną sekwencję i Transformer dostaje **wszystkie świece ze wszystkich TF** (~154 tokenów).

## Transformer Encoder

### Opis

Po konkatenacji sekwencji z Conv1D bloków (`[batch, Σseq, conv_filters]`), dane trafiają do bloków Transformer Encoder. Transformer dostaje wszystkie świece ze wszystkich timeframe'ów jako tokeny — może sam odkryć zależności między różnymi skalami czasowymi.

### Struktura bloku

```python
# Pre-norm residual (LN przed atencją)
MultiHeadAttention(d_model=conv_filters, n_heads, dropout)
LayerNorm + residual

# Feed-Forward
Dense(ff_dim, GELU)
Dropout
Dense(conv_filters)
LayerNorm + residual
```

Po M blokach Transformera: **Global Average Pooling** po osi sekwencji → `[batch, conv_filters]`.

## Position Features Branch

### Opis

Oprócz stanu rynkowego, model dostaje **kontekst pozycji** — osobna gałąź sieci która nie miesza się z danymi cenowymi do momentu łączenia w Dueling head.

### 10 cech pozycji

| # | Cecha | Opis |
|---|---|---|
| 1 | `is_long` | 1 jeśli otwarta pozycja LONG |
| 2 | `is_short` | 1 jeśli otwarta pozycja SHORT |
| 3 | `unrealized_pnl` | Niezrealizowany P&L bieżącej pozycji |
| 4 | `hold_6h` | Liczba kroków trzymania / 6h w krokach |
| 5 | `hold_48h` | Liczba kroków trzymania / 48h w krokach |
| 6 | `sin_hour` | sin(2π × godzina/24) |
| 7 | `cos_hour` | cos(2π × godzina/24) |
| 8 | `sin_week` | sin(2π × dzień_tygodnia/7) |
| 9 | `cos_week` | cos(2π × dzień_tygodnia/7) |
| 10 | `is_weekend` | 1 w weekendy (sobota, niedziela) |

### Implementacja

```python
# pos_fc: Linear(10 → 32) + GELU
if position_features is not None:
    pos = self.pos_fc(position_features)   # [batch, 32]
else:
    pos = torch.zeros(batch, 32)           # brak kontekstu = zera

x = torch.cat([trunk_out, pos], dim=1)    # [batch, trunk_h2 + 32]
```

## Dueling Architecture

### Opis

Trunk sieci (+ gałąź pozycji) rozdziela się na dwa strumienie:

**Value stream** V(s) — ocenia jak dobra jest ogólnie sytuacja.

**Advantage stream** A(s,a) — ocenia o ile lepsza jest każda akcja od średniej.

### Wzór końcowy

```
Q(s, a) = V(s) + (A(s, a) - mean(A(s, ·)))
```

Odejmowanie `mean(A)` jest kluczowe — bez niego V i A nie byłyby jednoznacznie wyznaczone (można by dodać stałą do V i odjąć od A bez zmiany Q).

### Action Masking

Nie wszystkie akcje są zawsze dozwolone:

```python
def forward(self, states, action_mask=None, position_features=None):
    q_values = ...
    if action_mask is not None:
        # -inf blokuje akcję — softmax/argmax nigdy jej nie wybierze
        q_values = q_values.masked_fill(action_mask == 0, float('-inf'))
    return q_values
```

| Stan pozycji | Dozwolone akcje |
|---|---|
| Brak pozycji | LONG, SHORT, HOLD |
| Otwarta LONG | HOLD, CLOSE |
| Otwarta SHORT | HOLD, CLOSE |

## Input Normalization

Przed Conv1D blokami sieć stosuje **LayerNorm na wejściu** — wyrównuje skale 11 cech. MACD jest unbounded, volume może skakać 100×, ceny mają Z-score ±3. Bez normalizacji wejścia cechy o dużej skali dominowałyby gradienty.

```python
self.input_norm = nn.LayerNorm(num_features)  # per-sample, brak trybu train/eval
```

## Inicjalizacja głowic

Dueling heads są inicjalizowane do bliskich zera, żeby zapobiec wczesnemu action collapse (gdy sieć od razu preferuje jedną akcję bez treningu):

```python
nn.init.uniform_(advantage_stream.weight, -0.01, 0.01)
nn.init.zeros_(advantage_stream.bias)
nn.init.uniform_(value_stream.weight, -0.01, 0.01)
nn.init.zeros_(value_stream.bias)
```

## Parametry modelu (aktualna konfiguracja)

| Parametr | Wartość | Config field |
|---|---|---|
| Cechy na świecę | 11 | `[features].num_features` |
| Liczba akcji | 4 (LONG/SHORT/HOLD/CLOSE) | `[model].num_actions` |
| Conv1D filters | 64 | `[model].conv1d_filters` |
| Conv1D kernel size | 3 | `[model].conv_kernel_size` |
| Transformer bloków | 1 | `[model].n_transformer_blocks` |
| Attention heads | 4 | `[model].n_attention_heads` |
| FF dim | 128 | `[model].ff_dim` |
| Dropout | 0.1 | `[training].dropout` |
| Position features | 10 → 32 | hardcoded |
| Trunk | 64→256→128 | conv_filters×4, ×2 |
| Łączna sekwencja | 154 tokenów | sum timeframes |

## Przepływ danych (przykład)

```python
# Wejście
states = {
    'candles_1m':  [batch, 60, 11],
    'candles_15m': [batch, 32, 11],
    'candles_1h':  [batch, 48, 11],
    'candles_1d':  [batch, 14, 11],
}
position_features = [batch, 10]   # kontekst pozycji

# Conv1D (per TF, po LayerNorm wejścia)
conv_1m  = conv_block_1m(states['candles_1m'])   # [batch, 60, 64]
conv_15m = conv_block_15m(states['candles_15m']) # [batch, 32, 64]
conv_1h  = conv_block_1h(states['candles_1h'])   # [batch, 48, 64]
conv_1d  = conv_block_1d(states['candles_1d'])   # [batch, 14, 64]

# Konkatenacja sekwencji
x = cat([conv_1m, conv_15m, conv_1h, conv_1d], dim=1)  # [batch, 154, 64]

# Transformer (1 blok)
x = transformer(x)    # [batch, 154, 64]
x = x.mean(dim=1)     # GAP: [batch, 64]

# Trunk
x = trunk(x)          # [batch, 64→256→128]

# Position branch
pos = pos_fc(position_features)  # [batch, 10→32]
x = cat([x, pos], dim=1)         # [batch, 160]

# Dueling heads
V = value_stream(x)      # [batch, 1]
A = advantage_stream(x)  # [batch, 4]
Q = V + (A - mean(A))    # [batch, 4]

# Action masking
Q = Q.masked_fill(action_mask == 0, float('-inf'))
```

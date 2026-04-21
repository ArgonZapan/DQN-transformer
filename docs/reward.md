# System nagradzania

## Przegląd

System nagradzania składa się z dwóch warstw:

1. **Nagroda per krok** — natychmiastowe sygnały przy każdym kroku (delta uPnL, kary za bezczynność i trzymanie)
2. **Nagroda przy zamknięciu pozycji** — zrealizowany P&L z prowizją, time decay i karą za drawdown

> **Ważne:** Wszystkie wartości są **konfigurowalne przez `config.toml` [reward]** — zero hardcoded w kodzie.

## Aktualna konfiguracja

```toml
[reward]
commission_open  = 0.00075   # prowizja Binance za otwarcie (taker 0.075%)
commission_close = 0.00075   # prowizja Binance za zamknięcie (taker 0.075%)
trade_penalty    = 0.0       # dodatkowa kara za otwarcie (można zwiększyć przy churning)
close_penalty    = 0.0       # dodatkowa kara za zamknięcie

post_close_cooldown_steps = 12  # trade_penalty ×2 gdy nowe otwarcie < N kroków po zamknięciu
intermediate_reward_max   = 0.1 # cap absolutny na delta uPnL per krok

loss_scale            = 1.4    # mnożnik ujemnych delt uPnL (straty ważniejsze)
hold_penalty_per_bar  = 0.0002 # kara per krok za trzymanie pozycji (anty-przetrzymanie)
time_decay_hours      = 4.0    # half-life: po 4h trzymania PnL × 0.5
drawdown_penalty      = 0.0    # kara za maxDrawdown podczas trzymania
idle_penalty_per_bar  = 0.0002 # kara per krok BEZ pozycji (anty-HOLD collapse)

clip_min = -1.0   # dolny clip całkowitej nagrody per krok
clip_max =  1.0   # górny clip całkowitej nagrody per krok

reward_scale            = 1.0   # globalny mnożnik wszystkich nagród
unrealized_reward_scale = 0.02  # mnożnik absolutnego uPnL dodawanego per krok
```

## Architektura nagród

### 1. Przy otwarciu pozycji

```
reward_open = -(commission_open + trade_penalty)
```

Jeśli otwarcie nastąpiło przed upłynięciem cooldown od ostatniego zamknięcia:
```
trade_penalty jest podwojona
```

Sieć od razu "czuje" koszt wejścia.

### 2. Per krok (HOLD z otwartą pozycją)

```
delta_upnl = unrealized_pnl[t] - unrealized_pnl[t-1]

# Asymetria: straty ważniejsze
if delta_upnl < 0:
    scaled_delta = delta_upnl * loss_scale     # × 1.4
else:
    scaled_delta = delta_upnl

# Cap absolutny
clipped = clip(scaled_delta, -intermediate_reward_max, +intermediate_reward_max)

# Kara za trzymanie (anty-przetrzymywanie)
reward_step = clipped × unrealized_reward_scale - hold_penalty_per_bar
```

### 3. Per krok (HOLD bez pozycji)

```
reward_step = -idle_penalty_per_bar   # zniechęcenie do bezczynności
```

### 4. Przy zamknięciu pozycji

```
pnl = realized P&L (LONG: (close-open)/open, SHORT: (open-close)/open)

# Time decay: zniechęca do przetrzymywania
holding_hours = (close_time - open_time) / 3_600_000
time_decay = 1.0 / (1.0 + holding_hours / time_decay_hours)
pnl_decayed = pnl * time_decay

# Kara za drawdown podczas trzymania
drawdown_loss = max_drawdown × drawdown_penalty

reward_close = pnl_decayed - commission_close - close_penalty - drawdown_loss
reward_close = clip(reward_close, clip_min, clip_max)
```

## Przepływ nagród przez epizod

```
Krok 1: LONG otwarta
   → reward = -(commission_open + trade_penalty)         ← natychmiastowy koszt

Krok 2-N: HOLD
   → reward = delta_uPnL × scale - hold_penalty_per_bar  ← per-krok signal

Krok N+1: CLOSE
   → reward = pnl × decay - commission_close            ← zrealizowany wynik

Krok N+2-M: HOLD (brak pozycji)
   → reward = -idle_penalty_per_bar                      ← zniechęcenie do leżenia
```

## Prowizja

### Binance taker fees

| Strona | Prowizja |
|---|---|
| Otwarcie | 0.075% |
| Zamknięcie | 0.075% |
| **Round trip** | **0.15%** |

### Dlaczego prowizja jest kluczowa?

Bez prowizji sieć nie widzi kosztu transakcji i uczy się churning'u (ciągłego kupowania i sprzedawania). Już przy 10 round-tripach na epizod koszty wynoszą 1.5% — więcej niż typowy zysk.

## Time Decay

```
time_decay = 1.0 / (1.0 + t_hours / time_decay_hours)
```

| Czas trzymania | Decay (time_decay_hours=4) |
|---|---|
| 0h | 1.00 (brak kary) |
| 4h | 0.50 (połowa P&L) |
| 8h | 0.33 |
| 24h | 0.14 |

Zniechęca do przetrzymywania pozycji bez wyraźnego trendu.

## Asymetria strat (loss_scale)

```python
if delta_upnl < 0:
    scaled = delta_upnl * 1.4   # straty ważą więcej
else:
    scaled = delta_upnl
```

Empirycznie straty bardziej destabilizują trening niż równoważne zyski — `loss_scale > 1.0` wzmacnia sygnał z ujemnych przejść.

## Cooldown po zamknięciu

```
post_close_cooldown_steps = 12

Jeśli kroki_od_zamknięcia < 12:
    trade_penalty = trade_penalty × 2
```

Zniechęca do natychmiastowego ponownego wejścia zaraz po zamknięciu pozycji.

## Kara za bezczynność (idle_penalty)

```python
if not position_open:
    reward -= idle_penalty_per_bar   # 0.0002 per krok
```

Zapobiega degeneracji do strategii "zawsze HOLD" bez otwartej pozycji.

## Kara za trzymanie (hold_penalty)

```python
if position_open:
    reward -= hold_penalty_per_bar   # 0.0002 per krok
```

Zniechęca do przetrzymywania — razem z time decay tworzy presję na zamknięcie pozycji w odpowiednim momencie.

## Clipowanie nagród

Całkowita nagroda per krok jest zawsze w przedziale `[clip_min, clip_max] = [-1.0, 1.0]`.

Clipowanie zapewnia **stabilność gradientów** — ekstremalne nagrody destabilizują trening.

## Reward scale

```python
reward = raw_reward * reward_scale   # globalna amplifikacja sygnału
```

Przy `reward_scale = 1.0` (domyślnie) — bez zmian. Zwiększenie wzmacnia sygnał gdy nagrody są zbyt małe do efektywnego uczenia.

## N-step Returns

Nagrody per krok są sumowane w N-krokowym zwrocie przed wysłaniem do bufora:

```python
# n_step = 25
R_t^n = Σ_{k=0}^{n-1} γ^k × r_{t+k}
target = R_t^n + (1 - done) × γ^n × Q(s_{t+n})
```

Pozwala sygnaływi nagrody propagować się przez 25 kroków wstecz.

## Typowe problemy

### Churning (overtrading)

**Objaw:** Sieć otwiera i zamyka pozycje co 1-2 kroki.

**Rozwiązanie:**
```toml
trade_penalty = 0.001         # dodaj karę za otwarcie
post_close_cooldown_steps = 20  # dłuższy cooldown
```

### HOLD collapse (zawsze HOLD)

**Objaw:** Sieć nigdy nie otwiera pozycji.

**Rozwiązanie:**
```toml
idle_penalty_per_bar = 0.0005   # silniejsza kara za bezczynność
trade_penalty = 0.0             # usuń lub zmniejsz karę za otwarcie
```

### Przetrzymywanie pozycji

**Objaw:** Sieć trzyma pozycje zbyt długo (straty narastają).

**Rozwiązanie:**
```toml
hold_penalty_per_bar = 0.0005  # silniejsza kara per krok z pozycją
time_decay_hours = 2.0          # szybszy decay PnL
```

### Nigdy nie zamyka ze stratą

**Objaw:** Sieć unika CLOSE gdy pozycja jest na minusie.

**Rozwiązanie:**
```toml
drawdown_penalty = 0.5    # kara za głęboki drawdown podczas trzymania
loss_scale = 1.8          # silniejsza penalizacja ujemnych delta uPnL
```

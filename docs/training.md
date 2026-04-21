# Proces treningu

## Przegląd

Trening odbywa się w architekturze Actor-Learner: Node.js Actorzy zbierają doświadczenia i wysyłają je do Python Learnera, który trenuje model i odsyła epsilon do regulacji eksploracji.

## Aktualna konfiguracja

```toml
[training]
gamma                    = 0.97       # dyskontowanie nagród
lr                       = 1.0e-4    # learning rate
batch_size               = 512
buffer_capacity          = 500000    # maks. pojemność bufora PER
min_buffer_size          = 100000    # min. doświadczeń przed startem treningu
target_update_interval   = 1000      # co ile kroków sync target network

epsilon_start            = 0.90
epsilon_end              = 0.10
epsilon_decay_fraction   = 0.20      # 20% kroków (× buffer_capacity) do osiągnięcia epsilon_end

dropout                  = 0.1
seed                     = 42

checkpoint_interval      = 20000     # co ile kroków zapis checkpointu
evaluation_interval      = 20000     # co ile kroków raport / ewaluacja
keep_last_n_checkpoints  = 24

validation_days          = 30        # ostatnie N dni jako OOS
training_months          = 48        # ile miesięcy danych treningowych (0 = cała historia)

episode_length           = 500       # max kroków epizodu
step_interval            = 15        # co ile świec 1m wykonywany jest krok aktora
max_trades_per_episode   = 1         # 1 = zamknięcie kończy epizod

return_mode              = "nstep"   # "mc" | "nstep" | "td"
n_step                   = 25        # horyzont n-krokowy

compile_model            = false     # torch.compile (PyTorch 2+)
min_hold_steps           = 4         # min kroków od otwarcia przed możliwością CLOSE
lr_scheduler             = "cosine"  # "none" | "cosine"
lr_min                   = 1e-5      # dolna granica LR dla cosine schedulera
hold_bias_min            = 0.5       # min szansa HOLD przy losowej eksploracji
hold_bias_max            = 0.5       # max szansa HOLD przy losowej eksploracji
epsilon_reset_target     = 0.50      # docelowy epsilon przy /ereset
epsilon_reset_duration   = 10000     # na ile kroków epsilon_reset_target jest utrzymywany
episode_mirror           = false     # dublowanie epizodów LONG→SHORT
actor_throttle_ms        = 20        # ms opóźnienia wysyłane do aktora gdy bufor pełny
```

## Return Modes

Leaner obsługuje trzy tryby obliczania celu Bellmana:

### nstep (domyślny, n_step=25)

```
R_t^n = Σ_{k=0}^{n-1} γ^k × r_{t+k}
target = R_t^n + (1 - done) × γ^n × Q_target(s_{t+n})
```

Sieć "widzi" przyszłość 25 kroków wprzód — szybsza propagacja sygnałów nagród.

### mc (Monte Carlo)

```
G_t = r_t + γ × r_{t+1} + γ² × r_{t+2} + ... (do końca epizodu)
target = G_t   (done=True wszędzie → no bootstrapping)
```

### td (1-step TD)

```
target = r_t + (1 - done) × γ × Q_target(s_{t+1})
```

Klasyczny 1-step Bellman. Szybki ale wysoka wariancja.

## Krok treningowy

```python
# 1. Sample z bufora PER
states, actions, rewards, next_states, dones, weights, indices = buffer.sample(batch_size)

# 2. Oblicz target (Double DQN)
with torch.no_grad():
    # Main network wybiera akcję — target network ocenia wartość
    next_actions = main_network(next_states).argmax(dim=1)
    next_values = target_network(next_states).gather(1, next_actions.unsqueeze(1))
    targets = rewards + (1 - dones) * gamma_n * next_values   # gamma_n = gamma^n_step

# 3. Oblicz loss (PER-weighted)
current_values = main_network(states).gather(1, actions.unsqueeze(1))
td_errors = (targets - current_values).squeeze(1)
loss = (td_errors ** 2 * weights).mean()

# 4. Backprop z AMP (Mixed Precision)
with torch.autocast('cuda'):
    loss = compute_loss()
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# 5. Aktualizuj priorytety w PER
buffer.update_priorities(indices, td_errors.abs().cpu().numpy())

# 6. Target network sync (co target_update_interval kroków)
if step % target_update_interval == 0:
    target_network.load_state_dict(main_network.state_dict())
```

## Mixed Precision (AMP)

Trening używa **FP16 Mixed Precision** gdy dostępne jest GPU (karta Ampere/Turing):

```python
self.use_amp = device != 'cpu' and torch.cuda.is_available()
self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
```

Korzyści: ~2× speedup na kartach z Tensor Cores, niższe zużycie VRAM.

## LR Scheduler (Cosine Annealing)

```toml
lr_scheduler = "cosine"
lr_min       = 1e-5
```

LR maleje od `lr=1e-4` do `lr_min=1e-5` przez czas trwania epsilon decay (20% × 500k = 100k kroków), zsynchronizowany z eksploracją.

```python
T_max = int(epsilon_decay_fraction × buffer_capacity)
scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=lr_min)
```

## Epsilon Decay

```python
epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) × (steps / decay_steps))
```

Gdzie `decay_steps = epsilon_decay_fraction × buffer_capacity = 0.20 × 500000 = 100000`.

### Hold Bias

Przy losowej eksploracji HOLD otrzymuje wyższe prawdopodobieństwo wyboru:

```toml
hold_bias_min = 0.5
hold_bias_max = 0.5
```

Zapobiega dominacji OPEN akcji w early exploration, co prowadziłoby do churning'u w buforze.

### Epsilon Reset

Mechanizm ręcznego resetu epsilon (przez /ereset lub przy wykrytym action collapse):

```toml
epsilon_reset_target   = 0.50
epsilon_reset_duration = 10000
```

Po wywołaniu resetu epsilon wraca do 0.5 i pozostaje tam przez 10000 kroków.

## Min Hold Steps

```toml
min_hold_steps = 4
```

Model nie może wybrać CLOSE przez pierwsze 4 kroki od otwarcia pozycji. Implementowane przez action masking — CLOSE jest blokowane dla `t < min_hold_steps`.

Zapobiega otwieraniu i natychmiastowemu zamykaniu pozycji.

## Episode Mirror

```toml
episode_mirror = false
```

Gdy `true` — po każdym epizodzie generowany jest lustrzany: wszystkie LONG↔SHORT są zamienione. Balansuje dane treningowe między kierunkami (50/50 LONG/SHORT).

## Actor Throttle

```toml
actor_throttle_ms = 20
```

Gdy bufor jest pełny (>95% pojemności), Learner wysyła throttle do Actorów — Node.js czeka 20ms przed kolejnym krokiem. Zapobiega przepełnieniu kolejki doświadczeń gdy Learner nie nadąża z treningiem.

## Replay Buffer

### Konfiguracja PER

```toml
[per]
alpha            = 0.6    # stopień priorytyzacji
beta_start       = 0.4    # początkowa waga IS (korekcja biasu)
beta_end         = 1.0    # docelowa waga IS
epsilon          = 0.0001 # minimalna wartość priorytetu

positive_ratio   = 0.4    # frakcja batcha z pozytywnych nagród
long_short_balance = 0.3  # frakcja z osobnych buforów LONG/SHORT
```

### DualPrioritizedBuffer

System używa `DualPrioritizedBuffer` — kompozycja trzech buforów:
- Główny PER (doświadczenia z wagami TD error)
- Bufor pozytywnych nagród (reward > 0) — przeciwdziała rzadkości sygnału
- Bufor LONG/SHORT balance — utrzymuje równy rozkład kierunków pozycji

### Pinned Memory

Pre-alokowany bufor z pinned memory:

```python
self.states_1m = torch.zeros(capacity, 60, 11, dtype=torch.float32).pin_memory()
```

Transfer CPU→GPU przez DMA (`non_blocking=True`) bez blokowania CPU.

## Checkpointing

### Zapis

```python
torch.save({
    'step': step,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'epsilon': epsilon,
}, f'checkpoint_step_{step}.pt')
```

### Priorytet ładowania

1. Jawne wskazanie w `config.toml`: `resume_from_checkpoint = "path/to/checkpoint.pt"`
2. Shutdown checkpoint: `checkpoints/shutdown_checkpoint.pt`
3. Trening od zera

```toml
[training]
resume_from_checkpoint = ""   # "" = auto (shutdown checkpoint lub od zera)
keep_last_n_checkpoints = 24  # usuwa starsze
```

## ONNX Export

Po każdym checkpoincie model jest eksportowany do ONNX — umożliwia lokalną inferencję w Node.js przez `onnxruntime-node` (bez ZMQ round-trip):

```toml
[onnx]
enabled     = true
export_path = "python/checkpoints/model.onnx"
device      = "cpu"   # eksport zawsze na CPU
```

## TensorBoard

```toml
[tensorboard]
log_dir               = "runs"
log_interval_sec      = 300   # co 5 min: loss, epsilon, Q-values, etc.
histogram_interval_sec = 60   # co minutę: histogramy wag
enable_actor_metrics  = true  # metryki tradingowe od Actorów
debug_interval_sec    = 300
```

### Logowane metryki

| Kategoria | Metryki |
|---|---|
| Loss | `train` (średnia z batchy) |
| Epsilon | wartość bieżąca |
| LR | aktualne learning rate |
| TD Error | mean, max_abs, std |
| Q-values | mean, max, min |
| Gradienty | total_norm |
| Buffer | size, fill_ratio |
| PER | beta, priority_mean, priority_max |
| Akcje | count_0..3, ratio_0..3 |
| Wagi | histogram per layer (co 60s) |
| Trading (per Actor) | win_rate, episode_pnl_avg, transactions |

## Walidacja out-of-sample

```toml
[training]
validation_days = 30   # ostatnie 30 dni nigdy nie są używane w treningu
```

Dane są podzielone:
```
|←────── historia - 30 dni ──────→|←── ostatnie 30 dni ──→|
         Trening                          OOS (walidacja)
```

Ewaluacja OOS co `evaluation_interval` kroków — monitoring Sharpe ratio i P&L na niewidzianych danych.

## Plan fazowy

Fazy są **sterowane ręcznie** przez zmianę `config.toml` i restart.

| Parametr | Faza 1 | Faza 2 | Faza 3 | Faza 4 |
|---|---|---|---|---|
| LR | 0.0003 | 0.0001 | 0.0001 | 0.00005 |
| Batch size | 64 | 128 | 256 | 512 |
| Buffer | 50k | 200k | 500k | 2M |
| Gamma | 0.97 | 0.97 | 0.97 | 0.999 |
| N-step | 5 | 10 | 25 | 25 |
| Target update | 500 | 500 | 1000 | 1000 |
| Transformer bloków | 0 | 0 | 1 | 2+ |

## Typowe problemy treningowe

### Loss nie maleje

- LR za wysoki lub za niski
- Zbyt mały batch_size
- Błędne nagrody (sprawdź reward.md)

### Action collapse (zawsze ta sama akcja)

- Zbyt mały epsilon lub za szybki decay
- Alert: `AlertSystem` wysyła Telegram przy `ratio > 0.95`
- Fix: `/ereset` lub zwiększenie `hold_bias_min`

### Q-values rosną bez końca

- Brak target network update
- LR za wysoki
- Sprawdź gradient norm (powinien być < 5.0)

### Gradient explosion

- Alert: `AlertSystem` wysyła Telegram przy `grad_norm > grad_explode_threshold`
- Fix: zmniejsz LR lub dodaj gradient clipping

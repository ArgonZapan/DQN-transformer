# Proces treningu

## Przegląd

Trening modelu odbywa się w **fazach** — od prostego baseline po pełną architekturę. Każda faza ma osobne parametry i cele walidacyjne.

## Plan fazowy

### Faza 1 — Weryfikacja (dni 1-3)

**Cel:** Potwierdzenie że podstawowy pipeline działa.

| Parametr | Wartość |
|---|---|
| Sieć | Mała (1-2 warstwy) |
| Pary | BTCUSDT tylko |
| Timeframe | 15m tylko |
| Buffer | 50k |
| Batch | 64 |
| Learning rate | 0.0003 |
| Gamma | 0.99 |
| N-step | 1 |
| Target update | 500 |

**Kryteria sukcesu:**
- Loss maleje w czasie
- Epsilon spada poprawnie
- Sieć nie wpada w churning (ciągłe kupowanie/sprzedawanie)

> **Ważne:** Nie przechodź dalej dopóki baseline nie działa.

### Faza 2 — Multi-scale (dni 4-10)

**Cel:** Dodanie wszystkich timeframe'ów i par.

| Parametr | Wartość |
|---|---|
| Sieć | Conv1D bloki |
| Pary | BTCUSDT, ETHUSDT, SOLUSDT |
| Timeframe | Wszystkie 5 (1m, 15m, 1h, 1d, 1w) |
| Buffer | 200k |
| Batch | 128 |
| Learning rate | 0.0001 |
| Gamma | 0.99 |
| N-step | 1 |
| Target update | 500 |

**Walidacja:** Weryfikacja out-of-sample na 20% najnowszych danych.

### Faza 3 — Transformer (dni 11-20)

**Cel:** Zastąpienie flat Dense przez bloki Transformer Encoder.

| Parametr | Wartość |
|---|---|
| Sieć | Conv1D + Transformer |
| Buffer | 500k |
| Batch | 256 |
| Learning rate | 0.0001 |
| Gamma | 0.99 |
| N-step | 3 |
| Target update | 1000 |

**Walidacja:** Monitoring czy Sharpe poprawia się względem Fazy 2.

### Faza 4 — Pełna architektura (dni 21-30)

**Cel:** Full Rainbow-lite + hyperparameter tuning.

| Parametr | Wartość |
|---|---|
| Sieć | Pełna Conv1D + Transformer |
| Buffer | 2M |
| Batch | 512 |
| Learning rate | 0.00005 |
| Gamma | 0.999 |
| N-step | 5 |
| Target update | 1000 |
| PER | Włączony |

**Opcjonalnie:** Population Based Training.

## Podsumowanie parametrów per faza

| Parametr | Faza 1 | Faza 2 | Faza 3 | Faza 4 |
|---|---|---|---|---|
| Learning rate | 0.0003 | 0.0001 | 0.0001 | 0.00005 |
| Batch size | 64 | 128 | 256 | 512 |
| Buffer capacity | 50k | 200k | 500k | 2M |
| Gamma | 0.99 | 0.99 | 0.99 | 0.999 |
| N-step | 1 | 1 | 3 | 5 |
| Target update | 500 | 500 | 1000 | 1000 |

## Sterowanie fazami

### Jak przełączane są fazy?

Fazy są **sterowane ręcznie przez użytkownika** — Opus zmienia konfigurację w `config.toml` i restartuje system. Nie ma automatycznego przełączania.

### Procedura zmiany fazy

1. **Zmień parametry w `config.toml`** — aktualna faza wymaga innych wartości
2. **Zapisz checkpoint** — system automatycznie zapisuje model co N kroków
3. **Zatrzymaj system** (Ctrl+C)
4. **Zaktualizuj config** — zmień parametry na wartości dla następnej fazy
5. **Załaduj checkpoint** (opcjonalne) — wznowienie treningu zamiast startu od zera
6. **Uruchom ponownie**

### Przykład: przejście z Fazy 1 do Fazy 2

```toml
# PRZED (Faza 1):
[model]
n_transformer_blocks = 0  # flat Dense tylko

[training]
batch_size = 64
buffer_capacity = 50000
gamma = 0.99
n_step = 1
target_update_interval = 500
lr = 0.0003

# PO (Faza 2):
[model]
n_transformer_blocks = 3  # Conv1D + Transformer

[training]
batch_size = 128
buffer_capacity = 200000
gamma = 0.99
n_step = 1
target_update_interval = 500
lr = 0.0001
```

### Konfiguracja aktywnej pary w Fazie 1

```toml
# ONLY BTC - Faza 1
[[actors]]
symbol = "BTCUSDT"
exchange = "binance"
leverage = 1

# Dodaj ETH, SOL w Fazie 2
# [[actors]]
# symbol = "ETHUSDT"
# ...
```

### Kryteria przejścia między fazami

| Z | Do | Kryterium |
|---|---|---|
| Faza 1 | Faza 2 | Loss stabilnie maleje przez >24h + brak churning |
| Faza 2 | Faza 3 | Sharpe out-of-sample > 0.5 |
| Faza 3 | Faza 4 | Sharpe out-of-sample wyższy niż Faza 2 |

### Automatyzacja (opcjonalnie)

Można zaimplementować skrypt który automatycznie zmienia config po spełnieniu kryteriów:

```python
# scripts/auto_phase_transition.py
def check_phase_transition(metrics, current_phase):
    if current_phase == 1:
        if metrics['loss_trend'] == 'decreasing' and not metrics['churning']:
            return 2
    elif current_phase == 2:
        if metrics['sharpe_oos'] > 0.5:
            return 3
    # ...
    return current_phase
```

> **Rekomendacja:** Ręczne sterowanie fazami daje lepszą kontrolę. Automatyzacja może przeoczyć subtelne problemy (np. overfitting który nie jest widoczny w metrykach).

## Faza wstępna — napełnianie bufora

Przez pierwsze **200-500k kroków** gdy epsilon jest wysoki (sieć i tak losuje akcje), Node.js generuje losowe akcje **bez pytania Pythona**.

### Korzyści
- Bufor zapełnia się znacznie szybciej
- Brak narzutu komunikacji ZMQ
- Brak narzutu inference modelu

### Kiedy kończyć fazę wstępną?
- Buffer zapełniony do ~10-20% pojemności
- epsilon wciąż > 0.8

## Proces trenowania

### Krok treningowy

```python
# 1. Sample z bufora
states, actions, rewards, next_states, dones = buffer.sample(batch_size)

# 2. Oblicz target (Double DQN)
with torch.no_grad():
    next_actions = main_network(next_states).argmax(dim=1)
    next_values = target_network(next_states).gather(1, next_actions.unsqueeze(1))
    targets = rewards + (1 - dones) * gamma * next_values

# 3. Oblicz loss
current_values = main_network(states).gather(1, actions.unsqueeze(1))
loss = mse_loss(current_values, targets)

# 4. Backprop
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### Target update

```python
# Co N kroków
if step % target_update_interval == 0:
    target_network.load_state_dict(main_network.state_dict())
```

## Replay Buffer — pre-alokowana pamięć

### Implementacja

Zamiast listy obiektów Python, bufor to **jeden duży pre-alokowany blok pamięci** per pole. Przy 2M pojemności zajmuje około 1GB RAM.

```python
self.states_1m = torch.zeros(capacity, 15, 8, dtype=torch.float32).pin_memory()
```

### Pinned Memory

**Pinned memory** to specjalny rodzaj RAM który **nie może być swapowany na dysk** — umożliwia szybszy asynchroniczny transfer CPU→GPU przez DMA.

### Transfer na GPU

```python
states = self.states_1m[idx].to('cuda', non_blocking=True)
```

`non_blocking=True` oznacza że transfer odbywa się **asynchronicznie** — CPU może robić inne rzeczy podczas gdy dane lecą na GPU.

## Checkpointing

### Zapisywanie modelu

```python
torch.save({
    'step': step,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'epsilon': epsilon,
}, f'checkpoint_step_{step}.pt')
```

### Ładowanie modelu

```python
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
step = checkpoint['step']
epsilon = checkpoint['epsilon']
```

### Priorytet ładowania checkpointu przy starcie

```python
def find_checkpoint_to_load(config):
    # 1. Jawne wskazanie w configu
    explicit = config['training'].get('resume_from_checkpoint', '')
    if explicit:
        return explicit

    # 2. Shutdown checkpoint (graceful stop)
    shutdown_path = 'checkpoints/shutdown_checkpoint.pt'
    if os.path.exists(shutdown_path):
        return shutdown_path

    # 3. Start od zera
    return None
```

Pełna dokumentacja graceful shutdown i `shutdown_checkpoint.pt`: [Graceful Shutdown](shutdown.md).

## Typowe problemy treningowe

### Loss nie maleje
- Learning rate za wysoki/niski
- Zbyt mały batch size
- Błędne nagrody

### Loss oscyluje
- Zbyt duży learning rate
- Zbyt mały buffer
- Brak target network update

### Sieć zawsze wybiera tę samą akcję
- Zbyt wysoka kara za transakcje
- Niewystarczająca eksploracja (epsilon za niski)
- Zbyt mała różnorodność danych

### Overfitting
- Trening na jednym krótkim wykresie
- Zbyt duża sieć względem danych
- Brak walidacji out-of-sample

## Walidacja out-of-sample

### Procedura

1. Odłóż 20% najnowszych danych przed treningiem
2. Nigdy nie trenuj na tych danych
3. Ewaluacja co 3-4 dni

### Interpretacja

| Sytuacja | Diagnoza |
|---|---|
| In-sample wysoki, out-of-sample niski | Overfitting |
| Oba niskie | Niedouczenie (underfitting) |
| Oba wysokie | Dobrze dopasowany model |
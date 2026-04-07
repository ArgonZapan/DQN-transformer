# System nagradzania

## Przegląd

System nagradzania jest kluczowym elementem uczenia przez wzmacnianie — to jedyny sygnał jaki sieć otrzymuje do nauki. Źle zaprojektowany system nagród prowadzi do niepożądanych zachowań.

> **Ważne:** Wszystkie wartości liczbowe (prowizje, kary, progi clipowania) są **konfigurowalne przez `config.toml`** — wartości pokazane w przykładach są ilustracyjne i nie są hardcoded w kodzie.

```toml
[reward]
commission_open = 0.001      # Prowizja za otwarcie (0.1%)
commission_close = 0.001     # Prowizja za zamknięcie (0.1%)
trade_penalty = 0.001        # Dodatkowa kara za transakcję
clip_min = -1.0              # Minimalna nagroda po clipowaniu
clip_max = 1.0               # Maksymalna nagroda po clipowaniu
intermediate_reward_max = 0.1 # Max nagrody pośredniej
```

## Zasady nagradzania

### Realized P&L tylko

Nagroda jest dawana **tylko przy zamknięciu pozycji** (realized P&L). Podczas trzymania pozycji (akcja HOLD) nagroda wynosi 0.

### Dlaczego nie per-krok?

Nagroda per-krok prowadzi do uczenia się na **unrealized huśtawce cenowej** — sieć widzi "zysk" który jeszcze nie istnieje i może szybko zniknąć.

```python
if position.open:
    reward = 0  # żadna nagroda podczas trzymania (HOLD)
else:
    reward = calculate_realized_pnl(position)
```

## Formuła nagrody

### Podstawowa formuła

```
nagroda = pnl_procentowy
        - prowizja_otwarcia
        - prowizja_zamknięcia
        - kara_za_transakcję
```

### Dla pozycji LONG

```python
def calculate_pnl_long(open_price, close_price):
    """LONG: zysk gdy cena rośnie"""
    return (close_price - open_price) / open_price
```

### Dla pozycji SHORT

```python
def calculate_pnl_short(open_price, close_price):
    """SHORT: zysk gdy cena spada"""
    return (open_price - close_price) / open_price
```

### Uniwersalna formuła

```python
def calculate_pnl(position):
    """Oblicz PnL niezależnie od kierunku pozycji"""
    if position.side == "LONG":
        pnl = (position.close_price - position.open_price) / position.open_price
    elif position.side == "SHORT":
        pnl = (position.open_price - position.close_price) / position.open_price
    else:
        return 0  # brak pozycji
    
    return pnl
```

## Prowizja

### Dlaczego prowizja jest ważna?

Prowizja musi być odjęta od nagrody — **bez niej sieć nie widzi kosztu transakcji** i uczy się overtradingu (churning).

### Binance fees

| Strona | Prowizja |
|---|---|
| Otwarcie | 0.1% |
| Zamknięcie | 0.1% |
| **Round trip** | **0.2%** |

### Implementacja

```python
# Wartości z config.toml, nie hardcoded!
config = load_config()
commission_open = config['reward']['commission_open']
commission_close = config['reward']['commission_close']

reward = pnl - commission_open - commission_close
```

## Kara za transakcję

### Opis

Mała **dodatkowa kara** niezależna od prowizji zniechęca do nadmiernego handlu.

```python
# Wartość z config.toml
trade_penalty = config['reward']['trade_penalty']
reward -= trade_penalty  # przy każdym otwarciu pozycji
```

### Dlaczego dodatkowa kara?

Prowizja Binance to tylko koszt finansowy. Dodatkowa kara modeluje:
- Slippage (różnica między oczekiwaną a rzeczywistą ceną)
- Koszt czasu i zasobów
- Preferencję dla rzadszych, lepszych sygnałów

## Skala nagród i clipping

### Zakresy nagród

| Typ | Zakres | Opis |
|---|---|---|
| Nagrody pośrednie | max ±0.1 | Bicie stop loss, take profit |
| Nagroda końcowa | ±1.0 | Wygrana/przegrana epizodu |
| Po clipowaniu | [-1, 1] | Ostateczny zakres |

### Clipowanie

```python
# Wartości z config.toml
clip_min = config['reward']['clip_min']  # -1.0
clip_max = config['reward']['clip_max']  # 1.0

reward = max(clip_min, min(clip_max, reward))
```

### Dlaczego clipowanie?

Ostateczna nagroda jest zawsze clipowana do zakresu `[-1, 1]` dla **stabilności gradientów**. Ekstremalne nagrody destabilizują trening.

### Dlaczego pośrednie << końcowe?

Nagrody pośrednie muszą być **znacznie mniejsze** niż końcowa — inaczej sieć optymalizuje pod nagrody pośrednie kosztem wyniku końcowego.

## Confidence score (opcjonalny modyfikator)

Jeśli model zwraca **confidence score** (pewność co do akcji), można go użyć do modyfikacji nagrody:

```python
def calculate_reward_with_confidence(position, confidence):
    """
    Confidence w zakresie [0, 1]:
    - 1.0 = pełna pewność (normalna nagroda)
    - 0.5 = średnia pewność (pomniejszona nagroda)
    - 0.0 = brak pewności (zerowa nagroda)
    """
    base_reward = calculate_reward(position)
    
    # Nagroda proporcjonalna do pewności
    reward = base_reward * confidence
    
    return max(clip_min, min(clip_max, reward))
```

### Dlaczego confidence score?

- Sieć uczy się nie tylko **którą akcję** wybrać, ale też **kiedy być pewna**
- Nagradza świadome decyzje, karze losowe zgadywanie
- Przydatne przy niskim epsilon (faza eksploatacji)

## Interaction z Prioritized Experience Replay

### Priorytet w PER

Priorytet doświadczenia w **Prioritized Experience Replay** jest liczony od **TD error**, nie od samej nagrody:

```python
# TD error = target - Q_value
td_error = target_q - current_q

# Priorytet = |TD error| + epsilon (mała stała)
priority = abs(td_error) + epsilon_per
```

### Dlaczego od TD error a nie od nagrody?

- **Clipowana nagroda** traci informację o skali błędu
- **TD error** zachowuje pełną informację o tym jak bardzo sieć się myliła
- Doświadczenia gdzie sieć **bardzo się myliła** (duży TD error) są najważniejsze do nauki

### Proces zapisu do PER

```python
def add_to_buffer(state, action, reward, next_state, done, td_error=None):
    """
    Dodaj doświadczenie do Prioritized Replay Buffer
    
    Jeśli td_error nie jest podany (nowe doświadczenie):
    - Użyj maksymalnego priorytetu (nowe = ważne)
    - Priorytet zostanie zaktualizowany po treningu
    """
    if td_error is None:
        # Nowe doświadczenie - max priorytet
        priority = max_priority
    else:
        # Zaktualizowany TD error z treningu
        priority = abs(td_error) + epsilon_per
    
    buffer.add(state, action, reward, next_state, done, priority)
```

### Aktualizacja priorytetów

```python
def update_priorities(indices, td_errors):
    """
    Po kroku treningu zaktualizuj priorytety samplowanych doświadczeń
    """
    for idx, td_err in zip(indices, td_errors):
        new_priority = abs(td_err) + epsilon_per
        buffer.update_priority(idx, new_priority)
```

## Typowe problemy

### Churning (overtrading)

**Objaw:** Sieć ciągle kupuje i sprzedaje.

**Przyczyny:**
- Brak prowizji w nagrodzie
- Nagroda per-krok zamiast realized only
- Zbyt mała kara za transakcję

**Rozwiązanie:**
```python
# Wartości z config.toml
commission = config['reward']['commission_open'] + config['reward']['commission_close']
trade_penalty = config['reward']['trade_penalty']

# Tylko przy zamknięciu pozycji (CLOSE)
if not position.closed:
    reward = 0
else:
    reward = calculate_pnl(position) - commission - trade_penalty
    reward = max(clip_min, min(clip_max, reward))
```

### Nigdy nie handluje

**Objaw:** Sieć nauczyła się zawsze czekać (akcja "hold").

**Przyczyny:**
- Zbyt duże kary za transakcje
- Zbyt wysoka prowizja
- Zbyt małe nagrody za wygrane

**Rozwiązanie:**
```python
# Zmniejsz wartości w config.toml:
# trade_penalty = 0.0005  # zamiast 0.001
# commission_open = 0.0005  # zamiast 0.001

# Zwiększ gamma aby sieć była cierpliwsza
# gamma = 0.999  # zamiast 0.99
```

### Overfit do danych treningowych

**Objaw:** Na danych treningowych działa świetnie, na nowych losowo.

**Przyczyny:**
- Trening na jednym krótkim wykresie
- Zawsze ten sam start epizodu
- Zbyt duża sieć względem danych

**Rozwiązanie:**
- Losowe starty epizodów
- Różne okresy treningowe
- Walidacja out-of-sample (20% danych)
- Redukcja wielkości sieci

### Gaming reward

**Objaw:** Sieć znajduje lukę w systemie nagród.

**Przykład:** Jeśli nagradzasz za każdą zamkniętą pozycję, sieć będzie otwierać i zamykać pozycje bez zysku byle dostać nagrodę.

**Rozwiązanie:** Nagroda proporcjonalna do P&L (ujemny P&L = ujemna nagroda) + kara za transakcję.

## Przykładowe implementacje

> Wszystkie wartości liczbowe w przykładach są **ilustracyjne**. W produkcji ładuj je z `config.toml`.

### Prosta nagroda (uniwersalna dla LONG/SHORT)

```python
def calculate_reward(position, config):
    """
    Prosta nagroda realized P&L z prowizją i karą.
    Działa zarówno dla LONG jak i SHORT.
    """
    if not position.closed:
        return 0
    
    # Oblicz PnL zależnie od kierunku
    if position.side == "LONG":
        pnl = (position.close_price - position.open_price) / position.open_price
    elif position.side == "SHORT":
        pnl = (position.open_price - position.close_price) / position.open_price
    else:
        return 0
    
    # Prowizja i kara z configu
    commission = config['reward']['commission_open'] + config['reward']['commission_close']
    trade_penalty = config['reward']['trade_penalty']
    
    reward = pnl - commission - trade_penalty
    reward = max(config['reward']['clip_min'], min(config['reward']['clip_max'], reward))
    
    return reward
```

### Nagroda z risk-adjusted return

```python
def calculate_reward_risk_adjusted(position, config):
    """
    Nagroda z karą za maksymalny drawdown podczas trzymania pozycji.
    Zachęca do zamykania pozycji zanim strata stanie się duża.
    """
    if not position.closed:
        return 0
    
    # PnL zależnie od kierunku
    if position.side == "LONG":
        pnl = (position.close_price - position.open_price) / position.open_price
    elif position.side == "SHORT":
        pnl = (position.open_price - position.close_price) / position.open_price
    
    # Kara za drawdown (im większy spadek, tym większa kara)
    max_drawdown = calculate_max_drawdown(position)  # funkcja pomocnicza
    risk_penalty = max_drawdown * config['reward']['drawdown_penalty']
    
    commission = config['reward']['commission_open'] + config['reward']['commission_close']
    trade_penalty = config['reward']['trade_penalty']
    
    reward = pnl - commission - trade_penalty - risk_penalty
    reward = max(config['reward']['clip_min'], min(config['reward']['clip_max'], reward))
    
    return reward
```

### Nagroda z time decay

```python
def calculate_reward_time_decay(position, config):
    """
    Nagroda zmniejszana czasem trzymania pozycji.
    Zachęca do szybszego zamykania pozycji.
    """
    if not position.closed:
        return 0
    
    # PnL zależnie od kierunku
    if position.side == "LONG":
        pnl = (position.close_price - position.open_price) / position.open_price
    elif position.side == "SHORT":
        pnl = (position.open_price - position.close_price) / position.open_price
    
    # Im dłużej trzymasz, tym mniejsza nagroda
    holding_time = position.close_time - position.open_time
    time_decay = 1.0 / (1.0 + holding_time / config['reward']['time_decay_hours'])
    
    commission = config['reward']['commission_open'] + config['reward']['commission_close']
    trade_penalty = config['reward']['trade_penalty']
    
    reward = pnl * time_decay - commission - trade_penalty
    reward = max(config['reward']['clip_min'], min(config['reward']['clip_max'], reward))
    
    return reward
```

### Nagroda z confidence score

```python
def calculate_reward_with_confidence(position, confidence, config):
    """
    Nagroda modyfikowana przez confidence score z modelu.
    Confidence w zakresie [0, 1].
    """
    # Bazowa nagroda
    base_reward = calculate_reward(position, config)
    
    # Modyfikacja przez pewność modelu
    confidence_scale = config.get('reward', {}).get('confidence_scale', 1.0)
    reward = base_reward * (confidence_scale * confidence + (1 - confidence_scale))
    
    reward = max(config['reward']['clip_min'], min(config['reward']['clip_max'], reward))
    
    return reward
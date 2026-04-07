# Algorytm DQN i Rainbow

## Przegląd

Projekt implementuje **Rainbow-lite DQN** — kombinację trzech najważniejszych ulepszeń bazowego algorytmu DQN:
1. Double DQN
2. Prioritized Experience Replay
3. Dueling Networks

## Bazowy DQN

### Wzór Bellmana

```
Q(s, a) ← Q(s, a) + α × [target - Q(s, a)]
```

Gdzie:
- `s` — aktualny stan
- `a` — podjęta akcja
- `α` — learning rate
- `target` — docelowa wartość Q

### Target Q-value

```
TD error = target - Q(stan, akcja)
target   = nagroda + γ × max(Q_target(następny_stan))
```

### Koniec epizodu

Przy zakończeniu epizodu (done=True) drugi składnik odpada:

```
target = nagroda
```

## Double DQN

### Problem bazowego DQN

Bazowy DQN używa tej samej sieci do **wyboru akcji** i do **oceny jej wartości**, co prowadzi do **przeszacowywania Q-values**.

### Rozwiązanie

Double DQN rozdziela te role:

```python
# Główna sieć wybiera akcję
akcja = argmax(główna_sieć(następny_stan))

# Target sieć ocenia wartość
wartość = target_sieć(następny_stan)[akcja]

# Oblicz target
target = nagroda + γ × wartość
```

### Dlaczego to działa?

- Główna sieć wie która akcja jest najlepsza
- Target sieć daje niezależną ocenę wartości
- Eliminacja biasu z przeszacowania

## Prioritized Experience Replay (PER)

### Problem

Nie wszystkie doświadczenia są **równie wartościowe**. Losowe samplowanie traktuje wszystkie jednakowo.

### Rozwiązanie

Doświadczenia gdzie sieć **myliła się bardziej** (duży TD error) są samplowane **częściej**.

### Priorytet

```python
priorytet = |TD error| + ε
```

Gdzie `ε` to mała stała zapobiegająca priorytetowi zero.

### Implementacja

Bufor używa **SumTree** do efektywnego samplowania zgodnie z rozkładem prawdopodobieństwa:

```python
P(i) = priority_i^α / Σ priority_j^α
```

Gdzie `α` kontroluje stopień priorytetyzacji:
- `α = 0` — losowe samplowanie (brak priorytetyzacji)
- `α = 1` — pełna priorytetyzacja
- Typowo: `α = 0.6`

### Bias correction

PER wprowadza bias — samplujemy częściej doświadczenia z dużym error, co może prowadzić do overfitu. kompensujemy przez **importance sampling weights**:

```python
w_i = (1/N × 1/P(i))^β
```

Gdzie `β` rośnie od ~0.4 do 1.0 w czasie treningu.

## Dueling Networks

### Architektura

Sieć rozdziela się na dwa strumienie:

**Value stream** V(s) — ocenia jak dobry jest stan
**Advantage stream** A(s, a) — ocenia o ile akcja jest lepsza od średniej

### Połączenie

```
Q(s, a) = V(s) + (A(s, a) - mean(A(s, ·)))
```

### Dlaczego odejmujemy mean(A)?

Bez odejmowania średniej, Value stream nie byłby jednoznacznie określony — można by dodać stałą do V i odjąć od A. Odejmowanie mean(A) identyfikuje model.

## Target Network

### Problem

Gdy cel (target) ucieka wraz z aktualizowanymi wagami, trening jest niestabilny.

### Rozwiązanie

**Target Network** — kopia głównej sieci zamrożona na N kroków.

```python
# Aktualizacja co N kroków
if step % target_update_interval == 0:
    target_network.load_state_dict(main_network.state_dict())
```

### Soft update (opcjonalny)

```python
for target_param, main_param in zip(target_network.parameters(), main_network.parameters()):
    target_param.data.copy_(τ * main_param.data + (1 - τ) * target_param.data)
```

Gdzie `τ` to mała stała (np. 0.001).

## Gamma — dyskontowanie przyszłych nagród

### Opis

Gamma określa jak bardzo sieć ceni **przyszłe nagrody** względem **natychmiastowych**.

### Wzór

```
G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... + γⁿ·r_{t+n}
```

### Znaczenie wartości gamma

| Gamma | Zachowanie | Zastosowanie |
|---|---|---|
| 0.9 | Krótkowzroczna | Scalping, szybkie zyski |
| 0.99 | Zbalansowana | Swing trading |
| 0.999 | Cierpliwa | Trading pozycyjny |

### Wpływ na trading

Przy `γ=0.999` nagroda za 100 kroków jest warta `0.999^100 ≈ 0.905` — sieć jest cierpliwa i gra długoterminowo.

## Monte Carlo Returns

### Opis

Po zakończeniu epizodu liczysz **zdyskontowany return** dla każdego kroku idąc od tyłu.

### Algorytm

```python
G = 0
for t in reversed(range(len(episode))):
    G = rewards[t] + gamma * G
    episode[t].return_G = G
```

### Dlaczego Monte Carlo?

Daje **dokładniejsze oszacowanie wartości długoterminowych** — sieć widzi rzeczywisty wynik całej sekwencji decyzji, a nie tylko natychmiastową nagrodę. Szczególnie ważne w tradingu gdzie konsekwencje decyzji mogą ujawnić się dopiero po wielu krokach.

## N-step Returns (opcjonalny)

### Opis

Zamiast 1-step TD error, używasz n-step lookahead:

```python
target = r_t + γ·r_{t+1} + ... + γ^{n-1}·r_{t+n-1} + γ^n × max(Q_target(s_{t+n}))
```

### Korzyści

- Lepsze oszacowanie wartości
- Szybsza konwergencja
- Mniejsza wariancja niż czyste Monte Carlo

## Epsilon-Greedy Eksploracja

### Opis

Sieć wybiera akcję:
- Z prawdopodobieństwem `ε` — losowa akcja (eksploracja)
- Z prawdopodobieństwem `1-ε` — najlepsza akcja wg modelu (eksploatacja)

### Decay

```python
epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * (steps / decay_steps))
```

### Parametry

| Parametr | Wartość | Config field |
|---|---|---|
| Epsilon start | 1.0 | `[training].epsilon_start` |
| Epsilon end | 0.05 | `[training].epsilon_end` |
| Decay fraction | 30% kroków | `[training].epsilon_decay_fraction` |
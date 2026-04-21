# System diagnostyczny

## Przegląd

Moduł `python/diagnostics/` zawiera narzędzia do monitorowania i zarządzania treningiem w czasie rzeczywistym. Wszystkie komponenty są opcjonalne i konfigurowane przez `config.toml`.

## Komponenty

### AlertSystem

Wysyła alerty przez Telegram przy krytycznych zdarzeniach treningowych.

```toml
[alerts]
enabled             = true
telegram_token      = "..."
telegram_chat_id    = "..."
cooldown_sec        = 300     # min. czas między kolejnymi alertami

grad_explode_threshold    = 5.0   # alert gdy grad_norm > próg
action_collapse_threshold = 0.95  # alert gdy jedna akcja > 95% wyborów
loss_plateau_steps        = 2000  # alert gdy loss nie spada przez N kroków
```

#### Typy alertów

| Alert | Wyzwalacz |
|---|---|
| `TRAINING_STARTED` | Pierwszy krok treningu po starcie |
| `NAN_DETECTED` | NaN w loss lub Q-values (natychmiastowy) |
| `GRADIENT_EXPLODE` | `grad_norm > grad_explode_threshold` |
| `ACTION_COLLAPSE` | Jedna akcja > `action_collapse_threshold` wyborów |
| `LOSS_PLATEAU` | Brak poprawy loss przez `loss_plateau_steps` kroków |
| `ADVANTAGE_DEAD` | `advantage_std < 0.001` przez > 500 kroków |

Każdy typ alertu ma osobny cooldown — nie zaśmieca chatu przy ciągłym problemie.

---

### TelegramCommands

Bot Telegram do sterowania treningiem w czasie rzeczywistym. Używa tego samego tokenu co `AlertSystem`.

#### Dostępne komendy

| Komenda | Opis |
|---|---|
| `/status` | Szybki podgląd: krok, sps, epsilon, loss, bufor |
| `/health` | Natychmiastowy health check sieci |
| `/pause` | Wstrzymaj pętlę treningową |
| `/resume` | Wznów pętlę treningową |
| `/checkpoint` | Wymusz zapis checkpointu |
| `/buffer` | Statystyki replay bufora |
| `/actors` | Status aktorów (epsilon, trades, PnL) |
| `/ereset [target] [steps]` | Reset eksploracji (np. `/ereset 0.5 10000`) |
| `/raport` | Wymusz natychmiastowy raport treningowy |
| `/set sekcja.klucz wartość` | Zmień wartość w configu bez restartu |
| `/get sekcja[.klucz]` | Pobierz wartość z configu |
| `/stop` | Graceful shutdown systemu |
| `/restart` | Reset modelu, bufora i checkpointów (nowe szkolenie) |
| `/delete [confirm]` | Usuwa checkpointy i model ONNX |

Bot używa long-pollingu co 2 sekundy.

---

### HealthRunner

Wykonuje health check sieci co godzinę (gdy bufor jest wystarczająco zapełniony).

#### Co sprawdza

- **Q-spread**: `max(Q) - min(Q)` — czy sieć różnicuje akcje
- **Action diversity**: entropia rozkładu akcji z 100 próbek
- **Q-sign consistency**: % kroków gdzie Q[najlepsza_akcja] > 0
- Porównanie z poprzednim checkpointem

#### Wyniki

Logowane do TensorBoard (prefix `Health/`) i do `python/diagnostics/health_checks.jsonl`.

```python
# Wywołanie
health_runner.maybe_run()   # zwraca dict metryk lub None
```

---

### BaselineComparator

Jednorazowo (przy starcie treningu) oblicza metryki zdrowia na **losowej sieci** (random weights) — tworzy punkt odniesienia.

```
Jeśli trained_sharpe < random_sharpe → sieć nie poprawia się ponad losowość
```

Wyniki zapisane do `python/diagnostics/baseline.json` i załadowane przy restarcie.

---

### MetricLogger

Zapis metryk treningowych do pliku JSONL (JSON Lines) dla późniejszej analizy.

```python
# python/diagnostics/metrics.jsonl
{"step": 1000, "loss": 0.023, "epsilon": 0.85, "buffer_size": 15000, ...}
{"step": 2000, "loss": 0.019, "epsilon": 0.82, ...}
```

```python
metric_logger.log(step, {'loss': loss, 'epsilon': epsilon, ...})
```

Buforuje wpisy i zapisuje je co N wpisów (`flush_every=50`) — minimalizuje I/O.

---

### AttentionMonitor

Monitoruje wzorce attention w Transformer Encoder — pomaga diagnostykować czy sieć uczy się sensownych zależności między timeframe'ami.

```python
# Rejestruje hooki tylko w "oknie debug" — usuwa po zakończeniu
# (hooki robią .cpu() na każdy forward pass, co spowalnia GPU)
attention_monitor.enable()    # włącz hooki
# ... kilka kroków ...
attention_monitor.disable()   # wyłącz hooki i zapisz wyniki
```

Wyniki logowane do TensorBoard jako `Attention/` images.

---

### TrainingReport

Generuje czytelny raport treningowy co N tysięcy update'ów.

```toml
[report]
enabled        = true
mode           = "full"   # "full" | "short"
every_n_updates = 5       # raport co 5000 update'ów
min_interval_sec = 0      # min. przerwa między raportami
```

Raport zawiera:
- Aktualne metryki: loss, epsilon, LR, grad_norm, buffer fill
- Metryki per Actor: win_rate, profit_factor, transactions, PnL
- Porównanie z baseline
- Ostrzeżenia (dead neurons, gradient issues)

Wysyłany przez Telegram gdy skonfigurowany.

---

### BacktestRunner

Uruchamia backtesty OOS bezpośrednio z kodu Learnera (bez restartu).

```python
from diagnostics.backtest_runner import BacktestRunner

runner = BacktestRunner(config, trainer.main_network)
results = runner.run()   # zwraca dict: sharpe, pnl, win_rate, ...
```

Używany przy automatycznej ewaluacji co `evaluation_interval` kroków.

## Konfiguracja

```toml
[alerts]
enabled          = true
telegram_token   = "8692060831:AAFkyj2rRsAnv5AKwzn_..."
telegram_chat_id = "8624710090"
cooldown_sec     = 300

grad_explode_threshold    = 5.0
action_collapse_threshold = 0.95
loss_plateau_steps        = 2000

[report]
enabled         = true
mode            = "full"
every_n_updates = 5
min_interval_sec = 0
```

## Pliki runtime (nie commitować)

```
python/diagnostics/
├── metrics.jsonl      ← log metryk treningowych (JSONL)
├── health_checks.jsonl ← wyniki health checkups
└── baseline.json      ← baseline losowej sieci (generowany raz)
```

Pliki są wykluczone przez `.gitignore`.

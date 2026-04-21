# Backtesting

## Przegląd

Backtesting to tryb **ewaluacji wytrenowanego modelu** na danych historycznych. W przeciwieństwie do treningu — model nie jest aktualizowany, tylko oceniany sekwencyjnie na wydzielonym przedziale czasowym.

## Różnice między trybem treningowym a backtestingiem

| Aspekt | Trening | Backtesting |
|---|---|---|
| Dane | Losowe starty (historia - validation_days) | Sekwencyjnie od początku do końca |
| Model | Aktualizowany co krok | Zamrożony — tylko inference |
| Replay Buffer | Aktywny | Nieużywany |
| Nagrody | Do bufora i treningu | Tylko do metryk |
| Epsilon | Maleje w czasie | 0 — zawsze najlepsza akcja |
| Dane OOS | Niedostępne | Ostatnie `validation_days=30` dni |

## Konfiguracja

```toml
[backtesting]
enabled = false              # true = tryb backtesting zamiast treningu
model_path = "python/checkpoints/best_model.pt"
start_date = "2023-01-01"
end_date = "2023-12-31"
results_dir = "results"
```

### Pary do backtestingu

Backtesting używa tych samych par co trening — z sekcji `[[actors]]` w `config.toml`. Można tymczasowo zakomentować pary żeby testować pojedyncze symbole.

## Uruchomienie

```bash
# Backtesting przez skrypt
cd scripts
python evaluate.py --config ../config.toml

# Lub przez run.bat z flagą
run.bat --mode backtest
```

Backtesting jest **oddzielnym trybem** — nie startuje Actorów, Learnera ani Monitoring Service. Uruchamia tylko `evaluate.py`.

## Proces backtestingu

### 1. Załaduj model

```python
model = load_model(config['backtesting']['model_path'])
model.eval()  # tryb inference, bez gradientów
```

### 2. Pobierz dane OOS

```python
# 20% najnowszych danych — nigdy nie widziane przez model podczas treningu
data = load_historical_data(symbol, config)
split_idx = int(len(data) * 0.8)
oos_data = data[split_idx:]
```

### 3. Sekwencyjne przejście przez dane

```python
for step in range(len(oos_data)):
    state = build_state(oos_data, step)
    action_mask = get_action_mask(position)

    with torch.no_grad():
        q_values = model(state, action_mask)
        action = q_values.argmax().item()

    position, reward = execute_action(action, oos_data[step])
    trades.append(record_trade(position, reward))
```

### 4. Oblicz metryki

```python
results = {
    'sharpe_ratio': calculate_sharpe(trades),
    'max_drawdown': calculate_max_drawdown(trades),
    'win_rate': win_rate(trades),
    'profit_factor': profit_factor(trades),
    'avg_win': avg_win(trades),
    'avg_loss': avg_loss(trades),
    'max_consecutive_losses': max_consecutive_losses(trades),
    'total_trades': len(trades),
    'total_pnl': sum(t.pnl for t in trades),
    'period': {'start': start_date, 'end': end_date},
}
```

### 5. Zapisz wyniki

```python
import json
from datetime import datetime

output_path = f"{config['backtesting']['results_dir']}/backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

logger.info(f"Wyniki zapisane do: {output_path}")
```

## Format wyników

```json
{
  "period": {
    "start": "2023-01-01",
    "end": "2023-12-31"
  },
  "model": "checkpoints/best_model.pt",
  "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
  "metrics": {
    "sharpe_ratio": 1.43,
    "max_drawdown": -0.12,
    "win_rate": 0.54,
    "profit_factor": 1.72,
    "avg_win": 0.023,
    "avg_loss": -0.014,
    "max_consecutive_losses": 5,
    "total_trades": 847,
    "total_pnl": 0.34
  },
  "per_symbol": {
    "BTCUSDT": { "sharpe": 1.61, "trades": 312, "pnl": 0.18 },
    "ETHUSDT": { "sharpe": 1.29, "trades": 287, "pnl": 0.09 },
    "SOLUSDT": { "sharpe": 1.38, "trades": 248, "pnl": 0.07 }
  }
}
```

## Struktura results/

```
results/
├── backtest_20240115_143200.json
├── backtest_20240118_091500.json
└── backtest_20240120_164322.json
```

## Interpretacja wyników

### Co oznacza dobry backtest?

| Metryka | Minimum | Dobry | Bardzo dobry |
|---|---|---|---|
| Sharpe Ratio | > 0.5 | > 1.0 | > 2.0 |
| Max Drawdown | < -30% | < -15% | < -10% |
| Win Rate | > 40% | > 50% | > 60% |
| Profit Factor | > 1.0 | > 1.5 | > 2.0 |

### Ostrzeżenia

**Overfitting:** Jeśli wyniki OOS są znacznie gorsze niż in-sample, model nauczył się specyfiki danych treningowych zamiast ogólnych wzorców. Rozwiązanie: więcej danych, mniejsza sieć, regularyzacja.

**Lookahead bias:** Upewnij się że backtesting używa wyłącznie danych z przedziału OOS (20% najnowszych) i że synchronizacja czasowa jest poprawna.

**Survivorship bias:** Dane historyczne mogą nie zawierać par które zostały usunięte z giełdy — wyniki mogą być zawyżone.

## Config — pełna sekcja

```toml
[backtesting]
enabled    = false
model_path = "python/checkpoints/best_model.pt"
start_date = "2023-01-01"
end_date   = "2023-12-31"
results_dir = "results"
step_interval = 5     # co ile świec 1m krok backtestowy
max_oos_days  = 30    # maksymalny horyzont OOS
fee = 0.00075         # prowizja używana w backtestingu (= commission_open + close)

[training]
validation_days = 30  # ostatnie N dni zarezerwowane jako OOS (nigdy nie trenujesz na nich)
```

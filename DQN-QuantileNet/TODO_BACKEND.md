# TODO — Backend Implementation Notes

Plik wygenerowany przez dashboard. Opisuje co należy doimplementować po stronie Python/Node,
żeby dashboard działał na prawdziwych danych zamiast mockowych.

---

## 1. Export `strategy_top100.json` — PRIORYTET WYSOKI

### Gdzie: `python/backtest.py`

Po wywołaniu `brute_force_tp_sl_top20()` (lub odpowiedniej funkcji rankującej),
serializuj listę `all_rows` do JSON:

```python
import json, pathlib
from datetime import datetime, timezone

def export_strategy_top100(all_rows: list, run_id: str):
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "test_range": {
            "start": TEST_START.isoformat(),   # datetime obiekty z backtestowego zakresu
            "end":   TEST_END.isoformat(),
        },
        "symbols": SYMBOLS,                    # np. ["BTCUSDT", ...]
        "entry_prob_grid": ENTRY_PROB_GRID,    # np. [0.34, 0.39, 0.46, 0.51, 0.56]
        "combos": []
    }
    for rank, row in enumerate(all_rows[:100], start=1):
        out["combos"].append({
            "rank":         rank,
            "symbol":       row["symbol"],
            "horizon":      row["horizon_h"],
            "threshold":    row["threshold_pct"],
            "direction":    row["direction"],
            "entry_prob":   row["entry_prob"],
            "tp_pct":       row["tp_pct"],
            "sl_pct":       row["sl_pct"],
            "n_trades":     row["n_trades"],
            "p_win":        row["p_win"],
            "avg_win":      row["avg_win"],
            "avg_loss":     row["avg_loss"],
            "total_return": row["total_return"],
            "sharpe":       row["sharpe"],
            "sqn":          row["sqn"],
            "ev":           row["ev"],
            "kelly":        row["kelly"],
            "tp_rate":      row["tp_rate"],
            "sl_rate":      row["sl_rate"],
            "hz_rate":      row["hz_rate"],
            "max_dd":       row["max_dd"],
            "trades": [
                {
                    "symbol":     t["symbol"],
                    "entry_time": int(t["entry_time"].timestamp() * 1000),  # Unix ms
                    "exit_time":  int(t["exit_time"].timestamp() * 1000),
                    "return":     round(t["return_pct"], 4),
                    "exit_type":  t["exit_type"],   # "tp" | "sl" | "hz"
                    "tp_pct":     t["tp_pct"],
                    "sl_pct":     t["sl_pct"],
                }
                for t in row.get("trades", [])
            ]
        })

    path = pathlib.Path(f"python/checkpoints/{run_id}/strategy_top100.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"[export] strategy_top100.json → {path} ({len(out['combos'])} combos)")
```

**Wywołanie w backtest.py:**
```python
export_strategy_top100(all_rows, run_id=RUN_ID)
```

**Oczekiwany plik:** `python/checkpoints/<run_id>/strategy_top100.json`

---

## 2. Export `calibration.json` — już częściowo istnieje

Sprawdź czy `calibration.json` zawiera pola `n` (liczba próbek) w każdym wpisie.
Dashboard oczekuje:

```json
{
  "generated_at": "...",
  "threshold_calibration": [
    {
      "horizon_h": 8,
      "direction": "long",
      "threshold_pct": 1.0,
      "mean_pred_pct": 19.4,
      "actual_pct": 17.6,
      "error_pct": 1.8,
      "n": 17280
    }
  ],
  "quantile_coverage": [
    {
      "horizon_h": 8,
      "direction": "long",
      "quantile": 0.10,
      "expected_pct": 10.0,
      "actual_pct": 7.0,
      "error_pct": -3.0
    }
  ]
}
```

---

## 3. Live prediction writer — PRIORYTET WYSOKI

### Gdzie: nowy plik `python/server/live_writer.py`

Po każdym zamknięciu świecy 15m, uruchom inferencję na najnowszym oknie i zapisz:

```python
import json, pathlib
from datetime import datetime, timezone

def write_live_predictions(model, data_windows: dict, run_id: str):
    """
    data_windows: {symbol: tensor} — najnowsze okna danych wejściowych
    """
    now = datetime.now(timezone.utc)
    out = {
        "generated_at": now.isoformat(),
        "model_checkpoint": f"{run_id}/ckpt_latest.pt",
        "predictions": []
    }

    for symbol, window in data_windows.items():
        with torch.no_grad():
            quantiles, thresholds = model(window.unsqueeze(0))
            # quantiles: [1, n_horizons, n_quantiles, 2]  (last dim: long/short)
            # thresholds: [1, n_horizons, n_thresholds, 2]

        current_price = get_latest_close(symbol)   # z danych rynkowych
        horizons_out = []

        for hi, h in enumerate(HORIZONS_HOURS):
            target_time = now.timestamp() + h * 3600
            horizons_out.append({
                "horizon_h": h,
                "target_time": datetime.fromtimestamp(target_time, tz=timezone.utc).isoformat(),
                "long": {
                    "quantiles": {
                        f"p{int(q*100)}": round(quantiles[0, hi, qi, 0].item() * 100, 2)
                        for qi, q in enumerate(QUANTILE_LEVELS)
                    },
                    "threshold_probs": {
                        f"{t}pct": round(torch.sigmoid(thresholds[0, hi, ti, 0]).item(), 4)
                        for ti, t in enumerate(THRESHOLD_LEVELS)
                    }
                },
                "short": {
                    "quantiles": {
                        f"p{int(q*100)}": round(quantiles[0, hi, qi, 1].item() * 100, 2)
                        for qi, q in enumerate(QUANTILE_LEVELS)
                    },
                    "threshold_probs": {
                        f"{t}pct": round(torch.sigmoid(thresholds[0, hi, ti, 1]).item(), 4)
                        for ti, t in enumerate(THRESHOLD_LEVELS)
                    }
                }
            })

        out["predictions"].append({
            "symbol": symbol,
            "current_price": current_price,
            "current_time": now.isoformat(),
            "horizons": horizons_out,
        })

    path = pathlib.Path("python/live_predictions.json")
    path.write_text(json.dumps(out, indent=2))
    print(f"[live] predictions written → {path}")
```

**Integracja:** wywołuj `write_live_predictions(...)` po każdym `on_candle_close()` w ZMQ serverze.

---

## 4. Dashboard — ładowanie prawdziwych danych zamiast mocków

Gdy pliki JSON będą dostępne, w komponentach React zastąp mock data:

### BacktestSection.jsx
```js
// Zastąp:
const CAL_DATA = genCalibration();
// Na:
const [calData, setCalData] = useState(null);
useEffect(() => {
  fetch('/python/checkpoints/<run_id>/calibration.json')
    .then(r => r.json())
    .then(setCalData);
}, []);
```

### StrategySection.jsx
```js
// Zastąp MOCK_COMBOS na:
useEffect(() => {
  fetch('/python/checkpoints/<run_id>/strategy_top100.json')
    .then(r => r.json())
    .then(d => setCombos(d.combos));
}, []);
```

### LiveSection.jsx
```js
// Zastąp genLivePredictions() na:
const refresh = async () => {
  const d = await fetch('/python/live_predictions.json').then(r => r.json());
  setLiveData(d);
  setLastRefresh(new Date(d.generated_at));
};
```

---

## 5. Serwowanie plików statycznych

Dashboard może czytać pliki JSON bezpośrednio, jeśli jest serwowany z katalogu projektu.
Opcje:

```bash
# Opcja A — Python http.server (dev)
cd DQN-QuantileNet
python -m http.server 8080

# Opcja B — Node.js (jeśli dodasz do node/)
npx serve . -p 8080

# Opcja C — Nginx/Caddy (produkcja)
# Wskaż root na DQN-QuantileNet/
```

Otwórz: `http://localhost:8080/dashboard.html`

---

## 6. Automatyczne odświeżanie (opcjonalne)

Zamiast pollingu, można użyć WebSocket lub Server-Sent Events:

```python
# python/server/sse_server.py
from flask import Flask, Response
import time, json

app = Flask(__name__)

@app.route('/predictions/stream')
def stream():
    def gen():
        while True:
            data = open('live_predictions.json').read()
            yield f"data: {data}\n\n"
            time.sleep(60)
    return Response(gen(), mimetype='text/event-stream')
```

W LiveSection.jsx:
```js
useEffect(() => {
  const es = new EventSource('/predictions/stream');
  es.onmessage = e => setLiveData(JSON.parse(e.data));
  return () => es.close();
}, []);
```

---

## Podsumowanie priorytetów

| # | Co | Gdzie | Priorytet |
|---|---|---|---|
| 1 | `export_strategy_top100()` w backtest.py | python/backtest.py | 🔴 Wysoki |
| 2 | `write_live_predictions()` po każdej świecy | python/server/ | 🔴 Wysoki |
| 3 | Sprawdzić format `calibration.json` | python/backtest.py | 🟡 Średni |
| 4 | Zastąpić mock data na fetch() w komponentach | components/*.jsx | 🟡 Średni |
| 5 | Serwer statyczny | — | 🟢 Niski |
| 6 | SSE zamiast pollingu | python/server/ | 🟢 Niski |

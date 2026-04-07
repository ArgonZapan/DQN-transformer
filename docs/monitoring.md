# Monitoring i Dashboard

## Przegląd

System monitorowania składa się z dwóch komponentów:
1. **Monitoring Service** (Node.js) — agregacja metryk
2. **Dashboard** (Vite + React) — wizualizacja danych

## Architektura monitorowania

```
Actor         ──ZMQ PUSH──► Monitoring Svc
Python/Learner ──ZMQ PUSH──► Monitoring Svc ◄──HTTP REST (co 1 min)── Dashboard
```

### Kanały komunikacji

| Kanał | Protokół | Kierunek |
|---|---|---|
| Actor → Monitoring | ZeroMQ PUSH/PULL | Jednokierunkowy |
| Learner → Monitoring | ZeroMQ PUSH/PULL | Jednokierunkowy |
| Dashboard → Monitoring | HTTP REST (GET) | Przeglądarka → Serwer |

## Monitoring Service

### Opis

Dedykowany serwis Node.js do **agregacji i udostępniania metryk**. Jest pasywny — tylko agreguje i udostępnia dane.

### Zasada działania

1. Każdy moduł push-uje swoje metryki przez ZeroMQ
2. Monitoring Service zbiera i agreguje dane
3. Dashboard odpytuje Monitoring Service co 1 minutę

### Metryki od Actorów

| Metryka | Opis |
|---|---|
| epsilon | Aktualna wartość epsilon |
| transactions | Liczba transakcji |
| episode_pnl | P&L epizodu |
| position_status | Status pozycji (open/closed/none) |
| step_loss | Loss krokowy |

### Metryki od Learnera

| Metryka | Opis |
|---|---|
| bufferSize | Rozmiar replay buffer |
| trainSteps | Liczba kroków treningowych |
| current_loss | Aktualny loss |
| sharpe_in_sample | Sharpe ratio na danych treningowych |
| sharpe_out_of_sample | Sharpe ratio na danych walidacyjnych |

### Implementacja

```javascript
const zmq = require('zeromq');
const sock = new zmq.Pull();

sock.bind('tcp://*:3002').then(() => {
    console.log('Monitoring service listening on port 3002');
    
    const metrics = {};
    
    sock.on('message', async (msg) => {
        const data = JSON.parse(msg.toString());
        
        // Aktualizuj metryki per moduł
        if (!metrics[data.source]) {
            metrics[data.source] = [];
        }
        
        metrics[data.source].push({
            timestamp: Date.now(),
            ...data.metrics
        });
        
        // Przechowuj tylko ostatnie N wpisów
        if (metrics[data.source].length > 1000) {
            metrics[data.source] = metrics[data.source].slice(-1000);
        }
    });
});
```

### HTTP REST Endpoint dla Dashboardu

Monitoring Service wystawia **HTTP REST API** (np. przez Express.js) dla Dashboardu:

```javascript
const express = require('express');
const app = express();
const PORT = 3001;

// GET /api/metrics - zwróć aktualne metryki
app.get('/api/metrics', (req, res) => {
    res.json({
        timestamp: Date.now(),
        actors: metrics['actor'] || [],
        learner: metrics['learner'] || []
    });
});

// GET /api/status - szybki check statusu
app.get('/api/status', (req, res) => {
    const lastActor = metrics['actor']?.slice(-1)[0];
    const lastLearner = metrics['learner']?.slice(-1)[0];
    
    res.json({
        actorConnected: !!lastActor,
        learnerConnected: !!lastLearner,
        lastUpdate: Math.max(
            lastActor?.timestamp || 0,
            lastLearner?.timestamp || 0
        )
    });
});

app.listen(PORT, () => {
    console.log(`Dashboard REST API on http://localhost:${PORT}`);
});
```

> **Dlaczego HTTP a nie ZMQ dla Dashboardu?** Dashboard działa w przeglądarce — `fetch()` jest natywny i nie wymaga dodatkowych bibliotek. ZMQ wymagałoby WebSocket proxy.

## Dashboard (Vite + React)

### Opis

Interfejs użytkownika do **wizualizacji metryk** systemu.

### Zasada działania

- Odpytuje Monitoring Service **co 1 minutę**
- Wyłącznie w trybie **GET** — nie modyfikuje danych
- Odpalany w **dev mode** z poziomu pliku uruchamiającego

### Typowe widoki

#### Loss Curve

Wykres loss w czasie — pokazuje konwergencję modelu.

```jsx
function LossChart({ data }) {
    return (
        <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timestamp" />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey="loss" stroke="#8884d8" />
        </LineChart>
    );
}
```

#### Epsilon Decay

Wykres epsilon w czasie — pokazuje postęp eksploracji.

```jsx
function EpsilonChart({ data }) {
    return (
        <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timestamp" />
            <YAxis domain={[0, 1]} />
            <Tooltip />
            <Line type="monotone" dataKey="epsilon" stroke="#82ca9d" />
        </LineChart>
    );
}
```

#### Equity Curve

Wykres P&L aktorów w czasie — pokazuje wyniki tradingowe.

```jsx
function EquityChart({ data }) {
    return (
        <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timestamp" />
            <YAxis />
            <Tooltip />
            <Legend />
            {actors.map(actor => (
                <Line key={actor} type="monotone" dataKey={`pnl_${actor}`} />
            ))}
        </LineChart>
    );
}
```

#### Replay Buffer Metrics

```jsx
function BufferMetrics({ data }) {
    return (
        <div>
            <h3>Replay Buffer</h3>
            <p>Size: {data.bufferSize.toLocaleString()}</p>
            <p>Capacity: {data.bufferCapacity.toLocaleString()}</p>
            <p>Fill %: {(data.bufferSize / data.bufferCapacity * 100).toFixed(1)}%</p>
            <p>Sampling rate: {data.samplingRate}/s</p>
        </div>
    );
}
```

#### Sharpe Ratio Comparison

```jsx
function SharpeComparison({ data }) {
    return (
        <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="period" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="in_sample" fill="#8884d8" name="In-Sample" />
            <Bar dataKey="out_of_sample" fill="#82ca9d" name="Out-of-Sample" />
        </BarChart>
    );
}
```

### Polling danych

```javascript
function useMetrics() {
    const [metrics, setMetrics] = useState(null);
    
    useEffect(() => {
        const fetchMetrics = async () => {
            const response = await fetch('http://localhost:3001/metrics');
            const data = await response.json();
            setMetrics(data);
        };
        
        // Poll co 60 sekund
        fetchMetrics();
        const interval = setInterval(fetchMetrics, 60000);
        
        return () => clearInterval(interval);
    }, []);
    
    return metrics;
}
```

## Metryki i interpretacja

### Sharpe Ratio

**Definicja:** Zysk podzielony przez zmienność.

```
Sharpe = (mean_return - risk_free_rate) / std_return
```

| Wartość | Interpretacja |
|---|---|
| < 0 | Strata |
| 0-1 | Niska jakość strategii |
| 1-2 | Dobra strategia |
| > 2 | Bardzo dobra strategia |

### Maximum Drawdown

**Definicja:** Największy spadek od szczytu equity curve.

```python
def max_drawdown(equity_curve):
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak
    return drawdown.min()
```

Mówi o **ryzyku straty kapitału**.

### Liczba transakcji per dzień

| Wartość | Interpretacja |
|---|---|
| < 1 | Bardzo konserwatywna strategia |
| 1-5 | Umiarkowana strategia |
| 5-20 | Aktywny trading |
| > 20 | Podejrzenie churning |

### Out-of-sample Sharpe

**Najważniejsza metryka.** 20% najnowszych danych odkładasz przed treningiem i nigdy nie trenujesz na nich.

```python
# Przed treningiem
train_data = data[:80%]
test_data = data[80%:]

# Ewaluacja co 3-4 dni
if step % evaluation_interval == 0:
    oos_sharpe = evaluate(test_data)
```

**Interpretacja:**

| Sytuacja | Diagnoza |
|---|---|
| In-sample wysoki, OOS niski | Overfitting |
| Oba niskie | Underfitting |
| Oba wysokie | Dobrze dopasowany model |

## Config

```toml
[monitoring]
host = "tcp://127.0.0.1"
port = 3001
metrics_push_interval_sec = 5
dashboard_poll_interval_sec = 60
## Metryki tradingowe

Oprócz Sharpe ratio i drawdown, system zbiera podstawowe metryki tradingowe które szybciej pokażą czy sieć uczy się czegoś sensownego.

### Win Rate

Procent zyskownych transakcji.

```python
def win_rate(trades: list) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t.pnl > 0)
    return wins / len(trades)
```

| Wartość | Interpretacja |
|---|---|
| < 40% | Strategia traci więcej niż wygrywa |
| 40-60% | Typowy zakres dla algorytmów tradingowych |
| > 60% | Dobry wynik (sprawdź profit factor) |

### Profit Factor

Suma zysków podzielona przez sumę strat — ile zarabiasz na każdą złotówkę straty.

```python
def profit_factor(trades: list) -> float:
    gains = sum(t.pnl for t in trades if t.pnl > 0)
    losses = abs(sum(t.pnl for t in trades if t.pnl < 0))
    if losses == 0:
        return float('inf')
    return gains / losses
```

| Wartość | Interpretacja |
|---|---|
| < 1.0 | Strategia traci pieniądze |
| 1.0-1.5 | Słaba strategia |
| 1.5-2.0 | Dobra strategia |
| > 2.0 | Bardzo dobra strategia |

### Average Win vs Average Loss

```python
def avg_win_loss(trades: list) -> dict:
    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [t.pnl for t in trades if t.pnl < 0]
    return {
        'avg_win': sum(wins) / len(wins) if wins else 0,
        'avg_loss': sum(losses) / len(losses) if losses else 0,
        'ratio': (sum(wins)/len(wins)) / abs(sum(losses)/len(losses)) if wins and losses else 0
    }
```

Stosunek `avg_win / avg_loss` powyżej 1.5 oznacza że wygrane są znacznie większe od strat.

### Max Consecutive Losses

Maksymalna seria przegranych z rzędu — wskaźnik ryzyka psychologicznego i stabilności strategii.

```python
def max_consecutive_losses(trades: list) -> int:
    max_streak = 0
    current_streak = 0
    for t in trades:
        if t.pnl < 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    return max_streak
```

| Wartość | Interpretacja |
|---|---|
| < 5 | Stabilna strategia |
| 5-10 | Akceptowalne |
| > 10 | Ryzykowna strategia — sprawdź churning |

### Metryki w Monitoring Service

Wszystkie metryki tradingowe są agregowane per Actor i dostępne przez HTTP REST:

```json
{
  "actors": {
    "BTCUSDT": {
      "win_rate": 0.54,
      "profit_factor": 1.72,
      "avg_win": 0.023,
      "avg_loss": -0.014,
      "max_consecutive_losses": 4,
      "total_trades": 342
    }
  }
}
```

### Widok w Dashboardzie

```jsx
function TradingMetrics({ data }) {
    return (
        <div className="metrics-grid">
            <MetricCard label="Win Rate" value={`${(data.win_rate * 100).toFixed(1)}%`} />
            <MetricCard label="Profit Factor" value={data.profit_factor.toFixed(2)} />
            <MetricCard label="Avg Win" value={`${(data.avg_win * 100).toFixed(2)}%`} />
            <MetricCard label="Avg Loss" value={`${(data.avg_loss * 100).toFixed(2)}%`} />
            <MetricCard label="Max Consec. Losses" value={data.max_consecutive_losses} />
            <MetricCard label="Total Trades" value={data.total_trades} />
        </div>
    );
}
```

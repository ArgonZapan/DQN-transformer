# Trading DQN — Distributed Reinforcement Learning dla Rynków Kryptowalut

Architektura Actor-Learner Ape-X z modelem Conv1D + Transformer do tradingu algorytmicznego na parach kryptowalut (BTC/ETH/SOL).

### Stack

| Moduł | Technologia |
|---|---|
| Learner | Python + PyTorch + FastAPI |
| Actor (N instancji) | Node.js |
| Monitoring Service | Node.js |
| Dashboard | Vite + React (dev mode) |
| Konfiguracja | TOML (jeden plik, zero hardcoded zmiennych) |
| Uruchomienie | Jeden plik .bat (Windows) |

### Stack jakości

- TDD — 100% coverage per moduł
- Folder `debug/` ze smoke-testami do szybkiej weryfikacji integracji
- Żadnych hardcoded zmiennych w kodzie — wszystko przez konfig TOML

---

## Przegląd architektury

```
┌─────────────────────────────────────────────────────────┐
│                        Node.js                          │
│                                                         │
│  Actor 1 ──┐                                            │
│  Actor 2 ──┼──► actorManager ──► POST /step ──┐        │
│  Actor N ──┘         ▲                         │        │
│                        │                         ▼        │
│              ◄── nextAction ◄────── POST /predict        │
│                                                         │
│  Każdy moduł ──► POST metrics ──► Monitoring Svc         │
└─────────────────────────────────────────────────────────┘
     │                                    │
     ▼                                    ▼
┌─────────────┐              ┌─────────────────────────┐
│   Python    │              │   Node.js Monitoring     │
│             │              │   (agregacja metryk)      │
│  FastAPI ──►│              ├─────────────────────────┤
│  Replay Buf │              │   Dashboard (Vite+React) │
│  Trainer ──►│              │   pull co 1 min, GET     │
│  Model      │              │   tylko wyswietla        │
└─────────────┘              └─────────────────────────┘
```

**Node.js (Actor)** odpowiada wyłącznie za symulację środowiska rynkowego — pobieranie danych, budowanie stanów, obliczanie nagród, zarządzanie epizodami.

**Python (Learner)** odpowiada wyłącznie za wszystko związane z ML — przechowywanie doświadczeń w replay buforze, trening modelu, predykcje.

---

## Dlaczego taki podział

Node.js ma dojrzałe biblioteki do obsługi Binance API i jest naturalnym środowiskiem dla logiki tradingowej. Python z PyTorch ma natywną obsługę CUDA i jest standardem dla RL — większość papierów badawczych i bibliotek (Stable Baselines, RLlib, CleanRL) jest w PyTorch.

Połączenie obu przez HTTP daje czystą separację odpowiedzialności. Każdą część można testować niezależnie. Model można wymienić bez dotykania logiki tradingowej i odwrotnie.

Jest to implementacja wzorca **Actor-Learner** gdzie Node.js to Actor działający w środowisku, a Python to Learner uczący się z doświadczeń i dostarczający politykę.

---

## Model sieci neuronowej

### Architektura ogólna

```
5 wejść (timeframe'y)
    │
    ├── Conv1D Block (1m)   ─┐
    ├── Conv1D Block (15m)  ─┤
    ├── Conv1D Block (1h)   ─┼──► Konkatenacja (640)
    ├── Conv1D Block (1d)   ─┤         │
    └── Conv1D Block (1w)   ─┘         │
                                  Projekcja → Reshape (5, 128)
                                       │
                              3× Transformer Encoder Block
                              MultiHeadAttention(8 heads, key_dim=64)
                              LayerNorm + Residual
                              FeedForward(512) + LayerNorm
                                       │
                              GlobalAveragePooling1D
                                       │
                              Dense(512) → Dense(256)
                                       │
                          ┌────────────┴────────────┐
                     Value(1)               Advantage(3)
                          └────────────┬────────────┘
                               Q = V + (A - mean(A))
```

### Conv1D Block

Każdy timeframe jest przetwarzany przez osobny blok konwolucyjny. Conv1D przesuwa małe okienko po sekwencji świec i wykrywa lokalne wzorce — formacje cenowe, skoki wolumenu, lokalne momentum. Każda warstwa Conv1D ma BatchNormalization dla stabilności treningu.

### Transformer Encoder

Po konkatenacji wektorów z Conv1D, dane są reshapowane do sekwencji 5 tokenów (jeden per timeframe) i przetwarzane przez bloki Transformer Encoder.

Mechanizm **multi-head attention** dla każdego tokenu (timeframe'u) oblicza jak bardzo powinien patrzeć na każdy inny token. Sieć sama odkrywa zależności między timeframe'ami — np. że sygnał z 1h potwierdza sygnał z 15m.

Każdy blok zawiera:
- Multi-Head Self-Attention (8 głowic, key_dim=64)
- Add & LayerNorm (residual connection)
- Feed-Forward Dense(512)
- Add & LayerNorm

Residual connections są kluczowe — pozwalają gradientom płynąć przez głęboką sieć bez zanikania.

### Dueling Architecture

Trunk sieci rozdziela się na dwa strumienie:

**Value stream** — ocenia jak dobra jest ogólnie sytuacja rynkowa niezależnie od akcji. "Rynek jest teraz niebezpieczny" to informacja niezależna od tego czy kupujesz czy sprzedajesz.

**Advantage stream** — ocenia o ile lepsza jest każda akcja od średniej w tej sytuacji.

Końcowe Q-values: `Q = V + (A - mean(A))`

Ten podział przyspiesza uczenie bo sieć może nauczyć się wartości stanu bez konieczności testowania każdej akcji osobno.

### Parametry modelu

| Parametr | Wartość |
|---|---|
| Cechy na świecę | 8 |
| Liczba akcji | 3 (kup / sprzedaj / czekaj) |
| Bloki Transformer | 3 |
| Głowice attention | 8 |
| Key dimension | 64 |
| Feed-Forward dim | 512 |
| Dropout | 0.1 |
| Łączne parametry | ~5-8M |

---

## Algorytm DQN i Rainbow

Projekt implementuje **Rainbow-lite DQN** — trzy najważniejsze ulepszenia bazowego DQN:

### Double DQN

Bazowy DQN używa tej samej sieci do wyboru akcji i do oceny jej wartości, co prowadzi do przeszacowywania Q-values. Double DQN rozdziela te role:

```
akcja = argmax( główna_sieć(następny_stan) )
wartość = target_sieć(następny_stan)[akcja]
target = nagroda + gamma × wartość
```

### Prioritized Experience Replay

Nie wszystkie doświadczenia są równie wartościowe. Doświadczenia gdzie sieć myliła się bardziej (duży TD error) są samplowane częściej. Priorytet doświadczenia to `|TD error| + ε`.

### Dueling Networks

Opisane wyżej w sekcji architektury.

### Wzór Bellmana

```
TD error = target - Q(stan, akcja)
target   = nagroda + gamma × max(Q_target(następny_stan))
```

Przy zakończeniu epizodu (done=True) drugi składnik odpada:
```
target = nagroda
```

### Target Network

Kopia głównej sieci zamrożona na N kroków. Używana tylko do liczenia targetu — zapobiega niestabilności gdy cel ucieka wraz z aktualizowanymi wagami.

```
co 1000 kroków: target_network.load_state_dict(main_network.state_dict())
```

### Gamma — dyskontowanie przyszłych nagród

Gamma określa jak bardzo sieć ceni przyszłe nagrody względem natychmiastowych.

```
G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... + γⁿ·r_{t+n}
```

Przy `γ=0.999` nagroda za 100 kroków jest warta `0.999^100 ≈ 0.905` — sieć jest cierpliwa i gra długoterminowo. Przy niskiej gammie sieć jest krótkowzroczna i preferuje szybkie małe zyski.

### Monte Carlo Returns

Po zakończeniu epizodu liczysz zdyskontowany return dla każdego kroku idąc od tyłu:

```python
G = 0
for t in reversed(range(len(episode))):
    G = rewards[t] + gamma * G
    episode[t].return_G = G
```

Dopiero po obliczeniu G dla wszystkich kroków epizodu wrzucasz je do głównego replay bufora.

---

## Replay Buffer

### Pre-alokowany bufor z Pinned Memory

Zamiast listy obiektów Python, bufor to jeden duży pre-alokowany blok pamięci per pole. Przy 2M pojemności zajmuje około 1GB RAM.

```python
self.states_1m = torch.zeros(capacity, 15, 8, dtype=torch.float32).pin_memory()
```

**Pinned memory** to specjalny rodzaj RAM który nie może być swapowany na dysk — umożliwia szybszy asynchroniczny transfer CPU→GPU przez DMA.

```python
states = self.states_1m[idx].to('cuda', non_blocking=True)
```

`non_blocking=True` oznacza że transfer odbywa się asynchronicznie — CPU może robić inne rzeczy podczas gdy dane lecą na GPU.

### Faza wstępna bez inference

Przez pierwsze 200-500k kroków gdy epsilon jest wysoki (sieć i tak losuje akcje), Node.js generuje losowe akcje bez pytania Pythona. Bufor zapełnia się znacznie szybciej bez narzutu HTTP i inference.

---

## Nagradzanie

### Zasady

Nagroda jest dawana tylko przy **zamknięciu pozycji** (realized P&L). Podczas trzymania pozycji nagroda wynosi 0. To zapobiega uczeniu się na unrealized huśtawce cenowej.

```
nagroda = (cena_zamknięcia - cena_otwarcia) / cena_otwarcia
        - prowizja_otwarcia
        - prowizja_zamknięcia
```

### Prowizja

Prowizja musi być odjęta od nagrody — bez niej sieć nie widzi kosztu transakcji i uczy się overtradingu (churning). Na Binance to 0.1% na każdą stronę, 0.2% per round trip.

### Kara za każdą transakcję

Mała dodatkowa kara niezależna od prowizji zniechęca do nadmiernego handlu:

```
nagroda -= 0.001  # przy każdym otwarciu pozycji
```

### Skala nagród i clipping

```
nagrody pośrednie (bicie pionka, etc.): max ±0.1
nagroda końcowa (wygrana/przegrana):    ±1.0
clip nagrody:                           [-1, 1]
```

Nagrody pośrednie muszą być znacznie mniejsze niż końcowa — inaczej sieć optymalizuje pod nagrody pośrednie kosztem wyniku końcowego. Ostateczna nagroda jest always clipped do zakresu [-1, 1] dla stabilności gradientów.

### Typowe problemy

**Churning** — sieć ciągle kupuje i sprzedaje. Przyczyna: brak prowizji w nagrodzie lub nagroda per krok zamiast realized only.

**Nigdy nie handluje** — sieć nauczyła się zawsze czekać. Przyczyna: zbyt duże kary za transakcje.

**Overfit do danych treningowych** — na danych treningowych działa świetnie, na nowych losowo. Przyczyna: trenowanie na jednym krótkim wykresie z tym samym startem epizodu.

---

## Wejście do sieci

### Multi-scale temporal representation

Różne timeframe'y pokrywają tę samą historię z różną rozdzielczością. Im bliżej teraźniejszości, tym więcej szczegółów:

| Timeframe | Świece | Pokryty okres |
|---|---|---|
| 1m  | 15 | 15 minut |
| 15m | 15 | 3.75 godziny |
| 1h  | 20 | 20 godzin |
| 1d  | 30 | 30 dni |
| 1w  | 54 | ~1 rok |

Żadna świeca nie pokrywa innej — czysta logarytmiczna skala rozdzielczości.

### 8 cech na świecę

```python
features = [
    (close - mean) / std,           # znormalizowana cena close
    (high - low) / close,           # relative range świecy
    (close - open) / close,         # kierunek świecy
    volume / mean_volume,           # znormalizowany wolumen
    rsi / 100,                      # RSI w zakresie [0, 1]
    macd_normalized,                # MACD znormalizowany
    (close - prev_close) / prev_close,  # zmiana % względem poprzedniej
    float(close > sma20),           # czy powyżej SMA20 (0 lub 1)
]
```

Nigdy surowe ceny — zawsze zmiany procentowe lub znormalizowane wartości. Sieć musi widzieć te same wzorce bez względu na to czy BTC kosztuje 1000 czy 100000.

---

## Środowisko i Actorzy

### Konfiguracja aktorów

Actorzy są zdefiniowani jako lista obiektów w pliku TOML. Każdy wpis zawiera symbol, exchange i indywidualne parametry. Dzięki temu dodanie nowej pary to jeden wpis w konfigu, zero zmian w kodzie.

```toml
[[actors]]
symbol = "BTCUSDT"
exchange = "binance"
leverage = 1

[[actors]]
symbol = "ETHUSDT"
exchange = "binance"
leverage = 1

[[actors]]
symbol = "SOLUSDT"
exchange = "binance"
leverage = 1
```

### Globalne timeframe'y

Timeframe'y są zdefiniowane globalnie — jeden zestaw dla wszystkich aktorów, jeden model. Wszystkie Actory używają tej samej struktury stanu i tego samego schematu nagród.

```toml
[timeframes]
candles_1m  = 15
candles_15m = 15
candles_1h  = 20
candles_1d  = 30
candles_1w  = 54
```

### Dlaczego wielu aktorów

Każdy Actor symuluje środowisko dla jednej pary. Wszystkie wysyłają doświadczenia do jednego Python Learnera.

Korzyści:
- Sieć uczy się ogólnych wzorców niezależnych od pary
- Replay buffer zawiera mix z różnych reżimów rynkowych
- Naturalne rozwiązanie problemu korelacji doświadczeń
- Skalowalne — dodaj kolejnego aktora przez wpis w TOML

### Batching requestów

`actorManager.js` zbiera requesty od wszystkich Actorów przez kilka milisekund i wysyła jeden zbiorczy batch do Pythona. Jedno wywołanie modelu obsługuje wielu Actorów jednocześnie.

### Faza epsilon-greedy

```
epsilon start: 1.0    (pełna eksploracja)
epsilon end:   0.05   (głównie eksploatacja)
decay: liniowy przez pierwsze 30% kroków
```

Przy wysokim epsilon Actor wybiera losową akcję bez pytania modelu — znacząco przyspiesza zapełnienie bufora w początkowej fazie.

---

## Komunikacja

### Endpointy FastAPI

```
POST /step
  body: {
    state: { s1m, s15m, s1h, s1d, s1w },  # stany per timeframe
    action: int,                            # akcja którą Actor wykonał
    reward: float,                          # nagroda
    nextState: { s1m, s15m, s1h, s1d, s1w },
    done: bool                              # czy koniec epizodu
  }
  response: { nextAction: int }

POST /predict
  body: { state: { s1m, s15m, s1h, s1d, s1w } }
  response: { action: int, qValues: [float, float, float] }

GET /status
  response: { bufferSize, trainSteps, epsilon, lastLoss }
```

### Format danych

Stany są wysyłane jako zagnieżdżone tablice floatów w JSON. Przy batch 3 Actorów payload jednego `/step` to kilkadziesiąt KB — akceptowalne przy localhost.

Dla większej liczby Actorów warto rozważyć MessagePack zamiast JSON — kilkukrotnie mniejszy payload.

---

## Monitoring

### Monitoring Service (Node.js)

Każdy moduł systemu (Actorzy, Python Learner) pushuje swoje metryki przez HTTP POST do dedykowanego Monitoring Service. Moduł jest pasywny — tylko agreguje i udostępnia dane.

```
Actor         ──POST──► Monitoring Svc
Python/Learner ──POST──► Monitoring Svc ◄──GET (co 1 min)── Dashboard
```

Metryki pushowane na bieżąco:
- **Actor:** epsilon, liczba transakcji, P&L epizodu, status pozycji, loss krokowy
- **Learner:** bufferSize, trainSteps, aktualny loss, Sharpe in-sample, Sharpe out-of-sample

### Dashboard (Vite + React)

Dashboard odpytuje Monitoring Service co 1 minutę wyłącznie w trybie GET. Nie modyfikuje danych, tylko je wyświetla. Odpalany w dev mode z poziomu pliku uruchamiającego.

Typowe widoki:
- Loss curve w czasie
- Epsilon decay
- Equity curve actorów
- Metryki replay buffer (size, sampling rate)
- Sharpe ratio in-sample vs out-of-sample

---

## Plan treningu fazowego

### Faza 1 — Weryfikacja (dni 1-3)

Mała sieć, jedna para (BTCUSDT), jeden timeframe (15m), uproszczone wejście. Cel: potwierdzenie że loss maleje, epsilon spada poprawnie, sieć nie wpada w churning.

**Nie przechodź dalej dopóki baseline nie działa.**

### Faza 2 — Multi-scale (dni 4-10)

Dodanie wszystkich 5 timeframe'ów. Trzy Actory (BTC, ETH, SOL). Weryfikacja out-of-sample.

### Faza 3 — Transformer (dni 11-20)

Zastąpienie flat Dense przez bloki Transformer Encoder po konkatenacji. Monitoring czy Sharpe poprawia się względem Fazy 2.

### Faza 4 — Pełna architektura (dni 21-30)

Multi-step returns (n=5), Prioritized Replay, pełne hyperparameter tuning. Population Based Training opcjonalnie.

### Parametry per faza

| | Faza 1 | Faza 2 | Faza 3 | Faza 4 |
|---|---|---|---|---|
| lr | 0.0003 | 0.0001 | 0.0001 | 0.00005 |
| batch | 64 | 128 | 256 | 512 |
| buffer | 50k | 200k | 500k | 2M |
| gamma | 0.99 | 0.99 | 0.999 | 0.999 |
| n-step | 1 | 1 | 3 | 5 |
| target update | 500 | 500 | 1000 | 1000 |

---

## Metryki i ewaluacja

Loss sam w sobie nie mówi czy strategia jest dobra. Ważniejsze metryki:

**Sharpe Ratio** — zysk podzielony przez zmienność. Wartość > 1.0 to dobry wynik, > 2.0 to bardzo dobry.

**Maximum Drawdown** — największy spadek od szczytu equity curve. Mówi o ryzyku straty kapitału.

**Średnia liczba transakcji per dzień** — zbyt wysoka oznacza churning, zbyt niska że sieć prawie nie handluje.

**Out-of-sample Sharpe** — najważniejsza metryka. 20% najnowszych danych odkładasz przed treningiem i nigdy nie trenujesz na nich. Ewaluacja co 3-4 dni.

Jeśli in-sample Sharpe jest wysoki a out-of-sample niski — model się overfittuje. Cofnij się do prostszej architektury lub dodaj regularyzację.

---

## Struktura plików

```
trading-dqn/
│
├── config.toml                    ← jedna konfiguracja dla całego systemu
├── run.bat                        ← jedno uruchomienie (Windows)
│
├── python/
│   ├── main.py
│   ├── config.py                  ← loader config.toml (sekcja [learner])
│   ├── model/
│   │   ├── network.py
│   │   ├── regime_detector.py
│   │   └── noisy_linear.py
│   ├── training/
│   │   ├── replay_buffer.py
│   │   ├── trainer.py
│   │   └── prioritized_buffer.py
│   ├── server/
│   │   ├── app.py
│   │   └── schemas.py
│   ├── utils/
│   │   ├── normalizer.py
│   │   └── metrics.py
│   ├── monitoring/
│   │   └── monitor_client.py    ← HTTP POST metryk do Monitoring Svc
│   └── checkpoints/
│
├── node/
│   ├── index.js
│   ├── config.js                  ← loader config.toml (sekcje [actors], [timeframes])
│   ├── env/
│   │   ├── tradingEnv.js
│   │   ├── reward.js
│   │   ├── state.js
│   │   └── episode.js
│   ├── data/
│   │   ├── binance.js
│   │   ├── indicators.js
│   │   ├── normalizer.js
│   │   └── cache/
│   ├── actors/
│   │   ├── actor.js
│   │   └── actorManager.js
│   ├── client/
│   │   └── pythonClient.js
│   └── monitoring/
│       └── monitor_client.js      ← HTTP POST metryk do Monitoring Svc
│
├── monitoring/
│   ├── server.js                  ← Monitoring Service (Node.js)
│   └── config.js                  ← loader config.toml (sekcja [monitoring])
│
├── dashboard/
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx
│       └── components/
│
├── shared/
│   └── stateSchema.js
│
├── scripts/
│   ├── download_data.js
│   ├── evaluate.py
│   └── visualize.py
│
├── tests/
│   ├── python/
│   └── node/
│
├── debug/                         ← smoke-testy integracyjne
│   └── ...
│
├── docker/
│   ├── Dockerfile.python
│   ├── Dockerfile.node
│   └── docker-compose.yml
│
└── Readme.md
```

---

## Uruchomienie

### Konfiguracja

Wszystkie parametry w jednym pliku `config.toml`. Zero hardcoded zmiennych w kodzie. Każdy moduł ładuje swoją sekcję.

```toml
[learner]
host = "localhost"
port = 8000
device = "cuda"

[[actors]]
symbol = "BTCUSDT"
exchange = "binance"
leverage = 1

[[actors]]
symbol = "ETHUSDT"
exchange = "binance"
leverage = 1

[[actors]]
symbol = "SOLUSDT"
exchange = "binance"
leverage = 1

[timeframes]
candles_1m  = 15
candles_15m = 15
candles_1h  = 20
candles_1d  = 30
candles_1w  = 54

[monitoring]
host = "localhost"
port = 3001
metrics_push_interval_sec = 5
dashboard_poll_interval_sec = 60

[training]
gamma = 0.999
lr = 0.0001
batch_size = 256
buffer_capacity = 500000
target_update_interval = 1000
epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay_fraction = 0.3

[api]
binance_key = "..."
simulation_mode = true
```

### Uruchomienie

```bash
# Jednym poleceniem — plik .bat startuje wszystkie procesy:
run.bat
```

Plik uruchamia równolegle:
1. **Python Learner** — FastAPI, trening, predykcje
2. **Monitoring Service** — Node.js, agregacja metryk
3. **Actorzy** — Node.js (N instancji wg config.toml)
4. **Dashboard** — Vite + React w dev mode

```bash
# Ręczne uruchamianie (opcjonalnie):

# Terminal 1 — Python Learner
cd python && python main.py

# Terminal 2 — Monitoring Service
cd monitoring && node server.js

# Terminal 3 — Actorzy (zarządza N instancjami przez actorManager)
cd node && node index.js

# Terminal 4 — Dashboard
cd dashboard && npm run dev
```
```

---

## Wymagania sprzętowe

| Komponent | Minimalne | Zalecane |
|---|---|---|
| GPU | RTX 3080 (10GB) | RTX 3090 (24GB) |
| RAM | 32GB | 64GB |
| CPU | 8 rdzeni | 16+ rdzeni |
| Dysk | 50GB SSD | 200GB NVMe |

Projekt był projektowany pod RTX 3090 + Ryzen 7950X + 64GB RAM. Przy 2M replay buffer i batch 512 zużycie VRAM wynosi około 8-12GB, RAM około 4-6GB.

---

## Architektura inspiracji

- **Ape-X** (Horgan et al., DeepMind 2018) — distributed Actor-Learner
- **Rainbow** (Hessel et al., DeepMind 2017) — kombinacja ulepszeń DQN
- **R2D2** (Kapturowski et al., DeepMind 2019) — rekurencja w Ape-X
- **Attention Is All You Need** (Vaswani et al., Google 2017) — Transformer
- **Dueling Network Architectures** (Wang et al., DeepMind 2016)
- **Prioritized Experience Replay** (Schaul et al., DeepMind 2015)

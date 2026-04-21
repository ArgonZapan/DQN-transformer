# Graceful Shutdown

## Przegląd

System składa się z czterech procesów które muszą być zamknięte w odpowiedniej kolejności — bez tego sockety ZMQ mogą blokować porty przy kolejnym uruchomieniu, a niezapisany checkpoint oznacza utratę postępu treningu.

## Kolejność zamykania

```
1. Actorzy        ← przestają generować doświadczenia
2. Learner        ← kończy bieżący krok, zapisuje checkpoint
3. Monitoring Svc ← kończy agregację metryk
4. Dashboard      ← zamknięcie okna przeglądarki
```

Actorzy muszą być zamknięci przed Learnerem — gdyby Learner zamknął się pierwszy, Actorzy zaczęliby rzucać błędami ZMQ timeout.

## stop.bat (Windows)

```bat
@echo off
echo Zatrzymywanie systemu Trading DQN...

:: 1. Zatrzymaj Actorzy (Node.js)
echo [1/4] Zatrzymywanie Aktorow...
taskkill /F /FI "WINDOWTITLE eq actors*" /T
timeout /t 3 /nobreak >nul

:: 2. Zatrzymaj Learner (Python)
echo [2/4] Zatrzymywanie Learnera (zapisuje checkpoint)...
taskkill /F /FI "WINDOWTITLE eq learner*" /T
timeout /t 10 /nobreak >nul

:: 3. Zatrzymaj Monitoring Service
echo [3/4] Zatrzymywanie Monitoring Service...
taskkill /F /FI "WINDOWTITLE eq monitoring*" /T
timeout /t 2 /nobreak >nul

:: 4. Zatrzymaj Dashboard
echo [4/4] Zatrzymywanie Dashboard...
taskkill /F /FI "WINDOWTITLE eq dashboard*" /T

echo System zatrzymany.
```

> **Uwaga:** `run.bat` musi nadawać oknom odpowiednie tytuły (`TITLE learner`, `TITLE actors` itd.) żeby `stop.bat` mógł je znaleźć.

## run.bat — nadawanie tytułów oknom

```bat
@echo off

:: Monitoring Service
start "monitoring" cmd /k "cd monitoring && node server.js"

:: Python Learner
start "learner" cmd /k "cd python && python main.py"

:: Poczekaj na start Learnera
timeout /t 5 /nobreak >nul

:: Actorzy
start "actors" cmd /k "cd node && node index.js"

:: Dashboard
start "dashboard" cmd /k "cd dashboard && npm run dev"

echo System Trading DQN uruchomiony.
echo Aby zatrzymac: stop.bat
```

## Obsługa SIGTERM w każdym module

### Python (Learner)

```python
import signal
import sys

def graceful_shutdown(signum, frame):
    logger.info("Otrzymano sygnał shutdown — zapisuję checkpoint...")

    # Zakończ bieżący krok treningowy
    trainer.finish_current_step()

    # Zapisz checkpoint i eksportuj ONNX
    trainer.save_checkpoint(f"checkpoints/shutdown_checkpoint.pt")
    trainer.export_onnx()   # checkpoints/model.onnx — używany przez Aktorów do lokalnej inferencji
    logger.info("Checkpoint i ONNX zapisane. Zamykam.")

    # Zamknij sockety ZMQ
    zmq_server.close()
    zmq_context.term()

    sys.exit(0)

signal.signal(signal.SIGTERM, graceful_shutdown)
signal.signal(signal.SIGINT, graceful_shutdown)  # Ctrl+C
```

### Node.js (Actorzy)

```javascript
async function gracefulShutdown() {
    logger.info('Otrzymano sygnał shutdown — kończę epizody...');

    // Zatrzymaj wszystkich Aktorów
    await actorManager.stopAll();

    // Zamknij sockety ZMQ
    zmqClient.close();
    monitoringClient.close();

    logger.info('Aktorzy zamknięci.');
    process.exit(0);
}

process.on('SIGTERM', gracefulShutdown);
process.on('SIGINT', gracefulShutdown);
```

### Node.js (Monitoring Service)

```javascript
async function gracefulShutdown() {
    logger.info('Monitoring Service — zamykam...');

    // Zakończ obsługę ostatnich wiadomości
    await new Promise(resolve => setTimeout(resolve, 1000));

    zmqPullSocket.close();
    httpServer.close();

    logger.info('Monitoring Service zamknięty.');
    process.exit(0);
}

process.on('SIGTERM', gracefulShutdown);
process.on('SIGINT', gracefulShutdown);
```

## Problemy przy ponownym uruchomieniu

### Port already in use

Jeśli poprzednie zamknięcie było niegraceful, porty ZMQ mogą być zajęte:

```bat
:: Sprawdź co używa portu
netstat -ano | findstr :5555

:: Zabij proces po PID
taskkill /F /PID <PID>
```

Lub zmień porty w `config.toml` i uruchom ponownie.

### Checkpoint z shutdown

Przy restarcie po shutdown, Learner automatycznie wczytuje `shutdown_checkpoint.pt` jeśli `resume_from_checkpoint` nie jest ustawiony:

```python
def find_checkpoint_to_load(config):
    explicit = config['training'].get('resume_from_checkpoint', '')
    if explicit:
        return explicit

    # Fallback do shutdown checkpoint
    shutdown_path = 'checkpoints/shutdown_checkpoint.pt'
    if os.path.exists(shutdown_path):
        logger.info(f"Wznawianie z shutdown checkpoint: {shutdown_path}")
        return shutdown_path

    return None  # Start od zera
```

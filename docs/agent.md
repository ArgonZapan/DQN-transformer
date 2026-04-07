# Instrukcje dla agenta

## Rola

Implementujesz system tradingu algorytmicznego zgodnie z dokumentacją w folderze `docs/`. Dokumentacja jest źródłem prawdy — implementuj dokładnie to co jest tam opisane, nie wymyślaj własnych rozwiązań.

## Zasady ogólne

**Czytaj dokumentację przed implementacją.** Przed napisaniem każdego modułu przeczytaj odpowiedni plik docs/. Architektura, kontrakty komunikacyjne i schematy danych są tam precyzyjnie opisane.

**Jeden moduł na raz.** Implementuj i testuj jeden moduł w pełni zanim przejdziesz do następnego. Nie pisz szkieletów wielu plików jednocześnie.

**TDD.** Najpierw testy, potem implementacja. Każdy moduł musi przejść 100% testów przed przejściem dalej. Testy w `tests/python/` i `tests/node/`.

**Zero hardcoded zmiennych.** Każda wartość liczbowa i konfiguracyjna pochodzi z `config.toml`. Jeśli coś nie ma swojego pola w configu — dodaj je zamiast wpisywać wartość w kodzie.

## Komentarze w kodzie

Komentarze tylko w miejscach nieoczywistych — tam gdzie logika mogłaby być niejasna bez wyjaśnienia. Nie komentuj tego co widać z samego kodu.

**Komentuj:**
- Pinned memory i `non_blocking=True` — dlaczego
- SumTree w PER — nieoczywista struktura danych
- Action masking — dlaczego `-inf` zamiast `0`
- Synchronizacja czasowa timeframe'ów — look-ahead bias
- Wzór Dueling DQN — odejmowanie `mean(A)`

**Nie komentuj:**
- Przypisań zmiennych
- Pętli for/while
- Ładowania configu
- Standardowych operacji PyTorch/Node.js
- Oczywistych warunków if/else

## Kolejność implementacji

Implementuj w tej kolejności — każdy etap zależy od poprzedniego:

1. `config.toml` + loadery konfiguracji (Python i Node.js)
2. `scripts/download_data.js` — pobieranie danych historycznych
3. `node/data/` — indicators, normalizer, binance client
4. `node/env/` — state, reward, episode, tradingEnv
5. `python/model/network.py` — architektura sieci
6. `python/training/replay_buffer.py` + `prioritized_buffer.py`
7. `python/server/zmq_server.py` + schemas
8. `python/training/trainer.py`
9. `python/main.py`
10. `node/actors/actor.js` + `actorManager.js`
11. `node/client/pythonClient.js`
12. `node/index.js`
13. `monitoring/server.js`
14. `dashboard/` — komponenty React
15. `run.bat` + `stop.bat`
16. `scripts/evaluate.py` — backtesting
17. `debug/` — smoke testy integracyjne

## Komunikacja między modułami

Trzymaj się kontraktów z `docs/communication.md`. Nie zmieniaj schematów wiadomości bez powodu — Actor i Learner są pisane niezależnie i muszą być kompatybilne.

Porty są w `config.toml` — nigdy hardcoded w kodzie.

## Obsługa błędów

Każdy moduł musi obsługiwać błędy na swoim poziomie:
- Błędy ZMQ: loguj i retry, nie crashuj całego systemu
- Błędy Binance API: exponential backoff zgodnie z `docs/data.md`
- Błędy konfiguracji: fail fast przy starcie, czytelny komunikat

## Pytania i decyzje

Jeśli dokumentacja jest niejednoznaczna w jakimś miejscu — zapytaj zamiast zgadywać. Lepiej potwierdzić intencję niż zaimplementować coś niezgodnego z resztą systemu.

Jeśli napotkasz sytuację której dokumentacja nie opisuje — zaproponuj rozwiązanie i zapytaj o akceptację przed implementacją.

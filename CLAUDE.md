# CLAUDE.md — instrukcje dla Claude Code

## Repozytorium

- GitHub: https://github.com/ArgonZapan/DQN-transformer
- Branch główny: `main`

## Po zakończeniu implementacji funkcjonalności

Po każdym zakończonym bloku pracy (nowa funkcja, poprawka, refaktor) wykonaj w kolejności:

1. Sprawdź status repozytorium:
   ```
   git status
   git diff
   ```

2. Przejrzyj zmiany — upewnij się że nie ma śmieciowych plików (logi, checkpointy, dane CSV, `venv_cuda/`, `runs/`, `__pycache__`).

3. Dodaj zmienione pliki (nigdy `git add .` bez przeglądu):
   ```
   git add <konkretne pliki>
   ```

4. Utwórz commit z opisowym komunikatem:
   - Pierwsza linia: krótki opis zmiany (max 72 znaki)
   - Jeśli potrzeba — pusta linia i rozwinięcie w punktach
   - Format: `typ: opis` np. `feat:`, `fix:`, `refactor:`, `docs:`

5. Wypchnij na GitHub:
   ```
   git push origin main
   ```

## Czego nigdy nie commitować

- `venv_cuda/` — środowisko wirtualne Python
- `runs/` — logi TensorBoard
- `python/checkpoints/` — checkpointy modelu
- `python/logs/` — pliki logów
- `node/data/historical/` — dane CSV
- `python/diagnostics/metrics.jsonl`, `health_checks.jsonl`, `baseline.json` — dane runtime
- `node_modules/` — zależności Node.js
- `*.pt`, `*.pth` — wagi modelu
- `shutdown.flag`

## Gitignore

Plik `.gitignore` powinien już wykluczać powyższe. Jeśli nie — uzupełnij go zanim zrobisz commit.

## Język komentarzy w kodzie

Komentarze w kodzie piszemy **wyłącznie po angielsku**.

Jeśli napotkasz komentarz w innym języku (polskim lub innym) — zaktualizuj go na angielski zanim skończysz pracę z danym plikiem.

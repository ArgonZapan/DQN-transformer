@echo off
echo Zatrzymywanie systemu Trading DQN...

:: 1. Zatrzymaj Aktorzy (Node.js) - każdy w osobnym oknie
echo [1/4] Zatrzymywanie Aktorow...
taskkill /F /FI "WINDOWTITLE eq actor_BTCUSDT*" /T >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq actor_ETHUSDT*" /T >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq actor_SOLUSDT*" /T >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq actors*" /T >nul 2>&1
timeout /t 3 /nobreak >nul

:: 2. Zatrzymaj Learner (Python) — zapisze checkpoint
echo [2/4] Zatrzymywanie Learnera (zapisuje checkpoint)...
taskkill /F /FI "WINDOWTITLE eq learner*" /T >nul 2>&1
timeout /t 10 /nobreak >nul

:: 3. Zatrzymaj Monitoring Service
echo [3/4] Zatrzymywanie Monitoring Service...
taskkill /F /FI "WINDOWTITLE eq monitoring*" /T >nul 2>&1
timeout /t 2 /nobreak >nul

:: 4. Zatrzymaj Dashboard
echo [4/4] Zatrzymywanie Dashboard...
taskkill /F /FI "WINDOWTITLE eq dashboard*" /T >nul 2>&1

echo.
echo System zatrzymany.
@echo off
echo Starting Trading DQN System...

:: Monitoring Service
start "monitoring" cmd /k "cd monitoring && node server.js"

:: Python Learner
start "learner" cmd /k "cd python && python main.py"

:: Wait for Learner to start
timeout /t 5 /nobreak >nul

:: Actors
start "actors" cmd /k "cd node && node index.js"

:: Dashboard (jeśli istnieje)
if exist dashboard\package.json (
    start "dashboard" cmd /k "cd dashboard && npm run dev"
)

echo.
echo System Trading DQN uruchomiony.
echo Aby zatrzymac: stop.bat

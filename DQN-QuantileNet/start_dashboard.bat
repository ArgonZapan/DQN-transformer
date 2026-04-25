@echo off
cd /d "%~dp0"

echo Starting QuantileNet Dashboard...
echo.

start "SSE Server" cmd /k "python -m python.server.sse_server --port 8080"
timeout /t 2 /nobreak >nul

start "Live Predictor" cmd /k "python -m python.live_predict --loop --interval 900"
timeout /t 2 /nobreak >nul

start "" "http://localhost:8080"

echo.
echo SSE Server:    http://localhost:8080
echo Live Predictor: writing every 900s
echo.
echo Close the two cmd windows to stop.
pause

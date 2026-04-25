@echo off
cd /d "%~dp0"

set VENV=%~dp0..\venv_cuda\Scripts\activate.bat

if not exist "%VENV%" (
    echo ERROR: venv_cuda not found at %VENV%
    pause
    exit /b 1
)

echo Starting QuantileNet Dashboard...
echo.

start "SSE Server" cmd /k "call %VENV% && python -m python.server.sse_server --port 8080"
timeout /t 2 /nobreak >nul

start "Live Predictor" cmd /k "call %VENV% && python -m python.live_predict --loop --interval 900"
timeout /t 2 /nobreak >nul

start "" "http://localhost:8080"

echo SSE Server:     http://localhost:8080
echo Live Predictor: writing every 900s
echo.
echo Close the two cmd windows to stop.
pause

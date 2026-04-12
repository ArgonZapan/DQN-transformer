@echo off
setlocal enabledelayedexpansion

echo Starting Trading DQN System...

:: Stop any existing processes
echo [Cleanup] Stopping existing processes...
taskkill /F /IM "python.exe" >nul 2>&1
taskkill /F /IM "node.exe" >nul 2>&1
timeout /t 2 /nobreak >nul

:: Check and install dependencies for Monitoring Service
if not exist "%~dp0monitoring\node_modules" (
    echo [Setup] Installing Monitoring Service dependencies...
    cd /d "%~dp0monitoring"
    call npm install
)

:: Check and install dependencies for Node.js Actors
if not exist "%~dp0node\node_modules" (
    echo [Setup] Installing Actors dependencies...
    cd /d "%~dp0node"
    call npm install
)

:: Dashboard
if exist "%~dp0dashboard\package.json" (
    if not exist "%~dp0dashboard\node_modules" (
        echo [Setup] Installing Dashboard dependencies...
        cd /d "%~dp0dashboard"
        call npm install
    )
)

:: Create venv_cuda if it doesn't exist
if not exist "%~dp0venv_cuda\Scripts\python.exe" (
    echo [Setup] Creating GPU environment with Python 3.12...
    py -3.12 -m venv venv_cuda
    echo [Setup] Installing PyTorch with CUDA...
    call venv_cuda\Scripts\pip.exe install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    call venv_cuda\Scripts\pip.exe install pyzmq msgpack toml tensorboard
)

:: ROOT bez trailing backslash (unika problemu z "path\" w cudzysłowach)
set ROOT=%~dp0
set ROOT=%ROOT:~0,-1%
set VENV=%ROOT%\venv_cuda\Scripts\activate.bat

:: Otwórz pierwsze okno Windows Terminal z zakładką Learner
wt new-tab --title "Learner" cmd /k "cd /d "%ROOT%" && call "%VENV%" && python python\main.py"
timeout /t 1 /nobreak >nul

:: Dodaj pozostałe zakładki do tego samego okna (-w 0)
wt -w 0 new-tab --title "Monitoring" cmd /k "cd /d "%ROOT%\monitoring" && node server.js"
timeout /t 1 /nobreak >nul

:: Aktorzy uruchamiani z ROOT (config.toml używa ścieżek relatywnych od roota)
wt -w 0 new-tab --title "BTC" cmd /k "timeout /t 8 /nobreak >nul && cd /d "%ROOT%" && node node\index.js --actor=BTCUSDT"
timeout /t 1 /nobreak >nul

wt -w 0 new-tab --title "ETH" cmd /k "timeout /t 8 /nobreak >nul && cd /d "%ROOT%" && node node\index.js --actor=ETHUSDT"
timeout /t 1 /nobreak >nul

wt -w 0 new-tab --title "SOL" cmd /k "timeout /t 8 /nobreak >nul && cd /d "%ROOT%" && node node\index.js --actor=SOLUSDT"
timeout /t 1 /nobreak >nul

wt -w 0 new-tab --title "Dashboard" cmd /k "cd /d "%ROOT%\dashboard" && npx vite"
timeout /t 1 /nobreak >nul

wt -w 0 new-tab --title "TensorBoard" cmd /k "cd /d "%ROOT%" && call "%VENV%" && tensorboard --logdir=runs --port=6006 --bind_all"

echo.
echo System uruchomiony w Windows Terminal (7 zakladek).
echo TensorBoard: http://localhost:6006
echo Aby zatrzymac: stop.bat

endlocal

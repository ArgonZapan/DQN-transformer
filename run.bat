@echo off
setlocal enabledelayedexpansion

echo Starting Trading DQN System with GPU...

:: Stop any existing processes that might be using our ports
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

:: Dashboard (instaluj zależności jeśli potrzeba)
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

:: Start processes using start command with full paths
start "monitoring" node "%~dp0monitoring\server.js"
start "learner" cmd /k "cd /d "%~dp0" && call venv_cuda\Scripts\activate.bat && python python\main.py"

:: Wait for Learner to start
timeout /t 5 /nobreak >nul

:: Start each actor in separate window
start "actor_BTCUSDT" node "%~dp0node\index.js" --actor=BTCUSDT
start "actor_ETHUSDT" node "%~dp0node\index.js" --actor=ETHUSDT
start "actor_SOLUSDT" node "%~dp0node\index.js" --actor=SOLUSDT

start "dashboard" cmd /k "cd /d "%~dp0dashboard" && npx vite"

start "tensorboard" cmd /k "cd /d "%~dp0" && call venv_cuda\Scripts\activate.bat && tensorboard --logdir=runs --port=6006 --bind_all"

echo.
echo System Trading DQN uruchomiony z GPU (RTX 3090).
echo TensorBoard: http://localhost:6006
echo Aby zatrzymac: stop.bat

endlocal
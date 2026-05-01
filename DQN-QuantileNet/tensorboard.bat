@echo off
cd /d "%~dp0"
echo [QuantileNet] Launching TensorBoard on http://localhost:6006

set PYW=%~dp0..\venv_cuda\Scripts\pythonw.exe
if not exist "%PYW%" (
    echo ERROR: pythonw.exe not found at %PYW%
    pause
    exit /b 1
)

REM Check if tensorboard is already running on port 6006
netstat -ano | findstr :6006 >nul 2>&1
if %errorlevel% equ 0 (
    echo [QuantileNet] TensorBoard already running on port 6006
    start "" http://localhost:6006
    exit /b 0
)

REM pythonw.exe has no console, so neither it nor any child subprocess
REM (file watcher, plugin loader, nvidia-smi probe) opens a cmd window.
start "" "%PYW%" -m tensorboard.main --logdir python/runs --port 6006 --reload_interval 10

echo [QuantileNet] TensorBoard starting... waiting for port 6006
set /a tries=0
:wait
timeout /t 1 /nobreak >nul
netstat -ano | findstr :6006 >nul 2>&1
if %errorlevel% equ 0 goto ready
set /a tries+=1
if %tries% geq 20 (
    echo ERROR: TensorBoard did not start within 20s. Check that 'tensorboard' is installed in venv_cuda.
    pause
    exit /b 1
)
goto wait

:ready
echo [QuantileNet] TensorBoard up. Opening browser.
start "" http://localhost:6006

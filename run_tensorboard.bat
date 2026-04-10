@echo off
REM Uruchomienie TensorBoard do wizualizacji treningu

cd /d "%~dp0"

REM Sprawdź czy venv istnieje
if not exist "venv_cuda\Scripts\python.exe" (
    echo [Error] venv_cuda nie znaleziony! Uruchom najpierw run_gpu.bat
    pause
    exit /b 1
)

echo [TensorBoard] Starting TensorBoard on http://localhost:6006
echo [TensorBoard] Logs directory: runs
echo [TensorBoard] Press Ctrl+C to stop

call venv_cuda\Scripts\activate.bat
tensorboard --logdir=runs --port=6006 --bind_all

pause
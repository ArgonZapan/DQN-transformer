@echo off
REM Uruchomienie projektu DQN-OPUS z użyciem GPU (RTX 3090)
REM Wymaga Python 3.12 + PyTorch CUDA

cd /d "%~dp0"
echo [GPU Training] Activating venv...
call venv_cuda\Scripts\activate.bat

echo [GPU Training] Starting learner with GPU...
echo [GPU Training] Logs visible in this window - watch for "TRAINING STARTED"
echo [GPU Training] Check TensorBoard at http://localhost:6006
echo [GPU Training] ====================================
python python\main.py

pause

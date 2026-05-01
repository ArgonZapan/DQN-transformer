@echo off
cd /d "%~dp0"
echo [QuantileNet] Starting learner...
"C:\Users\erykg\Desktop\DQN-OPUS\venv_cuda\Scripts\python.exe" python/main.py
pause

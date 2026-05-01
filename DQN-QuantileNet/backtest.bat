@echo off
cd /d "%~dp0"
echo [QuantileNet] Running backtest...
"C:\Users\erykg\Desktop\DQN-OPUS\venv_cuda\Scripts\python.exe" python/backtest.py %*
pause

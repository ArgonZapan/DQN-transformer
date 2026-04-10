@echo off
setlocal
set "ROOT=%~dp0"
"%ROOT%venv_cuda\Scripts\python.exe" "%ROOT%python\backtest.py" --config "%ROOT%config.toml" %*
pause
endlocal

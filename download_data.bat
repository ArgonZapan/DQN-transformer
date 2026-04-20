@echo off
setlocal enabledelayedexpansion

echo ================================================
echo Downloading historical data for Trading DQN
echo ================================================
echo.

:: Get symbol from command line argument
set SYMBOL=%~1
if "%SYMBOL%"=="" (
    echo Select trading pair:
    echo   1  - BTCUSDT
    echo   2  - ETHUSDT
    echo   3  - SOLUSDT
    echo   4  - BNBUSDT
    echo   5  - XRPUSDT
    echo   6  - ADAUSDT
    echo   7  - LINKUSDT
    echo   8  - DOTUSDT
    echo   9  - DOGEUSDT
    echo   10 - AVAXUSDT
    echo   ALL - wszystkie pary, wszystkie timeframe'y
    set /p CHOICE="Enter choice (1-10 or ALL): "
    if "!CHOICE!"=="1"  set SYMBOL=BTCUSDT
    if "!CHOICE!"=="2"  set SYMBOL=ETHUSDT
    if "!CHOICE!"=="3"  set SYMBOL=SOLUSDT
    if "!CHOICE!"=="4"  set SYMBOL=BNBUSDT
    if "!CHOICE!"=="5"  set SYMBOL=XRPUSDT
    if "!CHOICE!"=="6"  set SYMBOL=ADAUSDT
    if "!CHOICE!"=="7"  set SYMBOL=LINKUSDT
    if "!CHOICE!"=="8"  set SYMBOL=DOTUSDT
    if "!CHOICE!"=="9"  set SYMBOL=DOGEUSDT
    if "!CHOICE!"=="10" set SYMBOL=AVAXUSDT
    if /i "!CHOICE!"=="ALL" set SYMBOL=ALL
)
if "%SYMBOL%"=="" set SYMBOL=ALL

:: Get number of days from command line argument
set DAYS=%~2
if "%DAYS%"=="" (
    echo.
    echo How many days of history to download?
    echo (Press Enter for default 1500 days / ~4 years)
    set /p DAYS="Enter days: "
)
if "%DAYS%"=="" set DAYS=1500
echo.
echo ================================================
echo Downloading %DAYS% days of history
echo Symbol: %SYMBOL%
echo ================================================
echo.

:: Ensure dependencies are installed
echo [Setup] Checking dependencies...
if not exist "%~dp0node\node_modules" (
    cd /d "%~dp0node"
    call npm install
    cd /d "%~dp0"
)
echo.

:: Active timeframes (1w excluded — candles_1w=0 in config)
set "TFS=1m 15m 1h 1d"

:: All symbols
set "ALL_SYMBOLS=BTCUSDT ETHUSDT SOLUSDT BNBUSDT XRPUSDT ADAUSDT LINKUSDT DOTUSDT DOGEUSDT AVAXUSDT"
set /a TF_COUNT=4
set /a SYM_COUNT=10
set /a TOTAL=%SYM_COUNT%*%TF_COUNT%

if /i "%SYMBOL%"=="ALL" goto download_all

:: Single pair
for %%s in (%ALL_SYMBOLS%) do (
    if "%%s"=="%SYMBOL%" goto download_pair
)
echo ERROR: Unknown symbol "%SYMBOL%"
pause
exit /b 1

:download_pair
echo Downloading %SYMBOL% ...
set COUNTER=0
for %%t in (%TFS%) do (
    set /a COUNTER+=1
    call :download_single "%SYMBOL%" %%t %DAYS%
    if errorlevel 1 (
        echo ERROR: %SYMBOL% %%t download failed!
        pause
        exit /b 1
    )
    echo [!COUNTER!/%TF_COUNT%] %SYMBOL% %%t DONE
)
goto end

:download_all
set COUNTER=0
for %%s in (%ALL_SYMBOLS%) do (
    echo.
    echo ── %%s ──────────────────────────────────────────
    for %%t in (%TFS%) do (
        set /a COUNTER+=1
        call :download_single %%s %%t %DAYS%
        if errorlevel 1 (
            echo ERROR: %%s %%t download failed!
            pause
            exit /b 1
        )
        echo [!COUNTER!/%TOTAL%] %%s %%t DONE
    )
)
goto end

:download_single
setlocal enabledelayedexpansion
set "SYM=%~1"
set "INT=%~2"
set "DYS=%~3"
echo.
echo [!INT!] !SYM! !INT! (!DYS! days)
echo ------------------------------------------------
call node "%~dp0scripts\download_data.js" --symbol "!SYM!" --interval "!INT!" --days !DYS!
endlocal & exit /b %errorlevel%

:end
echo.
echo ================================================
echo ALL DOWNLOADS COMPLETED!
echo ================================================
echo.
echo Data saved to: node\data\historical\
dir "%~dp0node\data\historical\" /b
echo.
pause

endlocal

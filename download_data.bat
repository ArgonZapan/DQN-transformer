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
    echo   1 - BTCUSDT
    echo   2 - ETHUSDT
    echo   3 - SOLUSDT
    echo   4 - ALL (all pairs, all timeframes)
    set /p CHOICE="Enter choice (1-4): "
    if "%CHOICE%"=="1" set SYMBOL=BTCUSDT
    if "%CHOICE%"=="2" set SYMBOL=ETHUSDT
    if "%CHOICE%"=="3" set SYMBOL=SOLUSDT
    if "%CHOICE%"=="4" set SYMBOL=ALL
)
if "%SYMBOL%"=="" set SYMBOL=ALL

:: Get number of days from command line argument
set DAYS=%~2
if "%DAYS%"=="" (
    echo.
    echo How many days of history to download?
    echo (Press Enter for default 1460 days / 4 years)
    set /p DAYS="Enter days: "
)
if "%DAYS%"=="" set DAYS=1460
echo.
echo ================================================
echo Downloading %DAYS% days of history
echo Symbol: %SYMBOL%
echo ================================================
echo.

:: Ensure dependencies are installed
echo [Setup] Checking dependencies...
if not exist "%~dp0scripts\node_modules" (
    cd /d "%~dp0scripts"
    call npm install
)
cd /d "%~dp0"
echo.

:: Define timeframes
set "TFS=1m 15m 1h 1d 1w"

:: Function to download for a symbol
set COUNTER=0

if "%SYMBOL%"=="BTCUSDT" goto download_pair
if "%SYMBOL%"=="ETHUSDT" goto download_pair
if "%SYMBOL%"=="SOLUSDT" goto download_pair
if "%SYMBOL%"=="ALL" goto download_all

goto end

:download_pair
echo.
echo ================================================
echo Downloading %SYMBOL% data
echo ================================================
for %%t in (%TFS%) do (
    set /a COUNTER+=1
    call :download_single "%SYMBOL%" %%t %DAYS%
    if errorlevel 1 (
        echo ERROR: %SYMBOL% %%t download failed!
        pause
        exit /b 1
    )
    echo [%%t] %SYMBOL% %%t DONE
)
goto end

:download_single
setlocal enabledelayedexpansion
set "SYM=%~1"
set "INT=%~2"
set "DYS=%~3"
echo.
echo [!INT!] !SYM! !INT! (!DYS! days)
echo ================================================
call node "%~dp0scripts\download_data.js" --symbol "!SYM!" --interval "!INT!" --days !DYS!
endlocal & exit /b %errorlevel%

:download_all
:: BTCUSDT
set COUNTER=1
for %%t in (%TFS%) do (
    call :download_single BTCUSDT %%t %DAYS%
    if errorlevel 1 (
        echo ERROR: BTCUSDT %%t download failed!
        pause
        exit /b 1
    )
    echo [%COUNTER%/15] BTCUSDT %%t DONE
    set /a COUNTER+=1
)
echo.

:: ETHUSDT
for %%t in (%TFS%) do (
    call :download_single ETHUSDT %%t %DAYS%
    if errorlevel 1 (
        echo ERROR: ETHUSDT %%t download failed!
        pause
        exit /b 1
    )
    echo [%COUNTER%/15] ETHUSDT %%t DONE
    set /a COUNTER+=1
)
echo.

:: SOLUSDT
for %%t in (%TFS%) do (
    call :download_single SOLUSDT %%t %DAYS%
    if errorlevel 1 (
        echo ERROR: SOLUSDT %%t download failed!
        pause
        exit /b 1
    )
    echo [%COUNTER%/15] SOLUSDT %%t DONE
    set /a COUNTER+=1
)

:end
echo.
echo ================================================
echo ALL DOWNLOADS COMPLETED!
echo ================================================
echo.
echo Data saved to: node\data\historical\
dir "%~dp0node\data\historical\"
echo.
pause

endlocal

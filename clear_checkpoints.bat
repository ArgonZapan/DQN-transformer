@echo off
setlocal enabledelayedexpansion

echo ============================================
echo   USUWANIE CHECKPOINTOW I DANYCH TRENINGU
echo ============================================
echo.
echo Zostana usuniete:
echo   python\checkpoints\*.pt
echo   runs\  (TensorBoard logs)
echo.

:: Generuj losowy kod 6-cyfrowy
set /a CODE=%RANDOM% * 10 / 32768 + 1
set /a CODE=!CODE! * %RANDOM% / 32768
set /a CODE=!CODE! * 9 + 100000
if !CODE! LSS 100000 set /a CODE=!CODE! + 100000
if !CODE! GTR 999999 set /a CODE=999999

echo Aby potwierdzic, wpisz kod: !CODE!
echo.
set /p INPUT=Kod:

if "!INPUT!" NEQ "!CODE!" (
    echo.
    echo [ANULOWANO] Niepoprawny kod. Nic nie zostalo usuniete.
    pause
    exit /b 1
)

echo.
echo [OK] Kod poprawny. Usuwam...
echo.

if exist "python\checkpoints" (
    del /q "python\checkpoints\*.pt" 2>nul
    echo [OK] Checkpointy usuniete.
) else (
    echo [--] Brak folderu python\checkpoints
)

if exist "runs" (
    rmdir /s /q "runs"
    echo [OK] TensorBoard logs (runs\) usuniete.
) else (
    echo [--] Brak folderu runs\
)

echo.
echo Gotowe.
pause

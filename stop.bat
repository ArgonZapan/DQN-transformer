@echo off
echo Zatrzymywanie systemu Trading DQN...

:: 1. Zatrzymaj Learner gracefully przez plik-flagę (żeby zapisał checkpoint)
echo [1/3] Wysylam sygnal shutdown do Learnera...
echo. > "%~dp0python\shutdown.flag"
echo Czekam max 60s na zapis checkpointu...
set /a i=0
:wait_loop
timeout /t 1 /nobreak >nul
set /a i+=1
if not exist "%~dp0python\shutdown.flag" goto learner_done
if %i% geq 60 goto force_kill
goto wait_loop

:force_kill
echo UWAGA: Learner nie odpowiedzial w 60s, force kill...

:learner_done

:: 2. Zatrzymaj wszystkie procesy Python (Learner + TensorBoard)
echo [2/3] Zatrzymywanie procesow Python...
taskkill /F /IM "python.exe" /T >nul 2>&1

:: 3. Zatrzymaj wszystkie procesy Node (Aktorzy + Monitoring + Dashboard)
echo [3/3] Zatrzymywanie procesow Node...
taskkill /F /IM "node.exe" /T >nul 2>&1

echo.
echo System zatrzymany.

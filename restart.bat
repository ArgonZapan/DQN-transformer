@echo off
setlocal
set "ROOT=%~dp0"

:: Zapisz skrypt VBS — wscript.exe to proces GUI, nie konsolowy.
:: Nie zostanie ubity przez taskkill /IM WindowsTerminal.exe ani /IM cmd.exe
>  "%ROOT%_restart.vbs" echo WScript.Sleep 2000
>> "%ROOT%_restart.vbs" echo Set oShell = CreateObject("WScript.Shell")
>> "%ROOT%_restart.vbs" echo oShell.Run Chr(34) ^& "%ROOT%stop.bat" ^& Chr(34), 1, True
>> "%ROOT%_restart.vbs" echo WScript.Sleep 7000
>> "%ROOT%_restart.vbs" echo oShell.Run Chr(34) ^& "%ROOT%run.bat" ^& Chr(34), 1, False

start "" /b wscript.exe "%ROOT%_restart.vbs"
echo Restart zainicjowany -- system zatrzyma sie za ~8 sekund i uruchomi ponownie.
endlocal

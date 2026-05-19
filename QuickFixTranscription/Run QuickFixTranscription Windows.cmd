@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0LocalTranscriptionWorkflow.ps1"
echo.
pause

@echo off
setlocal
cd /d "%~dp0"

set "POWERSHELL_HOST="
where pwsh.exe >nul 2>nul
if not errorlevel 1 set "POWERSHELL_HOST=pwsh.exe"

if not defined POWERSHELL_HOST (
    where powershell.exe >nul 2>nul
    if not errorlevel 1 set "POWERSHELL_HOST=powershell.exe"
)

if not defined POWERSHELL_HOST (
    echo PowerShell was not found. Trying to install PowerShell with winget...
    where winget.exe >nul 2>nul
    if errorlevel 1 (
        echo winget was not found. Install PowerShell manually, then run this launcher again.
        goto done
    )

    winget install -e --id Microsoft.PowerShell --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo PowerShell installation failed. Install PowerShell manually, then run this launcher again.
        goto done
    )

    where pwsh.exe >nul 2>nul
    if not errorlevel 1 set "POWERSHELL_HOST=pwsh.exe"
    if not defined POWERSHELL_HOST if exist "%ProgramFiles%\PowerShell\7\pwsh.exe" set "POWERSHELL_HOST=%ProgramFiles%\PowerShell\7\pwsh.exe"
)

if not defined POWERSHELL_HOST (
    echo PowerShell was installed, but this shell cannot find it yet.
    echo Close this window and run the launcher again.
    goto done
)

"%POWERSHELL_HOST%" -NoProfile -ExecutionPolicy Bypass -File "%~dp0LocalTranscriptionWorkflow.ps1"

:done
echo.
pause

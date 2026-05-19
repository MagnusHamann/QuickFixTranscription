[CmdletBinding()]
param(
    [switch]$Yes
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = $PSScriptRoot
$requirements = Join-Path $root "requirements.txt"
$venvRoot = Join-Path $root ".venv"
$venvPython = Join-Path $root ".venv\Scripts\python.exe"

function Confirm-Yes {
    param([string]$Question)
    if ($Yes) {
        return $true
    }
    $answer = Read-Host "$Question [y/N]"
    return $answer.Trim().ToLowerInvariant().StartsWith("y")
}

function Test-PythonExecutable {
    param([string]$Python)
    if (-not $Python -or -not (Test-Path -LiteralPath $Python)) {
        return $false
    }
    try {
        & $Python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" *> $null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

function Find-Python {
    if (Test-PythonExecutable -Python $venvPython) {
        return $venvPython
    }

    $candidatePaths = New-Object System.Collections.Generic.List[string]

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python -and $python.Source -and ($python.Source -notlike "*\Microsoft\WindowsApps\*")) {
        $candidatePaths.Add($python.Source)
    }

    foreach ($root in @(
        (Join-Path $env:LocalAppData "Programs\Python"),
        $env:ProgramFiles,
        ${env:ProgramFiles(x86)},
        "C:\"
    )) {
        if (-not $root -or -not (Test-Path -LiteralPath $root)) {
            continue
        }

        Get-ChildItem -LiteralPath $root -Directory -Filter "Python3*" -ErrorAction SilentlyContinue | ForEach-Object {
            $candidate = Join-Path $_.FullName "python.exe"
            if (Test-Path -LiteralPath $candidate) {
                $candidatePaths.Add($candidate)
            }
        }
    }

    foreach ($candidate in ($candidatePaths | Select-Object -Unique)) {
        if (Test-PythonExecutable -Python $candidate) {
            return $candidate
        }
    }

    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        try {
            $resolved = & py -3 -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $resolved -and (Test-Path -LiteralPath $resolved.Trim())) {
                return $resolved.Trim()
            }
        }
        catch {
            return $null
        }
    }

    return $null
}

function Ensure-Python {
    $python = Find-Python
    if ($python) {
        return $python
    }

    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "Python was not found and winget is unavailable. Install Python 3 manually, then rerun this script."
    }

    if (-not (Confirm-Yes "Python 3 is missing. Install it with winget now?")) {
        throw "Python 3 is required."
    }

    & winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        throw "Python installation failed."
    }

    $python = Find-Python
    if (-not $python) {
        throw "Python was installed, but this shell cannot run it yet. Reopen PowerShell and rerun this script."
    }
    return $python
}

function Ensure-FFmpeg {
    $local = Join-Path $root ".tools\ffmpeg\bin\ffmpeg.exe"
    if ((Test-Path -LiteralPath $local) -or (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
        return
    }

    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "FFmpeg was not found and winget is unavailable. Install FFmpeg manually from https://ffmpeg.org/download.html."
    }

    if (-not (Confirm-Yes "FFmpeg is missing. Install it with winget now?")) {
        throw "FFmpeg is required."
    }

    & winget install -e --id Gyan.FFmpeg --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        throw "FFmpeg installation failed."
    }
}

function Invoke-Python {
    param(
        [string]$Python,
        [string[]]$Arguments
    )

    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE."
    }
}

function Remove-BrokenVenv {
    if ((Test-Path -LiteralPath $venvRoot) -and -not (Test-PythonExecutable -Python $venvPython)) {
        $rootFull = [System.IO.Path]::GetFullPath($root).TrimEnd("\") + "\"
        $venvFull = [System.IO.Path]::GetFullPath($venvRoot)
        if (-not $venvFull.StartsWith($rootFull, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to remove unexpected virtual environment path: $venvFull"
        }
        Write-Host "Removing incomplete .venv from a previous failed setup."
        Remove-Item -LiteralPath $venvRoot -Recurse -Force
    }
}

Write-Host "Bootstrapping QuickFixTranscription for Windows."
$python = Ensure-Python
Ensure-FFmpeg
Remove-BrokenVenv

if (-not (Test-Path -LiteralPath $venvPython)) {
    Invoke-Python -Python $python -Arguments @("-m", "venv", $venvRoot)
}

& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "pip upgrade failed."
}

& $venvPython -m pip install -r $requirements
if ($LASTEXITCODE -ne 0) {
    throw "Python dependency installation failed."
}

Write-Host "Bootstrap complete."
Write-Host "Launch with: .\Run QuickFixTranscription Windows.cmd"

[CmdletBinding()]
param(
    [switch]$SetupOnly,
    [switch]$StartupCheck,
    [switch]$Yes
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = $PSScriptRoot
$bootstrap = Join-Path $root "agent-bootstrap.ps1"
$main = Join-Path $root "main.py"
$requirements = Join-Path $root "requirements.txt"
$dependencyRoot = Join-Path (Split-Path -Parent $root) "QuickFixAppDependencies"
$venvPython = Join-Path (Join-Path (Join-Path $dependencyRoot ".venvs") (Split-Path -Leaf $root)) "Scripts\python.exe"

function Test-LocalPython {
    if (-not (Test-Path -LiteralPath $venvPython)) {
        return $false
    }
    try {
        # The invocation operator preserves the Python snippet as one argument.
        # Start-Process flattens ArgumentList to a string in Windows PowerShell 5,
        # which made a healthy environment look broken and retriggered setup.
        $null = & $venvPython -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" 2>$null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

function Ensure-PythonRequirements {
    if (-not (Test-LocalPython)) {
        return
    }
    if (-not (Test-Path -LiteralPath $requirements)) {
        return
    }
    & $venvPython -m transcription.runtime_check $requirements
    if ($LASTEXITCODE -eq 0) {
        return
    }

    Write-Host "Installing missing or changed QuickFixTranscription Python requirements."
    & $venvPython -m pip install --disable-pip-version-check -r $requirements
    if ($LASTEXITCODE -ne 0) {
        throw "Python package setup did not complete."
    }
    & $venvPython -m transcription.runtime_check $requirements
    if ($LASTEXITCODE -ne 0) {
        throw "Python package setup completed, but required package versions are still unavailable."
    }
}

function Ensure-RuntimeAssets {
    if (-not (Test-LocalPython)) {
        return
    }
    & $venvPython -m transcription.model_setup --yes
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Runtime asset setup did not complete. You can still choose local paths in the app."
    }
    & $venvPython -m transcription.dote_setup --yes
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "DOTE Whisper setup did not complete. Transcription will stay disabled until the integrated local runtime is ready."
    }
}

if ($SetupOnly -or -not (Test-LocalPython)) {
    $bootstrapArgs = @{}
    if ($Yes) {
        $bootstrapArgs["Yes"] = $true
    }
    & $bootstrap @bootstrapArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

Ensure-PythonRequirements
Ensure-RuntimeAssets

if ($SetupOnly) {
    exit 0
}

if (Test-LocalPython) {
    $mainArgs = @($main)
    if ($StartupCheck) {
        $mainArgs += "--startup-check"
    }
    & $venvPython @mainArgs
    exit $LASTEXITCODE
}

$py = Get-Command py -ErrorAction SilentlyContinue
if ($py) {
    & py -3 $main
    exit $LASTEXITCODE
}

$python = Get-Command python -ErrorAction SilentlyContinue
if ($python) {
    & python $main
    exit $LASTEXITCODE
}

Write-Host "Python was not found. Run .\agent-bootstrap.ps1 first." -ForegroundColor Red
exit 1

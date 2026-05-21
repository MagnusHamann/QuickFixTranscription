[CmdletBinding()]
param(
    [switch]$SetupOnly,
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
        & $venvPython -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" *> $null
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
    & $venvPython -m pip install -r $requirements
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Python package setup did not complete. The app may be missing optional local features."
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
}

if ($SetupOnly -or -not (Test-LocalPython)) {
    $args = @("-ExecutionPolicy", "Bypass", "-File", $bootstrap)
    if ($Yes) {
        $args += "-Yes"
    }
    & powershell.exe @args
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
    & $venvPython $main
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

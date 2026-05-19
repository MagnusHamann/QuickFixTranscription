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
$venvPython = Join-Path $root ".venv\Scripts\python.exe"

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

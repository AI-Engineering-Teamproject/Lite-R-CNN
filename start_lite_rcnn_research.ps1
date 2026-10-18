$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$Python312 = Join-Path $ProjectRoot ".venv312\Scripts\python.exe"
$PythonDefault = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (Test-Path -LiteralPath $Python312) {
    $Python = $Python312
} elseif (Test-Path -LiteralPath $PythonDefault) {
    $Python = $PythonDefault
} else {
    $Python = (Get-Command python -ErrorAction Stop).Source
}

& $Python (Join-Path $ProjectRoot "scripts\continuous_research.py") @args
exit $LASTEXITCODE

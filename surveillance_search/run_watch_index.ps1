param(
    [string]$Location = "amsterdam",
    [string]$Split = "test",
    [double]$PollSeconds = 10
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Python venv not found. Run .\run_end_to_end.ps1 once first."
}

& $python -m surveillance_search watch-index --locations $Location --splits $Split --poll-seconds $PollSeconds

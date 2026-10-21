param(
    [string]$Python = "py -3.8",
    [string]$VenvPath = ".venv"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path "requirements-jetson.txt")) {
    throw "Run this script from the AeroEyes project root."
}

Write-Host "Creating virtual environment at $VenvPath..."
Invoke-Expression "$Python -m venv `"$VenvPath`""

$pip = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $pip)) {
    throw "Virtual environment was not created: $pip"
}

& $pip -m pip install --upgrade "pip<24.1" setuptools wheel
& $pip -m pip install -r requirements-jetson.txt

Write-Host ""
Write-Host "Environment created. Activate it with:"
Write-Host ".\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "On Jetson, install the JetPack-compatible PyTorch 1.12.1 wheel separately."

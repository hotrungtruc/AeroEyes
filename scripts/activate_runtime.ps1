$ErrorActionPreference = "Stop"

$limitsPath = Join-Path $PSScriptRoot "..\config\runtime_limits.env"
if (-not (Test-Path $limitsPath)) {
    throw "Runtime limits file not found: $limitsPath"
}

Get-Content $limitsPath | ForEach-Object {
    $line = $_.Trim()
    if ($line -and -not $line.StartsWith("#")) {
        $name, $value = $line -split "=", 2
        [Environment]::SetEnvironmentVariable($name, $value, "Process")
    }
}

Write-Host "AeroEyes runtime limits loaded for this PowerShell process."
Write-Host "Device: $env:AEROEYES_DEVICE; precision: $env:AEROEYES_PRECISION; target FPS: $env:AEROEYES_TARGET_FPS"

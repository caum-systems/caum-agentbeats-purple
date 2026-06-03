$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outDir = "C:\Users\LVNET\OneDrive\Desktop\CAUM\05_HANDOFFS"
$zip = Join-Path $outDir "CAUM_AGENTBEATS_PURPLE_BUNDLE_$stamp.zip"

if (!(Test-Path $outDir)) {
    New-Item -ItemType Directory -Path $outDir | Out-Null
}

$temp = Join-Path $env:TEMP "caum_agentbeats_purple_bundle_$stamp"
if (Test-Path $temp) {
    Remove-Item -Recurse -Force -LiteralPath $temp
}
New-Item -ItemType Directory -Path $temp | Out-Null

$dest = Join-Path $temp "CAUM_AGENTBEATS_PURPLE"
Copy-Item -Recurse -Path $root -Destination $dest

Get-ChildItem -Path $dest -Recurse -Force -Directory |
    Where-Object { $_.Name -in @(".venv", "__pycache__", ".pytest_cache", "runs", "runs_smoke") } |
    Remove-Item -Recurse -Force

Get-ChildItem -Path $dest -Recurse -Force -File |
    Where-Object { $_.Name -like "*.pyc" -or $_.Name -like "*.env" -or $_.Name -eq ".env" } |
    Remove-Item -Force

Compress-Archive -Path (Join-Path $temp "CAUM_AGENTBEATS_PURPLE") -DestinationPath $zip -Force
$hash = Get-FileHash -Algorithm SHA256 -Path $zip
[pscustomobject]@{
    zip = $zip
    sha256 = $hash.Hash.ToLowerInvariant()
} | ConvertTo-Json -Compress

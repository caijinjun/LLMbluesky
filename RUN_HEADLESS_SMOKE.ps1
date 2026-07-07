$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:BLUESKY_ROOT = "$PSScriptRoot\bluesky_project"
$env:ATC_LOG_DIR = "$PSScriptRoot\headless_validation\headless_dynamic_logs"
python .\headless_validation\headless_dynamic_sector_validation.py

$ErrorActionPreference = "Stop"

# Requires the model service on cjj:
#   http://127.0.0.1:18080/predict
# The SSH tunnel exposes it locally so BlueSky can call it without opening a
# remote port.
$tunnel = Start-Process -FilePath "ssh" -ArgumentList @(
    "-N",
    "-L", "18080:127.0.0.1:18080",
    "cjj"
) -WindowStyle Hidden -PassThru

try {
    Start-Sleep -Seconds 2
    $env:ATC_ACTION_API_URL = "http://127.0.0.1:18080/predict"
    $env:ATC_ACTION_API_MODE = "predict"
    $env:ATC_ACTION_TIMEOUT_SEC = "12"
    $env:ATC_ACTION_MAX_TOKENS = "160"
    $env:ATC_LLM_MODEL = "qwen3-4b-lora-interval"
    Set-Location "$PSScriptRoot\bluesky_project"
    python .\BlueSky.py
}
finally {
    if ($tunnel -and -not $tunnel.HasExited) {
        Stop-Process -Id $tunnel.Id -Force -ErrorAction SilentlyContinue
    }
}

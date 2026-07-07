$ErrorActionPreference = "Stop"
$env:ATC_LLM_API_URL = "http://127.0.0.1:8000/v1/chat/completions"
$env:ATC_LLM_MODEL = "mock-atc-explainer"
Set-Location "$PSScriptRoot\bluesky_project"
python .\BlueSky.py

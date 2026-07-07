$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:ATC_LLM_API_URL = "http://127.0.0.1:8000/v1/chat/completions"
$env:ATC_LLM_MODEL = "mock-atc-explainer"
python .\llm\mock_openai_compatible_server.py --host 127.0.0.1 --port 8000

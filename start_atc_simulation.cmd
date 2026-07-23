@echo off
setlocal

set "PROJECT=%~dp0bluesky_project"
set "PYTHON=D:\anaconda3\python.exe"

if not exist "%PYTHON%" (
    echo ERROR: Python was not found at %PYTHON%
    pause
    exit /b 1
)

if not exist "%PROJECT%\BlueSky.py" (
    echo ERROR: BlueSky.py was not found in %PROJECT%
    pause
    exit /b 1
)

set "ATC_AUTO_START=1"
cd /d "%PROJECT%"
start "ATC Simulation" "%PYTHON%" BlueSky.py

endlocal

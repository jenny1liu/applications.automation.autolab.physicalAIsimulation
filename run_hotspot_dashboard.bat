@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -m hotspot_detector.ui
if errorlevel 1 (
    echo [ERROR] Hotspot Dashboard exited with an error.
    pause
    exit /b 1
)
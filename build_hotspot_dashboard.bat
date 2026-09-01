@echo off
REM Builds the Hotspot Detection Benchmark Dashboard into a standalone Windows
REM app (dist\HotspotDashboard\HotspotDashboard.exe) that runs without a
REM Python install. Bundles PyTorch/OpenVINO/OpenCV models and the pseudo
REM ground-truth dataset alongside the executable.
REM
REM Usage:
REM   build_hotspot_dashboard.bat            (onedir build, recommended)
REM   build_hotspot_dashboard.bat --onefile  (single .exe, slower to start)

cd /d "%~dp0"
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Could not activate .venv - run this from a machine that already has the project's venv set up.
    pause
    exit /b 1
)

echo [build] Ensuring PyInstaller is up to date (older releases don't support newer Python versions)...
python -m pip install --upgrade pyinstaller pyinstaller-hooks-contrib
if errorlevel 1 (
    echo [ERROR] Failed to install/upgrade PyInstaller.
    pause
    exit /b 1
)

set MODE=--onedir
if /I "%~1"=="--onefile" set MODE=--onefile

echo [build] Cleaning previous build output...
if exist "build\HotspotDashboard" rmdir /s /q "build\HotspotDashboard"
if exist "dist\HotspotDashboard" rmdir /s /q "dist\HotspotDashboard"

echo [build] Running PyInstaller (%MODE%)... this can take several minutes.
python -m PyInstaller --noconfirm %MODE% --name HotspotDashboard --paths . ^
    --collect-all ultralytics ^
    --collect-all openvino ^
    --collect-all cv2 ^
    --collect-all matplotlib ^
    --add-data "thermal\yolov8n.pt;thermal" ^
    --add-data "thermal\yolov8n_openvino_model;thermal\yolov8n_openvino_model" ^
    --add-data "hotspot_detector\pseudo_ground_truth;hotspot_detector\pseudo_ground_truth" ^
    --add-data "runs\c_cover_obb\weights\best.pt;runs\c_cover_obb\weights" ^
    --add-data "runs\c_cover_obb\weights\best_openvino_model;runs\c_cover_obb\weights\best_openvino_model" ^
    hotspot_dashboard_entry.py

if errorlevel 1 (
    echo [ERROR] PyInstaller build failed.
    pause
    exit /b 1
)

echo.
echo [build] Removing the incomplete intermediate copy in build\HotspotDashboard\
echo         (it has no _internal folder next to it and will not run - only dist\ is a real app).
if exist "build\HotspotDashboard\HotspotDashboard.exe" del /q "build\HotspotDashboard\HotspotDashboard.exe"

echo.
echo [build] Done. Run the app from:
echo   dist\HotspotDashboard\HotspotDashboard.exe
echo   (copy the WHOLE dist\HotspotDashboard folder when distributing - do NOT run anything under build\)
pause

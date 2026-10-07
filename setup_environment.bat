@echo off
title BlinkBridge AI - setup
cd /d "%~dp0"
echo ============================================================
echo   BlinkBridge AI: one-time setup of the project's own Python
echo ============================================================
echo.
echo [1/3] Creating the project environment .venv with Python 3.12...
py -3.12 -m venv .venv 2>nul || python -m venv .venv
if not exist ".venv\Scripts\python.exe" goto fail
echo.
echo [2/3] Installing the project's libraries. This takes several minutes...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto fail
echo.
echo [3/3] Checking that MediaPipe works...
".venv\Scripts\python.exe" -c "import mediapipe as mp, google.protobuf as pb; mp.solutions.face_mesh.FaceMesh(); print('MediaPipe OK, protobuf', pb.__version__)"
if errorlevel 1 goto fail
echo.
echo ============================================================
echo   Setup complete. Start the app by double-clicking
echo   run_blinkbridge.bat
echo ============================================================
pause
exit /b 0

:fail
echo.
echo Something went wrong. Take a screenshot of the messages above.
pause
exit /b 1

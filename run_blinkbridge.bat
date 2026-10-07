@echo off
title BlinkBridge AI
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
if not exist ".venv\Scripts\python.exe" (
    echo The project environment .venv was not found.
    echo Double-click setup_environment.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" gui.py
if errorlevel 1 pause

@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Create .venv and install requirements.txt first. See README.md.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -X utf8 launch_gui.py
if errorlevel 1 (
    pause
    exit /b 1
)

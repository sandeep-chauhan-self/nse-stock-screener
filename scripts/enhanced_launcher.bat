@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo   ENHANCED EARLY WARNING SYSTEM - DIRECT LAUNCHER
echo ===================================================
echo.
echo This tool launches the Enhanced Early Warning System directly.
echo.

:: Check if Python is installed
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python 3.10 or later from https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Run the Enhanced Launcher
cd ..\src
python enhanced_launcher.py
cd ..\scripts

echo.
echo ===================================================
echo           ENHANCED ANALYSIS COMPLETE
echo ===================================================
echo.
pause
exit /b 0
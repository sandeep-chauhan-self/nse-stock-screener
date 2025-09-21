@echo off
REM ================================================================
REM NSE Stock Screener - Cleanup Script
REM Removes old temporary files and manages disk space
REM ================================================================

echo 🧹 Starting cleanup process...

REM Get project root
set "PROJECT_ROOT=%~dp0.."
cd /d "%PROJECT_ROOT%"

REM Clean up old log files (keep last 30 days)
echo 📁 Cleaning old log files...
if exist "logs" (
    forfiles /p logs /m *.log /d -30 /c "cmd /c del @path" 2>nul
    echo    ✅ Old log files cleaned
) else (
    echo    ℹ️ No logs directory found
)

REM Clean up old temporary analysis files (keep last 7 days)
echo 📊 Cleaning old temporary files...
if exist "output\temp" (
    forfiles /p "output\temp" /m *.* /d -7 /c "cmd /c del @path" 2>nul
    echo    ✅ Temporary files cleaned
) else (
    echo    ℹ️ No temp directory found
)

REM Clean up old chart files (keep last 14 days)
echo 📈 Cleaning old chart files...
if exist "output\charts" (
    forfiles /p "output\charts" /m *.png /d -14 /c "cmd /c del @path" 2>nul
    echo    ✅ Old charts cleaned
) else (
    echo    ℹ️ No charts directory found
)

REM Keep only last 50 analysis reports
echo 📋 Managing analysis reports...
if exist "output\reports" (
    cd "output\reports"
    for /f "skip=50 delims=" %%i in ('dir /b /o:-d enhanced_analysis_*.csv 2^>nul') do (
        del "%%i" 2>nul
    )
    cd "%PROJECT_ROOT%"
    echo    ✅ Old reports cleaned (kept last 50)
) else (
    echo    ℹ️ No reports directory found
)

REM Clean Python cache files
echo 🐍 Cleaning Python cache...
for /r . %%d in (__pycache__) do (
    if exist "%%d" (
        rmdir /s /q "%%d" 2>nul
    )
)
del /s /q *.pyc 2>nul
echo    ✅ Python cache cleaned

REM Display disk usage summary
echo 💾 Disk usage summary:
for %%a in (output\reports output\charts logs) do (
    if exist "%%a" (
        for /f "tokens=3" %%b in ('dir /s "%%a" ^| find "File(s)"') do (
            echo    %%a: %%b bytes
        )
    )
)

echo ✅ Cleanup completed successfully!
exit /b 0
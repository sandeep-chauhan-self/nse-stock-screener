@echo off
REM ================================================================
REM NSE Stock Screener - Main Automated Launcher
REM Runs comprehensive automated analysis sequence
REM ================================================================

setlocal EnableDelayedExpansion

REM Get script directory
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."

REM Set up environment
cd /d "%PROJECT_ROOT%"

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir logs

REM Set log file with timestamp
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YYYY=%dt:~0,4%"
set "MM=%dt:~4,2%"
set "DD=%dt:~6,2%"
set "HH=%dt:~8,2%"
set "MIN=%dt:~10,2%"
set "SS=%dt:~12,2%"
set "TIMESTAMP=%YYYY%%MM%%DD%_%HH%%MIN%%SS%"
set "LOGFILE=logs\automation_%TIMESTAMP%.log"

echo ================================================================ >> "%LOGFILE%"
echo NSE Stock Screener - Automated Analysis Started >> "%LOGFILE%"
echo Date: %DATE% Time: %TIME% >> "%LOGFILE%"
echo ================================================================ >> "%LOGFILE%"

REM Function to log messages
call :log "🚀 Starting NSE Stock Screener Automated Analysis"

REM Step 1: Cleanup old files
call :log "📁 Step 1: Cleaning up old files..."
call scripts\cleanup.bat >> "%LOGFILE%" 2>&1
if !ERRORLEVEL! neq 0 (
    call :log "❌ Cleanup failed with error !ERRORLEVEL!"
    goto :error_exit
)
call :log "✅ Cleanup completed successfully"

REM Step 2: Update stock symbols
call :log "📊 Step 2: Updating NSE stock symbols..."
python scripts\fetch_stock_symbols.py >> "%LOGFILE%" 2>&1
if !ERRORLEVEL! neq 0 (
    call :log "⚠️ Symbol update failed, using existing symbols"
) else (
    call :log "✅ Stock symbols updated successfully"
)

REM Step 3: Run quick morning scan
call :log "🌅 Step 3: Running morning market scan..."
python cli_analyzer.py --banking --technology --min-score 60 --output-prefix "auto_morning_scan" >> "%LOGFILE%" 2>&1
if !ERRORLEVEL! neq 0 (
    call :log "❌ Morning scan failed with error !ERRORLEVEL!"
    goto :error_exit
)
call :log "✅ Morning scan completed successfully"

REM Step 4: Generate summary report
call :log "📈 Step 4: Generating summary report..."
python scripts\generate_summary.py >> "%LOGFILE%" 2>&1
if !ERRORLEVEL! neq 0 (
    call :log "⚠️ Summary generation failed, continuing..."
) else (
    call :log "✅ Summary report generated"
)

REM Step 5: Send notification (if configured)
call :log "📧 Step 5: Sending notification..."
python scripts\send_notification.py >> "%LOGFILE%" 2>&1
if !ERRORLEVEL! neq 0 (
    call :log "⚠️ Notification failed, continuing..."
) else (
    call :log "✅ Notification sent successfully"
)

call :log "🎉 Automated analysis completed successfully!"
echo.
echo ================================================================
echo 🎉 NSE Stock Screener - Automated Analysis COMPLETED
echo ================================================================
echo 📁 Results saved to: output\reports\
echo 📋 Log file: %LOGFILE%
echo ⏰ Completed at: %DATE% %TIME%
echo ================================================================
echo.

goto :end

:error_exit
call :log "❌ Automated analysis failed!"
echo.
echo ================================================================
echo ❌ NSE Stock Screener - Automated Analysis FAILED
echo ================================================================
echo 📋 Check log file: %LOGFILE%
echo ⏰ Failed at: %DATE% %TIME%
echo ================================================================
echo.
exit /b 1

:log
echo %~1
echo [%DATE% %TIME%] %~1 >> "%LOGFILE%"
exit /b

:end
exit /b 0
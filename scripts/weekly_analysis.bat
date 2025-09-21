@echo off
REM ================================================================
REM NSE Stock Screener - Weekly Deep Analysis
REM Comprehensive weekly analysis with extended features
REM ================================================================

setlocal EnableDelayedExpansion

echo ================================================================
echo 🚀 NSE Stock Screener - Weekly Deep Analysis
echo ================================================================
echo ⏰ Started at: %DATE% %TIME%
echo ================================================================

REM Get script directory and project root
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
cd /d "%PROJECT_ROOT%"

REM Create logs if needed
if not exist "logs" mkdir logs

REM Set up timestamp
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YYYY=%dt:~0,4%"
set "MM=%dt:~4,2%"
set "DD=%dt:~6,2%"
set "TIMESTAMP=%YYYY%%MM%%DD%"

echo 📊 Step 1: Banking Sector Deep Analysis
python cli_analyzer.py --banking --min-score 55 --max-results 30 --output-prefix "weekly_banking_%TIMESTAMP%"
if !ERRORLEVEL! neq 0 (
    echo ❌ Banking analysis failed
    goto :error
)
echo ✅ Banking analysis completed

echo 💻 Step 2: Technology Sector Deep Analysis  
python cli_analyzer.py --technology --min-score 55 --max-results 30 --output-prefix "weekly_technology_%TIMESTAMP%"
if !ERRORLEVEL! neq 0 (
    echo ❌ Technology analysis failed
    goto :error
)
echo ✅ Technology analysis completed

echo 🏭 Step 3: Custom High-Quality Stocks Analysis
python cli_analyzer.py -s "RELIANCE,TCS,INFY,HDFCBANK,ICICIBANK,LT,MARUTI,BHARTIARTL,ITC,WIPRO,TECHM,KOTAKBANK,AXISBANK,BAJFINANCE,HINDUNILVR,ASIANPAINT,NESTLEIND,ULTRACEMCO,POWERGRID,NTPC" --min-score 50 --output-prefix "weekly_bluechips_%TIMESTAMP%"
if !ERRORLEVEL! neq 0 (
    echo ❌ Blue chips analysis failed
    goto :error
)
echo ✅ Blue chips analysis completed

echo 📈 Step 4: Market Summary Generation
python scripts\weekly_summary.py
if !ERRORLEVEL! neq 0 (
    echo ⚠️ Summary generation failed, continuing...
)

echo 📧 Step 5: Weekly Report Email
python scripts\weekly_email.py
if !ERRORLEVEL! neq 0 (
    echo ⚠️ Email sending failed, continuing...
)

echo ================================================================
echo 🎉 Weekly Deep Analysis COMPLETED Successfully!
echo ================================================================
echo 📁 Results location: output\reports\
echo 📋 Generated files:
echo    • weekly_banking_%TIMESTAMP%.csv
echo    • weekly_technology_%TIMESTAMP%.csv  
echo    • weekly_bluechips_%TIMESTAMP%.csv
echo ⏰ Completed at: %DATE% %TIME%
echo ================================================================

goto :end

:error
echo ================================================================
echo ❌ Weekly Deep Analysis FAILED!
echo ================================================================
echo ⏰ Failed at: %DATE% %TIME%
echo 💡 Check individual step outputs above
echo ================================================================
exit /b 1

:end
exit /b 0
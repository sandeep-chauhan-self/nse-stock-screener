@echo off
color 0A
cls
echo ===========================================================
echo      STOCKS EARLY WARNING SYSTEM
echo ===========================================================
echo.
echo Welcome to the Stocks Early Warning System!
echo This tool helps identify high-momentum stocks with potential
echo for significant price movements in the next 1-7 days.
echo.

:: Change to scripts directory and run launcher
cd scripts
call launcher.bat

:: Return to original directory
cd ..
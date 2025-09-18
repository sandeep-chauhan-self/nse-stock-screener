@echo off

cls

echo ===========================================================
echo      STOCKS EARLY WARNING SYSTEM - AUTOMATED SEQUENCE
echo ===========================================================

echo.
echo Welcome to the Stocks Early Warning System!
echo This tool will automatically perform the following steps:
echo  1. Clean Repository
echo  2. Manage Stock Lists
echo  3. Analyze Stocks
echo.
echo Processing...
echo.

echo ===========================================================
echo STEP 1: CLEANING REPOSITORY
echo ===========================================================
echo Running cleanup script...
echo.
powershell -ExecutionPolicy Bypass -File cleanup.ps1

echo.
echo ===========================================================
echo STEP 2: MANAGING STOCK LISTS
echo ===========================================================
echo Launching Stock List Manager...
echo.
call manage_stock_lists.bat

echo.
echo ===========================================================
echo STEP 3: ANALYZING STOCKS
echo ===========================================================
echo Launching Stock Analysis Tool...
echo.
call stocks_analysis.bat

echo.
echo ===========================================================
echo PROCESS COMPLETED
echo ===========================================================
exit /b 0
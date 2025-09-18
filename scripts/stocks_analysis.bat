@echo off
setlocal enabledelayedexpansion

cls
echo ===================================================
echo   STOCKS EARLY WARNING SYSTEM - ANALYSIS TOOL
echo ===================================================
echo.
echo This tool analyzes stocks for potential high momentum setups
echo with significant price movement potential in the next 1-7 days.
echo.

:: Check if Python is installed
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python 3.10 or later from https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Check if required packages are installed
echo Checking dependencies...

:: Use the fixed dependency check script
python check_deps.py
if %ERRORLEVEL% neq 0 (
    echo Installing missing dependencies...
    pip install yfinance pandas numpy matplotlib requests beautifulsoup4
)

:: Clean up the temporary file
rem No need to delete any temporary file with the fixed script

echo.
echo Choose an analysis mode:
echo 1. Full interactive mode (customize all options)
echo 2. Quick analysis mode (analyze stocks from a text file)
echo 3. Exit
echo.

set /p mode_choice=Enter your choice (1-3): 

if "%mode_choice%"=="1" goto full_mode
if "%mode_choice%"=="2" goto quick_mode
if "%mode_choice%"=="3" goto exit_script

echo Invalid choice. Please try again.
pause
goto :eof

:full_mode
    echo.
    echo === FULL INTERACTIVE MODE ===
    echo.
    echo Stock Input Options:
    echo -------------------
    echo 1. Use default stocks (35 NSE stocks)
    echo 2. Use a text file (one stock per line)
    echo 3. Use a CSV file (with Symbol column)
    echo 4. Enter stocks manually (comma-separated)
    echo.

    set /p choice=Select an option (1-4): 

    set STOCK_ARGS=

    if "%choice%"=="1" (
        echo Using default stock list...
    ) else if "%choice%"=="2" (
        echo.
        echo Enter the path to your text file (or press Enter for nse_only_symbols.txt):
        set /p file_path=
        
        if "!file_path!"=="" (
            set file_path=..\data\nse_only_symbols.txt
        )
        
        if not exist "!file_path!" (
            echo Error: File !file_path! not found.
            echo Using default stock list instead.
        ) else (
            set STOCK_ARGS=--file "!file_path!"
            echo Using stock list from !file_path!
        )
    ) else if "%choice%"=="3" (
        echo.
        echo Enter the path to your CSV file (or press Enter for sample_stocks.csv):
        set /p file_path=
        
        if "!file_path!"=="" (
            set file_path=..\data\sample_stocks.csv
        )
        
        if not exist "!file_path!" (
            echo Error: File !file_path! not found.
            echo Using default stock list instead.
        ) else (
            set STOCK_ARGS=--file "!file_path!"
            echo Using stock list from !file_path!
        )
    ) else if "%choice%"=="4" (
        echo.
        echo Enter stock symbols separated by commas (e.g. SBIN,RELIANCE,TCS):
        set /p stock_list=
        
        if not "!stock_list!"=="" (
            set STOCK_ARGS=--stocks "!stock_list!"
            echo Using manually entered stocks
        ) else (
            echo No stocks entered. Using default stock list.
        )
    ) else (
        echo Invalid choice. Using default stock list.
    )

    echo.
    echo Batch Processing Options:
    echo -----------------------
    echo Enter batch size (stocks per batch, default is 50):
    set /p batch_size=

    if not "!batch_size!"=="" (
        set STOCK_ARGS=!STOCK_ARGS! --batch-size !batch_size!
    )

    echo Enter timeout between batches in seconds (default is 10):
    set /p timeout=

    if not "!timeout!"=="" (
        set STOCK_ARGS=!STOCK_ARGS! --timeout !timeout!
    )

    echo.
    echo Output Options:
    echo -------------
    echo Enter output directory (or press Enter for current directory):
    set /p output_dir=

    if not "!output_dir!"=="" (
        set STOCK_ARGS=!STOCK_ARGS! --output-dir "!output_dir!"
    ) else (
        set STOCK_ARGS=!STOCK_ARGS! --output-dir "..\output"
    )
    
    goto run_analysis

:quick_mode
    echo.
    echo === QUICK ANALYSIS MODE ===
    echo.
    echo This mode will analyze stocks listed in a text file.
    echo Each stock symbol should be on a separate line.
    echo.

    :ask_for_file
    echo Please enter the path to your stock list text file:
    echo (Example: my_stocks.txt or C:\Stocks\my_list.txt)
    echo.
    set /p stock_file="Stock file path: "

    if "!stock_file!"=="" (
        set stock_file=..\data\nse_only_symbols.txt
        echo Using default stock list: !stock_file!
    )

    if not exist "!stock_file!" (
        echo.
        echo Error: File "!stock_file!" not found.
        echo.
        goto ask_for_file
    )

    echo.
    echo File found. Configuring batch processing...
    echo.

    set batch_size=50
    set timeout=5


    set STOCK_ARGS=--file "!stock_file!" --batch-size !batch_size! --timeout !timeout! --output-dir "..\output"

    echo.
    echo ===================================================
    echo ANALYSIS CONFIGURATION:
    echo - Stock list: !stock_file!
    echo - Batch size: !batch_size! stocks per batch
    echo - Timeout: !timeout! seconds between batches
    echo - Output directory: ..\output
    echo ===================================================
    echo.
    
    goto run_analysis

:exit_script
    echo Exiting...
    exit /b 0

:run_analysis
    echo.
    echo Running Early Warning System...
    echo Command: python ..\src\early_warning_system.py !STOCK_ARGS!
    echo.
    echo ===================================================
    echo   RUNNING STOCK ANALYSIS
    echo ===================================================
    echo.

    :: Run the Early Warning System with arguments
    python ..\src\early_warning_system.py !STOCK_ARGS!

    echo.
    echo ===================================================
    echo               ANALYSIS COMPLETE
    echo ===================================================
    echo.
    echo Check the '..\output\charts' and '..\output\reports' folders for results.
    echo.
    pause
    exit /b 0
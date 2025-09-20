@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo  STOCK LIST MANAGER - AUTO MODE
echo ===================================================
echo.
echo Automatically fetching real stock symbols from exchanges...
echo.

goto fetch_real_symbols

:fetch_real_symbols
  echo.
  echo ===================================================
  echo  FETCH REAL STOCK SYMBOLS FROM EXCHANGES
  echo ===================================================
  echo.
  echo This will fetch real stock symbols from
  echo various stock exchanges like NSE, NYSE, NASDAQ, etc.
  echo and save them to sample_stocks.txt in the data folder
  echo.
  echo Requirements:
  echo - Python with yfinance, pandas, requests, beautifulsoup4
  echo.
 
  
  :: Check if required packages are installed
  echo Checking dependencies...
  
  :: Use the dependency check script
  python ..\src\check_deps.py
  if %ERRORLEVEL% neq 0 (
    echo Installing missing dependencies...
    pip install yfinance pandas requests beautifulsoup4
  )
  
  :: No need to clean up any temporary file
  
  echo.
  echo Running the stock symbol fetcher...
  echo.
  
  :: Run the Python script
  :: U can also use fetch_stock_symbols.py which was used originally
  python ..\src\Equity_all.py  
  
  echo.
  echo ===================================================
  echo Process completed Stock symbols have been saved to the data folder.
  echo ===================================================
  goto end_script

:end_script
echo.
exit /b 0

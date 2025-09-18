# nse-stock-screener



A comprehensive tool for analyzing stock momentum and identifying potential high-probability trading setups.## Overview

The Stocks Early Warning System is a tool designed to detect high-momentum stocks with potential for significant price movements in a short time frame (1-7 days). It analyzes stocks based on technical indicators and volume patterns to identify high-probability setups.

## Quick Start

## Folder Structure

1. **Run the tool**: Double-click `start.bat` or run it from command line```

2. **Choose analysis mode**: Full interactive or quick analysisStocksTool/

3. **Select stock source**: Default list, file, or manual entry├── .venv/                 # Python virtual environment

4. **Review results**: Check the `output/` folder for charts and reports├── charts/                # Generated technical charts for analyzed stocks

├── dist/                  # Standalone executable version

## Project Structure├── reports/               # CSV reports of analysis results

├── early_warning_system.py  # Main Python script

```├── analyze_stocks.bat     # Simple batch file to run analysis on a custom list

StocksTool/├── fetch_stocks.bat       # Batch file to fetch real stock symbols

├── start.bat              # Main launcher (start here)├── fetch_stock_symbols.py # Python script to download real stock symbols

├── src/                   # Python source code├── launcher.bat           # Main launcher for EXE and batch versions

│   ├── early_warning_system.py    # Main analysis engine├── run_stocks_tool.bat    # Interactive batch file with options

│   └── fetch_stock_symbols.py     # Stock symbol fetcher├── sample_stocks.csv      # Sample stock list in CSV format

├── scripts/               # Batch files and utilities└── sample_stocks.txt      # Sample stock list in text format

│   ├── launcher.bat       # Main menu system```

│   ├── stocks_analysis.bat # Stock analysis interface

│   ├── manage_stock_lists.bat # Stock list management## Essential Files

│   └── cleanup.ps1        # Repository cleanup utility

├── data/                  # Input data and stock lists### Core Files

│   └── sample_stocks.txt  # Sample stock symbols- **early_warning_system.py**: The main Python script that performs the analysis

├── output/                # Analysis results- **launcher.bat**: The primary entry point for using the tool (recommended)

│   ├── charts/            # Generated technical charts

│   └── reports/           # CSV analysis reports### Execution Options

└── docs/                  # Documentation- **dist/StocksEarlyWarningSystem.exe**: Standalone executable (no Python required)

    ├── README.md          # Detailed documentation- **run_stocks_tool.bat**: Interactive batch file (requires Python)

    ├── USAGE_GUIDE.md     # How to interpret results- **analyze_stocks.bat**: Simple batch file focused on analyzing custom stock lists

    └── CUSTOM_STOCKS_GUIDE.md # Creating custom stock lists

```### Stock List Management

- **fetch_stocks.bat**: Batch file to download real stock symbols from exchanges

## Features- **sample_stocks.txt**: Text file with stock symbols (one per line)

- **sample_stocks.csv**: CSV file with stock symbols and additional info

- **Stock Analysis**: Identifies high-momentum stocks using technical indicators

- **Custom Stock Lists**: Analyze your own stock selections### Documentation

- **Real Stock Data**: Fetch live symbols from major exchanges- **README.md**: This file - overview and getting started

- **Batch Processing**: Efficiently process large lists of stocks- **CUSTOM_STOCKS_GUIDE.md**: Guide for creating and using custom stock lists

- **Visual Charts**: Automatic chart generation for identified opportunities- **EXECUTABLE_GUIDE.md**: Guide for using the executable version

- **Detailed Reports**: CSV exports with probability scores and metrics- **USAGE_GUIDE.md**: Detailed guide on interpreting results



## Requirements## Quick Start Guide



- Python 3.10 or later### Option 1: Using the Launcher (Recommended)

- Required packages: yfinance, pandas, numpy, matplotlib, requestsThe launcher provides a simple menu to choose between the executable and batch file versions:



## Getting Started```

launcher.bat

The easiest way to start is by running `start.bat` which will guide you through the setup and analysis process.```



For detailed documentation, see `docs/README.md`.### Option 2: Directly Using the Executable
Run the standalone executable (no Python required):

```
dist\StocksEarlyWarningSystem.exe
```

### Option 3: Directly Using the Batch File
Run the interactive batch file (requires Python):

```
run_stocks_tool.bat
```

## How to Use Custom Stock Lists

### Step 1: Get Stock Symbols
You have two options:

#### Option A: Fetch Real Symbols from Exchanges
Run the stock symbol fetcher:
```
fetch_stocks.bat
```
This will:
1. Download real stock symbols from various exchanges
2. Save them to `sample_stocks.txt`

#### Option B: Create Your Own List
Create a text file with one stock symbol per line:
```
SBIN
RELIANCE
TCS
```

### Step 2: Analyze Your Custom List
Run the analyzer with your stock list:
```
analyze_stocks.bat
```
When prompted:
1. Enter the path to your stock list file
2. Configure batch processing settings
3. Wait for the analysis to complete

## How to Handle Large Lists (100-1000 Stocks)
When analyzing large lists of stocks:

1. Use appropriate batch settings:
   - Smaller batch size (10-20) for slower connections
   - Larger batch size (50-100) for faster connections
   - Timeout between batches (5-15 seconds) to avoid API rate limits

2. The system automatically:
   - Processes stocks in batches with progress tracking
   - Limits chart generation to top high probability stocks
   - Limits backtesting to top high probability stocks

## Features
- **Unusual Volume Detection**: Identifies stocks with significantly higher than average volume
- **Momentum Analysis**: Uses RSI and MACD to detect strong momentum
- **Automated Chart Generation**: Creates technical charts with key indicators for identified stocks
- **Backtesting**: Tests historical performance of similar signals to validate current setups
- **Report Generation**: Saves both high and medium probability candidates to CSV reports
- **Visualization**: Creates detailed charts showing signal performance
- **Custom Stock Lists**: Analyze your own list of stocks via file input or command line
- **Batch Processing**: Efficiently process large lists of 100-1000+ stocks

## Understanding the Results

### Output Files
Analysis results are saved in:
- **charts/**: Technical charts for identified stocks
- **reports/**: CSV reports with detailed analysis:
  - `high_probability_stocks_[timestamp].csv`
  - `medium_probability_stocks_[timestamp].csv`
  - `all_momentum_stocks_[timestamp].csv`

### Key Indicators
The system combines multiple signals:
- **Volume Analysis**: Looks for stocks with 1.5-3x average volume
- **RSI**: Identifies stocks in the 60-80 range (momentum sweet spot)
- **MACD**: Confirms bullish momentum via crossovers
- **Moving Averages**: 20-day and 50-day MAs for trend confirmation
- **Combined Score**: Overall probability rating

## Command Line Options
For advanced users, you can use command-line arguments directly:
```
python early_warning_system.py [options]
```

Options:
- `-f, --file FILE`: Path to a file containing stock symbols
- `-s, --stocks STOCKS`: Comma-separated list of stock symbols
- `-b, --batch-size SIZE`: Number of stocks per batch (default: 50)
- `-t, --timeout SECONDS`: Timeout between batches (default: 10)
- `-o, --output-dir DIR`: Directory to save output files

## Maintenance

### Rebuilding the Executable
If you make changes to the Python script and want to rebuild the executable:
```
build_exe.bat
```

### Updating Stock Lists
To update your stock symbols with the latest from exchanges:
```
fetch_stocks.bat
```

## Disclaimer
This tool provides probability-based analysis, not guaranteed predictions. Always perform your own due diligence before making trading decisions.
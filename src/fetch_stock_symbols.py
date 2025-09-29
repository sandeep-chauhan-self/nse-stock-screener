"""
Stock Symbol Fetcher - Downloads real stock symbols from various exchanges
and saves them to a text file for use with the Early Warning System.
"""

import os
import requests
import pandas as pd
import time
import random
from bs4 import BeautifulSoup, Tag
import yfinance as yf
from typing import cast

# File to save the stock symbols
OUTPUT_FILE = "..\\data\\sample_stocks.txt"
TEMP_DIR = "..\\data\\temp"

def fetch_nse_stocks():
    """Fetch NSE (India) stock symbols"""
    print("Fetching NSE (India) stock list...")
    
    try:
        # Create temp directory if it doesn't exist
        os.makedirs(TEMP_DIR, exist_ok=True)
        
        # NSE list from NSE India website
        url = "https://www1.nseindia.com/content/indices/ind_nifty500list.csv"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
            'Referer': 'https://www1.nseindia.com/'
        }
        
        # Try an alternative source if the primary one fails
        try:
            print("Trying primary source...")
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                raise Exception(f"Failed with status code: {response.status_code}")
                
            # Save the CSV temporarily
            temp_file = os.path.join(TEMP_DIR, "nse_temp.csv")
            with open(temp_file, "wb") as f:
                f.write(response.content)
            
            # Read the CSV
            df = pd.read_csv(temp_file)
            symbols = df["Symbol"].tolist()
            
            # Clean up
            os.remove(temp_file)
            
            # Add .NS suffix for Yahoo Finance
            symbols = [f"{symbol}.NS" for symbol in symbols]
            return symbols
            
        except Exception as e:
            print(f"Primary source failed: {e}")
            print("Trying alternative source...")
            
            # Try an alternative source - NIFTY 50 index
            alt_url = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
            response = requests.get(alt_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                # Save the CSV temporarily
                temp_file = os.path.join(TEMP_DIR, "nse_temp.csv")
                with open(temp_file, "wb") as f:
                    f.write(response.content)
                
                # Read the CSV
                df = pd.read_csv(temp_file)
                symbols = df["Symbol"].tolist()
                
                # Clean up
                os.remove(temp_file)
                
                # Add .NS suffix for Yahoo Finance
                symbols = [f"{symbol}.NS" for symbol in symbols]
                return symbols
            else:
                # Try fetching directly from Yahoo Finance top stocks in India
                print("Trying Yahoo Finance top Indian stocks...")
                
                # Predefined list of top NSE stocks as fallback
                top_nse_stocks = [
                    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
                    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "HDFC.NS", "BAJFINANCE.NS",
                    "BHARTIARTL.NS", "KOTAKBANK.NS", "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS",
                    "MARUTI.NS", "HCLTECH.NS", "SUNPHARMA.NS", "TITAN.NS", "BAJAJFINSV.NS",
                    "TATASTEEL.NS", "NTPC.NS", "POWERGRID.NS", "ULTRACEMCO.NS", "TATAMOTORS.NS",
                    "JSWSTEEL.NS", "TECHM.NS", "ADANIENT.NS", "WIPRO.NS", "M&M.NS",
                    "DIVISLAB.NS", "NESTLEIND.NS", "GRASIM.NS", "SBILIFE.NS", "HINDALCO.NS",
                    "APOLLOHOSP.NS", "DRREDDY.NS", "COALINDIA.NS", "HDFCLIFE.NS", "ONGC.NS",
                    "BAJAJ-AUTO.NS", "INDUSINDBK.NS", "UPL.NS", "CIPLA.NS", "BPCL.NS",
                    "EICHERMOT.NS", "ADANIPORTS.NS", "BRITANNIA.NS", "HEROMOTOCO.NS", "TATACONSUM.NS"
                ]
                print(f"Fetched {len(top_nse_stocks)} top NSE stocks from fallback list")
                return top_nse_stocks
        
    except Exception as e:
        print(f"Error fetching NSE stocks: {e}")
        return []

def fetch_us_stocks():
    """Fetch US stock symbols from major exchanges"""
    print("Fetching US stock list...")
    
    try:
        # Enhanced headers for avoiding blocks
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
        
        # Try primary source
        print("Trying primary source (Wikipedia)...")
        try:
            # S&P 500 components
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            session = requests.Session()
            response = session.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                table = soup.find('table', {'class': 'wikitable'})
                if not isinstance(table, Tag):
                    raise Exception("Could not find wikitable")
                
                symbols = []
                rows = list(cast(Tag, table).find_all('tr')) if isinstance(table, Tag) else []
                for row in rows[1:]:
                    cells = cast(Tag, row).find_all('td') if isinstance(row, Tag) else []
                    if cells:
                        symbol = cells[0].get_text(strip=True)
                        symbols.append(symbol)
                
                return symbols
            else:
                raise Exception(f"Failed with status code: {response.status_code}")
                
        except Exception as e:
            print(f"Primary source failed: {e}")
            print("Trying alternative source (GitHub list)...")
            
            # Try alternative source
            try:
                alt_url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
                response = requests.get(alt_url, headers=headers, timeout=15)
                
                if response.status_code == 200:
                    # Create temp directory if it doesn't exist
                    os.makedirs(TEMP_DIR, exist_ok=True)
                    
                    # Save the CSV temporarily
                    temp_file = os.path.join(TEMP_DIR, "sp500_temp.csv")
                    with open(temp_file, "wb") as f:
                        f.write(response.content)
                    
                    # Read the CSV
                    df = pd.read_csv(temp_file)
                    symbols = df["Symbol"].tolist()
                    
                    # Clean up
                    os.remove(temp_file)
                    
                    return symbols
                else:
                    raise Exception(f"Alternative source failed with status code: {response.status_code}")
            
            except Exception as e:
                print(f"Alternative source failed: {e}")
                print("Using built-in list of top US stocks...")
                
                # Fallback to a manually curated list of top US stocks
                top_us_stocks = [
                    "AAPL", "MSFT", "AMZN", "GOOGL", "FB", "TSLA", "BRK-B", "JNJ", "JPM", "V",
                    "PG", "UNH", "HD", "MA", "NVDA", "DIS", "BAC", "ADBE", "CMCSA", "XOM",
                    "NFLX", "VZ", "KO", "CSCO", "PEP", "T", "ABT", "WMT", "CRM", "TMO",
                    "INTC", "NKE", "PYPL", "MRK", "PFE", "ABBV", "ORCL", "CVX", "ACN", "AVGO",
                    "MCD", "DHR", "NEE", "LLY", "QCOM", "MDT", "TXN", "UPS", "BMY", "UNP"
                ]
                print(f"Fetched {len(top_us_stocks)} top US stocks from fallback list")
                return top_us_stocks
        
    except Exception as e:
        print(f"Error fetching US stocks: {e}")
        return []

def fetch_dow_jones():
    """Fetch Dow Jones Industrial Average components"""
    print("Fetching Dow Jones stocks...")
    
    try:
        # Enhanced headers for avoiding blocks
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
        
        # Try primary source
        print("Trying primary source (Wikipedia)...")
        try:
            url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
            session = requests.Session()
            response = session.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                tables = list(soup.find_all('table', {'class': 'wikitable'}))
                
                # The table with the components
                for table in tables:
                    if isinstance(table, Tag) and 'Components' in table.get_text():
                        symbols = []
                        rows = list(cast(Tag, table).find_all('tr')) if isinstance(table, Tag) else []
                        for row in rows[1:]:
                            cells = cast(Tag, row).find_all('td') if isinstance(row, Tag) else []
                            if len(cells) >= 2:
                                symbol = cells[1].get_text(strip=True)
                                symbols.append(symbol)
                        return symbols
                
                raise Exception("Could not find Components table in the page")
            else:
                raise Exception(f"Failed with status code: {response.status_code}")
                
        except Exception as e:
            print(f"Primary source failed: {e}")
            print("Using built-in list of Dow Jones stocks...")
            
            # Fallback to a manually curated list of Dow Jones stocks
            dow_jones_stocks = [
                "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "DOW",
                "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM",
                "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA", "WMT"
            ]
            print(f"Fetched {len(dow_jones_stocks)} Dow Jones stocks from fallback list")
            return dow_jones_stocks
        
    except Exception as e:
        print(f"Error fetching Dow Jones stocks: {e}")
        return []

def fetch_nasdaq_100():
    """Fetch NASDAQ-100 components"""
    print("Fetching NASDAQ-100 stocks...")
    
    try:
        # Enhanced headers for avoiding blocks
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
        
        # Try primary source
        print("Trying primary source (Wikipedia)...")
        try:
            url = "https://en.wikipedia.org/wiki/Nasdaq-100"
            session = requests.Session()
            response = session.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                tables = list(soup.find_all('table', {'class': 'wikitable'}))
                
                # The table with the components
                for table in tables:
                    if isinstance(table, Tag) and 'Ticker symbol' in table.get_text():
                        symbols = []
                        rows = list(cast(Tag, table).find_all('tr')) if isinstance(table, Tag) else []
                        for row in rows[1:]:
                            cells = cast(Tag, row).find_all('td') if isinstance(row, Tag) else []
                            if len(cells) >= 2:
                                symbol = cells[1].get_text(strip=True)
                                symbols.append(symbol)
                        return symbols
                
                raise Exception("Could not find table with Ticker symbol in the page")
            else:
                raise Exception(f"Failed with status code: {response.status_code}")
                
        except Exception as e:
            print(f"Primary source failed: {e}")
            print("Trying alternative source (direct API)...")
            
            # Try alternative source
            try:
                alt_url = "https://api.nasdaq.com/api/quote/list-type/nasdaq100"
                response = requests.get(alt_url, headers={**headers, 'Accept': 'application/json'}, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'data' in data and 'table' in data['data'] and 'rows' in data['data']['table']:
                        symbols = [item['symbol'] for item in data['data']['table']['rows']]
                        return symbols
                    else:
                        raise Exception("Unexpected JSON structure")
                else:
                    raise Exception(f"Alternative source failed with status code: {response.status_code}")
            
            except Exception as e:
                print(f"Alternative source failed: {e}")
                print("Using built-in list of NASDAQ-100 stocks...")
                
                # Fallback to a manually curated list of NASDAQ-100 stocks
                nasdaq_100_stocks = [
                    "AAPL", "ADBE", "ADI", "ADP", "ADSK", "AEP", "ALGN", "AMAT", "AMD", "AMGN",
                    "AMZN", "ANSS", "ASML", "ATVI", "AVGO", "BIDU", "BIIB", "BKNG", "CDNS", "CDW",
                    "CERN", "CHKP", "CHTR", "CMCSA", "COST", "CPRT", "CRWD", "CSCO", "CSX", "CTAS",
                    "CTSH", "DLTR", "DOCU", "DXCM", "EA", "EBAY", "EXC", "FAST", "FB", "FISV",
                    "FOX", "FOXA", "GILD", "GOOG", "GOOGL", "IDXX", "ILMN", "INCY", "INTC", "INTU",
                    "ISRG", "JD", "KDP", "KHC", "KLAC", "LRCX", "LULU", "MAR", "MCHP", "MDLZ",
                    "MELI", "MNST", "MRNA", "MRVL", "MSFT", "MU", "NFLX", "NTES", "NVDA", "NXPI",
                    "OKTA", "ORLY", "PAYX", "PCAR", "PDD", "PEP", "PTON", "PYPL", "QCOM", "REGN",
                    "ROST", "SBUX", "SGEN", "SIRI", "SNPS", "SPLK", "SWKS", "TEAM", "TMUS", "TSLA",
                    "TXN", "VRSK", "VRSN", "VRTX", "WBA", "WDAY", "XEL", "XLNX", "ZM"
                ]
                print(f"Fetched {len(nasdaq_100_stocks)} NASDAQ-100 stocks from fallback list")
                return nasdaq_100_stocks
        
    except Exception as e:
        print(f"Error fetching NASDAQ-100 stocks: {e}")
        return []

def fetch_ftse_100():
    """Fetch FTSE 100 components"""
    print("Fetching FTSE 100 stocks...")
    
    try:
        # Enhanced headers for avoiding blocks
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
        
        # Try primary source
        print("Trying primary source (Wikipedia)...")
        try:
            url = "https://en.wikipedia.org/wiki/FTSE_100_Index"
            session = requests.Session()
            response = session.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                tables = list(soup.find_all('table', {'class': 'wikitable'}))
                
                # The table with the components
                for table in tables:
                    if isinstance(table, Tag) and 'Ticker' in table.get_text():
                        symbols = []
                        rows = list(cast(Tag, table).find_all('tr')) if isinstance(table, Tag) else []
                        for row in rows[1:]:
                            cells = cast(Tag, row).find_all('td') if isinstance(row, Tag) else []
                            if len(cells) >= 2:
                                symbol = cells[1].get_text(strip=True) + ".L"  # Add .L suffix for London Stock Exchange
                                symbols.append(symbol)
                        return symbols
                
                raise Exception("Could not find table with Ticker in the page")
            else:
                raise Exception(f"Failed with status code: {response.status_code}")
                
        except Exception as e:
            print(f"Primary source failed: {e}")
            print("Trying alternative source (London Stock Exchange)...")
            
            # Try alternative source
            try:
                alt_url = "https://www.londonstockexchange.com/indices/ftse-100/constituents/table"
                response = requests.get(alt_url, headers=headers, timeout=15)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    table = soup.find('table', {'class': 'full-width'})
                    
                    if isinstance(table, Tag):
                        symbols = []
                        tbody = table.find('tbody')
                        rows = list(cast(Tag, tbody).find_all('tr')) if isinstance(tbody, Tag) else []
                        for row in rows:
                            cells = cast(Tag, row).find_all('td') if isinstance(row, Tag) else []
                            if len(cells) >= 2:
                                ticker_cell = cells[1]
                                symbol = ticker_cell.get_text(strip=True) + ".L"
                                symbols.append(symbol)
                        
                        if symbols:
                            return symbols
                    
                    raise Exception("Could not find ticker table")
                else:
                    raise Exception(f"Alternative source failed with status code: {response.status_code}")
            
            except Exception as e:
                print(f"Alternative source failed: {e}")
                print("Using built-in list of FTSE 100 stocks...")
                
                # Fallback to a manually curated list of FTSE 100 stocks
                ftse_100_stocks = [
                    "AAL.L", "ABF.L", "ADM.L", "AHT.L", "ANTO.L", "AUTO.L", "AV.L", "AZN.L", "BA.L", "BARC.L",
                    "BATS.L", "BDEV.L", "BKG.L", "BLND.L", "BLW.L", "BME.L", "BNZL.L", "BP.L", "BRBY.L", "BT-A.L",
                    "CCH.L", "CCL.L", "CNA.L", "CPG.L", "CRDA.L", "CRH.L", "DCC.L", "DGE.L", "DLG.L", "EVR.L",
                    "EXPN.L", "EZJ.L", "FERG.L", "FRES.L", "GLEN.L", "GSK.L", "HL.L", "HLMA.L", "HSBA.L", "IAG.L",
                    "ICP.L", "IHG.L", "III.L", "IMB.L", "INF.L", "ITRK.L", "ITV.L", "JD.L", "JMAT.L", "KGF.L",
                    "LAND.L", "LGEN.L", "LLOY.L", "LMI.L", "LSEG.L", "MGGT.L", "MKS.L", "MNDI.L", "MRO.L", "NG.L",
                    "NWG.L", "NXT.L", "OCDO.L", "PHNX.L", "PRU.L", "PSN.L", "PSON.L", "RB.L", "REL.L", "RIO.L",
                    "RKT.L", "RMV.L", "ROR.L", "RR.L", "RS1.L", "RSW.L", "SBRY.L", "SDR.L", "SGE.L", "SGRO.L",
                    "SHEL.L", "SKG.L", "SMDS.L", "SMIN.L", "SMT.L", "SN.L", "SPX.L", "SSE.L", "STAN.L", "STJ.L",
                    "SVT.L", "TSCO.L", "TUI.L", "ULVR.L", "UU.L", "VOD.L", "WEIR.L", "WPP.L", "WTB.L"
                ]
                print(f"Fetched {len(ftse_100_stocks)} FTSE 100 stocks from fallback list")
                return ftse_100_stocks
        
    except Exception as e:
        print(f"Error fetching FTSE 100 stocks: {e}")
        return []

def validate_symbols(symbols, max_to_check=100):
    """Validate that symbols exist on Yahoo Finance"""
    print(f"Validating symbols (checking up to {max_to_check} random symbols)...")
    
    valid_symbols = []
    
    # Shuffle and take a subset to check
    random.shuffle(symbols)
    to_check = symbols[:min(max_to_check, len(symbols))]
    
    for i, symbol in enumerate(to_check):
        print(f"Checking symbol {i+1}/{len(to_check)}: {symbol}")
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Check if we got valid data
            if 'symbol' in info:
                valid_symbols.append(symbol)
                print(f"✓ Valid: {symbol} - {info.get('shortName', 'Unknown')}")
            else:
                print(f"✗ Invalid: {symbol}")
                
            # Be gentle with the API
            time.sleep(1)
        except Exception as e:
            print(f"✗ Error checking {symbol}: {e}")
    
    # Return both the validated subset and the full list
    return valid_symbols, symbols

def save_symbols(symbols, filename=OUTPUT_FILE, limit=1000):
    """Save symbols to a text file"""
    # Ensure data directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Limit the number of symbols if necessary
    if len(symbols) > limit:
        print(f"Limiting to {limit} symbols from {len(symbols)} available")
        random.shuffle(symbols)
        symbols = symbols[:limit]
    
    print(f"Saving {len(symbols)} symbols to {filename}")
    
    with open(filename, 'w') as f:
        for symbol in symbols:
            f.write(f"{symbol}\n")
    
    print(f"Successfully saved {len(symbols)} stock symbols to {filename}")

def main():
    """Main function to fetch and save stock symbols"""
    # Import os here to ensure it's available in this scope
    import os
    
    print("=" * 50)
    print("STOCK SYMBOL FETCHER")
    print("=" * 50)
    print("This script will fetch real stock symbols from various exchanges")
    print("and save them to a file for use with the Early Warning System.")
    print("=" * 50)
    print()
    
    # Create necessary directories
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Ask user for preferences
    print("Which markets would you like to include?")
    print("1. NSE (India)")
    print("2. US Markets (S&P 500, Dow Jones, NASDAQ-100)")
    print("3. FTSE 100 (UK)")
    print("4. All available markets")
    print("5. Custom selection")
    print("6. Go back to previous menu")
    print("7. Exit")
    
    choice = input("\nEnter your choice (1-7): ")
    
    if choice == '6':
        return []  # Go back - return empty list
    elif choice == '7':
        print("Exiting...")
        exit(0)
    
    all_symbols = []
    
    if choice == '1' or choice == '4':
        all_symbols.extend(fetch_nse_stocks())
    
    if choice == '2' or choice == '4':
        all_symbols.extend(fetch_us_stocks())
        all_symbols.extend(fetch_dow_jones())
        all_symbols.extend(fetch_nasdaq_100())
    
    if choice == '3' or choice == '4':
        all_symbols.extend(fetch_ftse_100())
    
    if choice == '5':
        print("\nCustom selection:")
        include_nse = input("Include NSE (India)? (y/n): ").lower() == 'y'
        include_sp500 = input("Include S&P 500 (US)? (y/n): ").lower() == 'y'
        include_dow = input("Include Dow Jones (US)? (y/n): ").lower() == 'y'
        include_nasdaq = input("Include NASDAQ-100 (US)? (y/n): ").lower() == 'y'
        include_ftse = input("Include FTSE 100 (UK)? (y/n): ").lower() == 'y'
        
        if include_nse:
            all_symbols.extend(fetch_nse_stocks())
        if include_sp500:
            all_symbols.extend(fetch_us_stocks())
        if include_dow:
            all_symbols.extend(fetch_dow_jones())
        if include_nasdaq:
            all_symbols.extend(fetch_nasdaq_100())
        if include_ftse:
            all_symbols.extend(fetch_ftse_100())
    
    print(f"\nFound {len(all_symbols)} symbols total")
    
    # If no symbols were found, inform the user
    if len(all_symbols) == 0:
        print("\n⚠️ No stock symbols could be fetched from online sources.")
        print("This is likely due to connection issues or website restrictions.")
        print("Try again later or select option 2 in the main menu to generate simulated stock symbols.")
        return
    
    # Remove duplicates
    all_symbols = list(set(all_symbols))
    print(f"After removing duplicates: {len(all_symbols)} unique symbols")
    
    # Validate a subset of symbols
    print("\nWould you like to validate symbols with Yahoo Finance?")
    print("This helps ensure the symbols are valid but takes longer.")
    validate = input("Validate symbols? (y/n): ").lower() == 'y'
    
    if validate:
        max_to_check = int(input("How many symbols to check (10-100, default 20): ") or "20")
        valid_subset, all_symbols = validate_symbols(all_symbols, max_to_check)
        
        if len(valid_subset) < max_to_check * 0.5 and max_to_check >= 10:
            print("\nWARNING: Many symbols failed validation.")
            proceed = input("Proceed anyway? (y/n): ").lower() == 'y'
            if not proceed:
                print("Exiting without saving.")
                return
    
    # Ask for limit
    print("\nHow many symbols would you like to save?")
    limit = int(input(f"Enter number (1-{len(all_symbols)}, default 1000): ") or "1000")
    
    # Save to file
    save_symbols(all_symbols, OUTPUT_FILE, limit)
    
    print("\nDone! You can now use these symbols with the Early Warning System.")
    print(f"File saved to: {OUTPUT_FILE}")
    
    # Convert relative path to absolute for display
    abs_path = os.path.abspath(OUTPUT_FILE)
    print(f"Absolute path: {abs_path}")

if __name__ == "__main__":
    import os  # Import os here for absolute path in error handling
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    
    input("\nPress Enter to exit...")
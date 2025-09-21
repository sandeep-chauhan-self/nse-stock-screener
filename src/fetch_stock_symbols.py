"""
Stock Symbol Fetcher - Downloads real stock symbols from various exchanges
and saves them to a text file for use with the Early Warning System.
Cross-platform compatible with proper path handling and command-line options.
Enhanced with robust data fetching, retry logic, and caching.
"""
from pathlib import Path
import argparse
import logging
import os
import random
import sys
import time
from bs4 import BeautifulSoup
import pandas as pd
import requests
from .common.paths import PathManager, add_output_argument, resolve_output_path, get_data_path, get_temp_path, ensure_dir
from .data.compat import enhanced_yfinance as yf, enhanced_requests, get_nse_symbols
from .data.validation import validate_symbol

# Set[str] up logging
logger = logging.getLogger(__name__)

# Initialize path manager
pm = PathManager()

# Default output file (can be overridden via command line)
DEFAULT_OUTPUT_FILE = "sample_stocks.txt"
def fetch_nse_stocks():
    """Fetch NSE (India) stock symbols using enhanced data fetcher"""
    logger.info("Fetching NSE stock symbols using enhanced data layer")
    print("Fetching NSE (India) stock List[str]...")
    try:

        # Use the enhanced NSE fetcher with caching and retry logic
        symbols = get_nse_symbols(force_refresh=False)
        if symbols:

            # Add .NS suffix for Yahoo Finance compatibility
            symbols_with_suffix = [f"{symbol}.NS" for symbol in symbols]
            logger.info(f"Successfully fetched {len(symbols_with_suffix)} NSE symbols")
            print(f"✅ Successfully fetched {len(symbols_with_suffix)} NSE symbols")
            return symbols_with_suffix
        else:

            # Fallback to hardcoded List[str] if enhanced fetcher fails
            logger.warning("Enhanced NSE fetcher failed, using fallback List[str]")
            print("⚠️ Using fallback NSE stock List[str]...")
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
            print(f"📋 Using fallback List[str] with {len(top_nse_stocks)} top NSE stocks")
            return top_nse_stocks
    except Exception as e:
        logger.error(f"Error in NSE stock fetching: {e}")
        logging.error(f"❌ Error fetching NSE stocks: {e}")
        return []
def fetch_us_stocks():
    """Fetch US stock symbols from major exchanges"""
    print("Fetching US stock List[str]...")
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
                symbols = []
                for row in table.findAll('tr')[1:]:
                    symbol = row.findAll('td')[0].text.strip()
                    symbols.append(symbol)
                return symbols
            else:
                raise Exception(f"Failed with status code: {response.status_code}")
        except Exception as e:
            print(f"Primary source failed: {e}")
            print("Trying alternative source (GitHub List[str])...")

            # Try alternative source
            try:
                alt_url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
                response = requests.get(alt_url, headers=headers, timeout=15)
                if response.status_code == 200:

                    # Create temp directory if it doesn't exist
                    temp_dir = pm.ensure_temp_dir()

                    # Save the CSV temporarily
                    temp_file = pm.get_temp_path("sp500_temp.csv")
                    temp_file.write_bytes(response.content)

                    # Read the CSV
                    df = pd.read_csv(temp_file)
                    symbols = df["Symbol"].tolist()

                    # Clean up
                    temp_file.unlink()
                    return symbols
                else:
                    raise Exception(f"Alternative source failed with status code: {response.status_code}")
            except Exception as e:
                print(f"Alternative source failed: {e}")
                print("Using built-in List[str] of top US stocks...")

                # Fallback to a manually curated List[str] of top US stocks
                top_us_stocks = [
                    "AAPL", "MSFT", "AMZN", "GOOGL", "FB", "TSLA", "BRK-B", "JNJ", "JPM", "V",
                    "PG", "UNH", "HD", "MA", "NVDA", "DIS", "BAC", "ADBE", "CMCSA", "XOM",
                    "NFLX", "VZ", "KO", "CSCO", "PEP", "T", "ABT", "WMT", "CRM", "TMO",
                    "INTC", "NKE", "PYPL", "MRK", "PFE", "ABBV", "ORCL", "CVX", "ACN", "AVGO",
                    "MCD", "DHR", "NEE", "LLY", "QCOM", "MDT", "TXN", "UPS", "BMY", "UNP"
                ]
                print(f"Fetched {len(top_us_stocks)} top US stocks from fallback List[str]")
                return top_us_stocks
    except Exception as e:
        logging.error(f"Error fetching US stocks: {e}")
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
                tables = soup.findAll('table', {'class': 'wikitable'})

                # The table with the components
                for table in tables:
                    if 'Components' in table.text:
                        symbols = []
                        for row in table.findAll('tr')[1:]:
                            cells = row.findAll('td')
                            if len(cells) >= 2:
                                symbol = cells[1].text.strip()
                                symbols.append(symbol)
                        return symbols
                raise Exception("Could not find Components table in the page")
            else:
                raise Exception(f"Failed with status code: {response.status_code}")
        except Exception as e:
            print(f"Primary source failed: {e}")
            print("Using built-in List[str] of Dow Jones stocks...")

            # Fallback to a manually curated List[str] of Dow Jones stocks
            dow_jones_stocks = [
                "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "DOW",
                "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM",
                "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA", "WMT"
            ]
            print(f"Fetched {len(dow_jones_stocks)} Dow Jones stocks from fallback List[str]")
            return dow_jones_stocks
    except Exception as e:
        logging.error(f"Error fetching Dow Jones stocks: {e}")
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
                tables = soup.findAll('table', {'class': 'wikitable'})

                # The table with the components
                for table in tables:
                    if 'Ticker symbol' in table.text:
                        symbols = []
                        for row in table.findAll('tr')[1:]:
                            cells = row.findAll('td')
                            if len(cells) >= 2:
                                symbol = cells[1].text.strip()
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
                alt_url = "https://api.nasdaq.com/api/quote/List[str]-type/nasdaq100"
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
                print("Using built-in List[str] of NASDAQ-100 stocks...")

                # Fallback to a manually curated List[str] of NASDAQ-100 stocks
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
                print(f"Fetched {len(nasdaq_100_stocks)} NASDAQ-100 stocks from fallback List[str]")
                return nasdaq_100_stocks
    except Exception as e:
        logging.error(f"Error fetching NASDAQ-100 stocks: {e}")
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
                tables = soup.findAll('table', {'class': 'wikitable'})

                # The table with the components
                for table in tables:
                    if 'Ticker' in table.text:
                        symbols = []
                        for row in table.findAll('tr')[1:]:
                            cells = row.findAll('td')
                            if len(cells) >= 2:
                                symbol = cells[1].text.strip() + ".L"
  # Add .L suffix for London Stock Exchange
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
                    if table:
                        symbols = []
                        rows = table.find('tbody').findAll('tr')
                        for row in rows:
                            cells = row.findAll('td')
                            if len(cells) >= 2:
                                ticker_cell = cells[1]
                                symbol = ticker_cell.text.strip() + ".L"
                                symbols.append(symbol)
                        if symbols:
                            return symbols
                    raise Exception("Could not find ticker table")
                else:
                    raise Exception(f"Alternative source failed with status code: {response.status_code}")
            except Exception as e:
                print(f"Alternative source failed: {e}")
                print("Using built-in List[str] of FTSE 100 stocks...")

                # Fallback to a manually curated List[str] of FTSE 100 stocks
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
                print(f"Fetched {len(ftse_100_stocks)} FTSE 100 stocks from fallback List[str]")
                return ftse_100_stocks
    except Exception as e:
        logging.error(f"Error fetching FTSE 100 stocks: {e}")
        return []
def validate_symbols(symbols, max_to_check=100):
    """Validate that symbols exist using enhanced data fetcher"""
    logger.info(f"Validating symbols using enhanced data fetcher (checking up to {max_to_check} random symbols)")
    print(f"Validating symbols (checking up to {max_to_check} random symbols)...")
    valid_symbols = []

    # Shuffle and take a subset to check
    random.shuffle(symbols)
    to_check = symbols[:min(max_to_check, len(symbols))]
    for i, symbol in enumerate(to_check):
        print(f"Checking symbol {i+1}/{len(to_check)}: {symbol}")
        try:

            # Use enhanced validation from our data layer
            is_valid = validate_symbol(symbol)
            if is_valid:
                valid_symbols.append(symbol)
                print(f"✓ Valid: {symbol}")
            else:
                print(f"✗ Invalid: {symbol}")

            # Be gentle with the API
            time.sleep(0.5)
        except Exception as e:
            logger.warning(f"Error checking {symbol}: {e}")
            logging.error(f"✗ Error checking {symbol}: {e}")
    logger.info(f"Validated {len(valid_symbols)}/{len(to_check)} symbols")

    # Return both the validated subset and the full List[str]
    return valid_symbols, symbols
def save_symbols(symbols, output_path: Path, limit=1000):
    """Save symbols to a text file"""

    # Ensure parent directory exists
    ensure_dir(output_path.parent)

    # Limit the number of symbols if necessary
    if len(symbols) > limit:
        print(f"Limiting to {limit} symbols from {len(symbols)} available")
        random.shuffle(symbols)
        symbols = symbols[:limit]
    print(f"Saving {len(symbols)} symbols to {output_path}")

    # Write symbols to file
    output_path.write_text('\n'.join(symbols) + '\n')
    print(f"Successfully saved {len(symbols)} stock symbols to {output_path}")
def main():
    """Main function to fetch and save stock symbols"""

    # Set[str] up argument parsing
    parser = argparse.ArgumentParser(
        description="Fetch stock symbols from various exchanges and save to file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --output my_symbols.txt --limit 500
  %(prog)s --market nse --limit 100
  %(prog)s --market all --output /path/to/output.txt
        """
    )

    # Add arguments
    add_output_argument(parser, DEFAULT_OUTPUT_FILE, "Output file for stock symbols")
    parser.add_argument(
        '--market', '-m',
        choices=['nse', 'us', 'ftse', 'all'],
        default='nse',
        help='Market to fetch symbols from (default: nse)'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=1000,
        help='Maximum number of symbols to save (default: 1000)'
    )
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode with menu selection'
    )

    # Parse arguments
    args = parser.parse_args()

    # Resolve output path
    output_path = resolve_output_path(args.output, DEFAULT_OUTPUT_FILE, 'data')
    print("=" * 50)
    print("STOCK SYMBOL FETCHER")
    print("=" * 50)
    print("This script will fetch real stock symbols from various exchanges")
    logging.warning("and save them to a file for use with the Early Warning System.")
    print(f"Output will be saved to: {output_path}")
    print("=" * 50)
    print()

    # Ensure directories exist
    pm.ensure_data_dir()
    pm.ensure_temp_dir()
    all_symbols = []
    if args.interactive:

        # Interactive mode - show menu
        all_symbols = run_interactive_mode()
        if not all_symbols:
  # User chose to exit or go back
            return
    else:

        # Command-line mode - use specified market
        if args.market == 'nse':
            print("Fetching NSE (India) symbols...")
            nse_symbols = fetch_nse_stocks()
            if nse_symbols:
                all_symbols.extend(nse_symbols)
        elif args.market == 'us':
            print("Fetching US market symbols...")
            us_symbols = fetch_us_stocks()
            if us_symbols:
                all_symbols.extend(us_symbols)
        elif args.market == 'ftse':
            print("Fetching FTSE 100 symbols...")
            ftse_symbols = fetch_ftse_100()
            if ftse_symbols:
                all_symbols.extend(ftse_symbols)
        elif args.market == 'all':
            print("Fetching symbols from all available markets...")
            for fetch_func, name in [
                (fetch_nse_stocks, "NSE"),
                (fetch_us_stocks, "US Markets"),
                (fetch_ftse_100, "FTSE 100")
            ]:
                try:
                    print(f"\nFetching {name} symbols...")
                    symbols = fetch_func()
                    if symbols:
                        all_symbols.extend(symbols)
                        print(f"✓ Added {len(symbols)} {name} symbols")
                    else:
                        print(f"✗ No {name} symbols fetched")
                except Exception as e:
                    print(f"✗ Failed to fetch {name} symbols: {e}")

    # Check if we got any symbols
    if not all_symbols:
        print("❌ No symbols were fetched. Please check your internet connection and try again.")
        return
    print(f"\n📊 Total symbols fetched: {len(all_symbols)}")

    # Remove duplicates while preserving order
    seen = Set[str]()
    unique_symbols = []
    for symbol in all_symbols:
        if symbol not in seen:
            seen.add(symbol)
            unique_symbols.append(symbol)
    if len(unique_symbols) != len(all_symbols):
        print(f"🔄 Removed {len(all_symbols) - len(unique_symbols)} duplicate symbols")
        all_symbols = unique_symbols

    # Validate symbol limit
    if len(all_symbols) > args.limit:
        print(f"⚠️  You have {len(all_symbols)} symbols but requested limit of {args.limit}")
        if not args.interactive:
            print(f"🎲 Randomly selecting {args.limit} symbols from {len(all_symbols)} available")
        else:
            proceed = input("Proceed anyway? (y/n): ").lower() == 'y'
            if not proceed:
                print("Exiting without saving.")
                return

    # Save to file
    save_symbols(all_symbols, output_path, args.limit)
    logging.warning("\n✅ Done! You can now use these symbols with the Early Warning System.")
    print(f"📁 File saved to: {output_path}")
    print(f"🗂️  Absolute path: {output_path.resolve()}")
def run_interactive_mode():
    """Run interactive mode with menu selection"""
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
        return []
  # Go back - return empty List[str]
    elif choice == '7':
        print("Exiting...")
        exit(0)
    all_symbols = []
    if choice == '1' or choice == '4':
        print("Fetching NSE symbols...")
        nse_symbols = fetch_nse_stocks()
        if nse_symbols:
            all_symbols.extend(nse_symbols)
    if choice == '2' or choice == '4':
        print("Fetching US market symbols...")
        us_symbols = fetch_us_stocks()
        dow_symbols = fetch_dow_jones()
        nasdaq_symbols = fetch_nasdaq_100()
        if us_symbols:
            all_symbols.extend(us_symbols)
        if dow_symbols:
            all_symbols.extend(dow_symbols)
        if nasdaq_symbols:
            all_symbols.extend(nasdaq_symbols)
    if choice == '3' or choice == '4':
        print("Fetching FTSE 100 symbols...")
        ftse_symbols = fetch_ftse_100()
        if ftse_symbols:
            all_symbols.extend(ftse_symbols)
    if choice == '5':
        print("\nCustom selection:")
        include_nse = input("Include NSE (India)? (y/n): ").lower() == 'y'
        include_sp500 = input("Include S&P 500 (US)? (y/n): ").lower() == 'y'
        include_dow = input("Include Dow Jones (US)? (y/n): ").lower() == 'y'
        include_nasdaq = input("Include NASDAQ-100 (US)? (y/n): ").lower() == 'y'
        include_ftse = input("Include FTSE 100 (UK)? (y/n): ").lower() == 'y'
        if include_nse:
            nse_symbols = fetch_nse_stocks()
            if nse_symbols:
                all_symbols.extend(nse_symbols)
        if include_sp500:
            us_symbols = fetch_us_stocks()
            if us_symbols:
                all_symbols.extend(us_symbols)
        if include_dow:
            dow_symbols = fetch_dow_jones()
            if dow_symbols:
                all_symbols.extend(dow_symbols)
        if include_nasdaq:
            nasdaq_symbols = fetch_nasdaq_100()
            if nasdaq_symbols:
                all_symbols.extend(nasdaq_symbols)
        if include_ftse:
            ftse_symbols = fetch_ftse_100()
            if ftse_symbols:
                all_symbols.extend(ftse_symbols)
    return all_symbols
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        logging.error(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()

    # Only prompt for input in interactive mode
    if len(sys.argv) == 1:
  # No command line arguments provided
        input("\nPress Enter to exit...")

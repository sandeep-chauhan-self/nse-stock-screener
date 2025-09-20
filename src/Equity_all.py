"""
NSE Equity Symbol Fetcher

Downloads NSE equity symbols from the official NSE archives and saves them
to a text file. Cross-platform compatible with proper path handling.

Usage:
    python Equity_all.py --output my_symbols.txt
    python Equity_all.py --help
"""

import argparse
import requests
import pandas as pd
from io import StringIO
from pathlib import Path

# Import path utilities for cross-platform compatibility
from .common.paths import (
    add_output_argument, resolve_output_path, ensure_dir
)

# Default output file
DEFAULT_OUTPUT_FILE = "nse_only_symbols.txt"

# NSE master equity list URL
NSE_EQUITY_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"

# HTTP headers to mimic browser request
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/114.0.0.0 Safari/537.36"
}


def fetch_nse_symbols():
    """
    Fetch NSE equity symbols from the official NSE archives.
    
    Returns:
        List of symbol strings, or None if fetch failed
    """
    try:
        print("📡 Fetching NSE equity symbols from official archives...")
        print(f"🌐 URL: {NSE_EQUITY_URL}")
        
        # Fetch CSV directly into memory
        response = requests.get(NSE_EQUITY_URL, headers=HEADERS, timeout=30)
        response.raise_for_status()  # Raise exception for bad status codes
        
        print(f"✅ Successfully downloaded data ({len(response.text)} characters)")
        
        # Parse CSV data
        data = StringIO(response.text)
        df = pd.read_csv(data)
        
        print(f"📊 Parsed CSV with {len(df)} rows and columns: {list(df.columns)}")
        
        # Extract symbols
        if "SYMBOL" not in df.columns:
            available_columns = list(df.columns)
            raise ValueError(f"Expected 'SYMBOL' column not found. Available columns: {available_columns}")
        
        symbols = df["SYMBOL"].dropna().astype(str).tolist()
        symbols = [symbol.strip() for symbol in symbols if symbol.strip()]  # Clean up symbols
        
        print(f"🎯 Extracted {len(symbols)} valid symbols")
        return symbols
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error while fetching NSE data: {e}")
        return None
    except pd.errors.EmptyDataError:
        print("❌ Downloaded file is empty or corrupted")
        return None
    except Exception as e:
        print(f"❌ Error processing NSE data: {e}")
        return None


def save_symbols(symbols, output_path):
    """
    Save symbols to a text file.
    
    Args:
        symbols: List of symbol strings
        output_path: Path object where to save the symbols
    """
    try:
        # Ensure parent directory exists
        ensure_dir(output_path.parent)
        
        print(f"💾 Saving {len(symbols)} symbols to: {output_path}")
        
        # Write symbols to file (one per line)
        output_path.write_text('\n'.join(symbols) + '\n', encoding='utf-8')
        
        print(f"✅ NSE symbols saved successfully")
        print(f"📁 File location: {output_path.resolve()}")
        
    except Exception as e:
        print(f"❌ Error saving symbols: {e}")
        raise


def main():
    """Main function to fetch and save NSE symbols"""
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Fetch NSE equity symbols from official archives",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
  %(prog)s --output custom_symbols.txt
  %(prog)s --output /path/to/symbols.txt
        """
    )
    
    # Add output argument
    add_output_argument(parser, DEFAULT_OUTPUT_FILE, "Output file for NSE symbols")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Resolve output path
    output_path = resolve_output_path(args.output, DEFAULT_OUTPUT_FILE, 'data')
    
    print("=" * 60)
    print("🇮🇳 NSE EQUITY SYMBOL FETCHER")
    print("=" * 60)
    print("Downloading official NSE equity symbols from NSE archives")
    print(f"Output will be saved to: {output_path}")
    print("=" * 60)
    
    # Fetch symbols
    symbols = fetch_nse_symbols()
    
    if not symbols:
        print("\n❌ Failed to fetch NSE symbols. Please check your internet connection and try again.")
        return 1
    
    if len(symbols) == 0:
        print("\n⚠️  No symbols found in the downloaded data.")
        return 1
    
    # Save symbols
    try:
        save_symbols(symbols, output_path)
        print(f"\n🎉 Successfully processed {len(symbols)} NSE equity symbols!")
        print("💡 You can now use this file with the NSE Stock Screener.")
        return 0
        
    except Exception as e:
        print(f"\n❌ Failed to save symbols: {e}")
        return 1


if __name__ == "__main__":
    import sys
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

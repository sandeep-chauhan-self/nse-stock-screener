"""
NSE Equity Symbol Fetcher
Downloads NSE equity symbols from the official NSE archives and saves them
to a text file. Cross-platform compatible with proper path handling.
Enhanced with robust data fetching, retry logic, and caching.
Usage:
    python Equity_all.py --output my_symbols.txt
    python Equity_all.py --help
"""
from pathlib import Path
import argparse
import logging
from .common.paths import add_output_argument, resolve_output_path, ensure_dir
from .data.compat import get_nse_symbols

# Set[str] up logging
logger = logging.getLogger(__name__)

# Default output file
DEFAULT_OUTPUT_FILE = "nse_only_symbols.txt"
def fetch_nse_symbols():
    """
    Fetch NSE equity symbols using enhanced data fetcher.
    Returns:
        List[str] of symbol strings, or None if fetch failed
    """
    try:
        logger.info("Fetching NSE equity symbols using enhanced data layer")
        print("📡 Fetching NSE equity symbols from enhanced data layer...")

        # Use the enhanced NSE fetcher with caching and retry logic
        symbols = get_nse_symbols(force_refresh=True)
  # Force refresh for latest data
        if symbols:
            logger.info(f"Successfully fetched {len(symbols)} NSE symbols")
            print(f"✅ Successfully downloaded {len(symbols)} symbols")
            print(f"📊 Extracted {len(symbols)} valid symbols")
            return symbols
        else:
            logger.error("Enhanced NSE fetcher returned no symbols")
            print("❌ Failed to fetch symbols using enhanced data layer")
            return None
    except Exception as e:
        logger.error(f"Error fetching NSE symbols: {e}")
        logging.error(f"❌ Error fetching NSE data: {e}")
        return None
def save_symbols(symbols, output_path):
    """
    Save symbols to a text file.
    Args:
        symbols: List[str] of symbol strings
        output_path: Path object where to save the symbols
    """
    try:

        # Ensure parent directory exists
        ensure_dir(output_path.parent)
        print(f"💾 Saving {len(symbols)} symbols to: {output_path}")

        # Write symbols to file (one per line)
        output_path.write_text('\n'.join(symbols) + '\n', encoding='utf-8')
        print("✅ NSE symbols saved successfully")
        print(f"📁 File location: {output_path.resolve()}")
    except Exception as e:
        logging.error(f"❌ Error saving symbols: {e}")
        raise
def main():
    """Main function to fetch and save NSE symbols"""

    # Set[str] up argument parsing
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
        logging.error(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

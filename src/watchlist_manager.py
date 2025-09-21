#!/usr/bin/env python3
"""
Production-Ready Watchlist Manager for NSE Stock Screener
Handles external configuration, validation, and dynamic updates
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class WatchlistManager:
    """
    Production-ready watchlist management with external configuration
    """

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize watchlist manager

        Args:
            config_file: Path to watchlists configuration file
        """
        self.project_root = Path(__file__).resolve().parent.parent

        if config_file:
            self.config_file = Path(config_file)
        else:
            self.config_file = self.project_root / "config" / "watchlists.json"

        self.watchlists = {}
        self.search_aliases = {}
        self.config = {}

        self._load_configuration()

    def _load_configuration(self) -> bool:
        """
        Load watchlists from external configuration file

        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            if not self.config_file.exists():
                logger.warning(f"Configuration file not found: {self.config_file}")
                self._create_default_config()
                return False

            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.watchlists = data.get('watchlists', {})
            self.search_aliases = data.get('search_aliases', {})
            self.config = data.get('config', {})

            logger.info(f"Loaded {len(self.watchlists)} watchlists from {self.config_file}")
            return True

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file: {e}")
            return False
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return False

    def _create_default_config(self) -> None:
        """Create default configuration if none exists"""
        default_config = {
            "watchlists": {
                "default": {
                    "name": "Default Watchlist",
                    "description": "Basic stock selection",
                    "symbols": ["RELIANCE.NS", "TCS.NS", "INFY.NS"],
                    "sector": "Mixed"
                }
            },
            "search_aliases": {
                "REL": "RELIANCE.NS",
                "TCS": "TCS.NS",
                "INFY": "INFY.NS"
            },
            "config": {
                "auto_add_ns_suffix": True,
                "validate_symbols": True,
                "max_symbols_per_analysis": 20,
                "default_watchlist": "default",
                "last_updated": datetime.now().strftime("%Y-%m-%d")
            }
        }

        # Ensure config directory exists
        self.config_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2)
            logger.info(f"Created default configuration: {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to create default configuration: {e}")

    def get_watchlists(self) -> Dict[str, Dict]:
        """
        Get all available watchlists

        Returns:
            Dict containing all watchlists
        """
        return self.watchlists.copy()

    def get_watchlist(self, watchlist_id: str) -> Optional[Dict]:
        """
        Get specific watchlist by ID

        Args:
            watchlist_id: ID of the watchlist to retrieve

        Returns:
            Dictionary containing watchlist data or None if not found
        """
        return self.watchlists.get(watchlist_id)

    def get_watchlist_symbols(self, watchlist_id: str) -> List[str]:
        """
        Get symbols from a specific watchlist

        Args:
            watchlist_id: ID of the watchlist

        Returns:
            List of stock symbols
        """
        watchlist = self.get_watchlist(watchlist_id)
        if watchlist:
            return watchlist.get('symbols', [])
        return []

    def list_watchlists(self) -> List[Tuple[str, str, str]]:
        """
        Get list of all watchlists with basic info

        Returns:
            List of tuples: (id, name, description)
        """
        result = []
        for wl_id, wl_data in self.watchlists.items():
            result.append((
                wl_id,
                wl_data.get('name', wl_id),
                wl_data.get('description', 'No description')
            ))
        return result

    def search_symbol(self, search_term: str) -> List[str]:
        """
        Search for symbols using aliases and partial matching

        Args:
            search_term: Term to search for

        Returns:
            List of matching symbols
        """
        search_term = search_term.upper().strip()
        matches = []

        # Check direct alias match
        if search_term in self.search_aliases:
            matches.append(self.search_aliases[search_term])

        # Check partial matches in aliases
        for alias, symbol in self.search_aliases.items():
            if search_term in alias and symbol not in matches:
                matches.append(symbol)

        # Check partial matches in watchlist symbols
        for watchlist in self.watchlists.values():
            for symbol in watchlist.get('symbols', []):
                symbol_base = symbol.replace('.NS', '')
                if search_term in symbol_base and symbol not in matches:
                    matches.append(symbol)

        return matches

    def validate_symbols(self, symbols: List[str]) -> Tuple[List[str], List[str]]:
        """
        Validate and normalize stock symbols

        Args:
            symbols: List of symbols to validate

        Returns:
            Tuple of (valid_symbols, invalid_symbols)
        """
        valid = []
        invalid = []

        for symbol in symbols:
            symbol = symbol.strip().upper()

            # Add .NS suffix if needed and configured
            if self.config.get('auto_add_ns_suffix', True):
                if not symbol.endswith('.NS'):
                    symbol += '.NS'

            # Basic validation (you can extend this with actual API checks)
            if len(symbol) >= 3 and symbol.replace('.NS', '').isalpha():
                valid.append(symbol)
            else:
                invalid.append(symbol)

        return valid, invalid

    def add_watchlist(self, watchlist_id: str, name: str, description: str,
                     symbols: List[str], sector: str = "Mixed") -> bool:
        """
        Add a new watchlist

        Args:
            watchlist_id: Unique ID for the watchlist
            name: Display name
            description: Description
            symbols: List of stock symbols
            sector: Sector classification

        Returns:
            bool: True if added successfully
        """
        valid_symbols, invalid_symbols = self.validate_symbols(symbols)

        if invalid_symbols:
            logger.warning(f"Invalid symbols ignored: {invalid_symbols}")

        if not valid_symbols:
            logger.error("No valid symbols provided")
            return False

        self.watchlists[watchlist_id] = {
            "name": name,
            "description": description,
            "symbols": valid_symbols,
            "sector": sector,
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        return self._save_configuration()

    def remove_watchlist(self, watchlist_id: str) -> bool:
        """
        Remove a watchlist

        Args:
            watchlist_id: ID of watchlist to remove

        Returns:
            bool: True if removed successfully
        """
        if watchlist_id in self.watchlists:
            del self.watchlists[watchlist_id]
            return self._save_configuration()
        return False

    def update_watchlist_symbols(self, watchlist_id: str, symbols: List[str]) -> bool:
        """
        Update symbols in an existing watchlist

        Args:
            watchlist_id: ID of watchlist to update
            symbols: New list of symbols

        Returns:
            bool: True if updated successfully
        """
        if watchlist_id not in self.watchlists:
            return False

        valid_symbols, invalid_symbols = self.validate_symbols(symbols)

        if invalid_symbols:
            logger.warning(f"Invalid symbols ignored: {invalid_symbols}")

        if valid_symbols:
            self.watchlists[watchlist_id]['symbols'] = valid_symbols
            self.watchlists[watchlist_id]['updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return self._save_configuration()

        return False

    def _save_configuration(self) -> bool:
        """
        Save current configuration to file

        Returns:
            bool: True if saved successfully
        """
        try:
            # Update last_updated timestamp
            self.config['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            data = {
                "watchlists": self.watchlists,
                "search_aliases": self.search_aliases,
                "config": self.config
            }

            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"Configuration saved to {self.config_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False

    def get_stats(self) -> Dict:
        """
        Get statistics about watchlists

        Returns:
            Dictionary with statistics
        """
        total_watchlists = len(self.watchlists)
        total_symbols = sum(len(wl.get('symbols', [])) for wl in self.watchlists.values())
        unique_symbols = len(set(
            symbol for wl in self.watchlists.values()
            for symbol in wl.get('symbols', [])
        ))

        sectors = {}
        for wl in self.watchlists.values():
            sector = wl.get('sector', 'Unknown')
            sectors[sector] = sectors.get(sector, 0) + 1

        return {
            "total_watchlists": total_watchlists,
            "total_symbols": total_symbols,
            "unique_symbols": unique_symbols,
            "sectors": sectors,
            "search_aliases": len(self.search_aliases),
            "last_updated": self.config.get('last_updated', 'Unknown')
        }

# Usage example and testing
if __name__ == "__main__":
    # Set up basic logging
    logging.basicConfig(level=logging.INFO)

    # Initialize manager
    wm = WatchlistManager()

    # Test functionality
    print("📊 Watchlist Manager Test")
    print("=" * 40)

    # List all watchlists
    watchlists = wm.list_watchlists()
    print(f"📋 Available Watchlists ({len(watchlists)}):")
    for wl_id, name, desc in watchlists:
        symbols = wm.get_watchlist_symbols(wl_id)
        print(f"  • {name} ({wl_id}): {len(symbols)} symbols")
        print(f"    {desc}")
        print()

    # Test search
    print("🔍 Search Test:")
    search_results = wm.search_symbol("TCS")
    print(f"  Search 'TCS': {search_results}")

    search_results = wm.search_symbol("HDFC")
    print(f"  Search 'HDFC': {search_results}")

    # Show statistics
    print("\n📈 Statistics:")
    stats = wm.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
"""
Corporate Actions module for NSE Stock Screener.

This module handles fetching, storing, and applying corporate actions
(splits, dividends, bonuses) to maintain both raw and adjusted OHLCV series.
"""

import logging
import sqlite3
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import requests
from bs4 import BeautifulSoup
import yfinance as yf

from ..common.config import get_config


logger = logging.getLogger(__name__)

# Constants for repeated literals
DIVIDENDS_COLUMN = 'Dividends'
STOCK_SPLITS_COLUMN = 'Stock Splits'
NS_SUFFIX = '.NS'


@dataclass
class CorporateAction:
    """Corporate action data structure."""
    symbol: str
    ex_date: date
    action_type: str  # 'split', 'dividend', 'bonus', 'rights'
    ratio: Optional[float]  # Split/bonus ratio (e.g., 2.0 for 1:2 split)
    amount: Optional[float]  # Dividend amount per share
    adjustment_factor: float  # Price adjustment factor
    record_date: Optional[date] = None
    announcement_date: Optional[date] = None
    details: Optional[str] = None


class CorporateActionsManager:
    """
    Manages corporate actions data for NSE stocks.

    Fetches corporate actions from multiple sources and applies adjustments
    to maintain both raw and adjusted price series.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize corporate actions manager.

        Args:
            db_path: Path to SQLite database for storing corporate actions
        """
        self.config = get_config().config.data
        self.db_path = db_path or "data/cache/corporate_actions.db"
        self.session = requests.Session()

        # Set headers for NSE scraping
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })

        self._init_database()

    def _init_database(self) -> None:
        """Initialize database for corporate actions."""
        import os
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS corporate_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    ex_date TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    ratio REAL,
                    amount REAL,
                    adjustment_factor REAL NOT NULL,
                    record_date TEXT,
                    announcement_date TEXT,
                    details TEXT,
                    source TEXT DEFAULT 'manual',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, ex_date, action_type, ratio, amount)
                )
            """)

            # Create index for efficient queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_symbol_ex_date
                ON corporate_actions(symbol, ex_date)
            """)

    def fetch_corporate_actions_nse(self, symbol: str,
                                   start_date: Optional[date] = None,
                                   end_date: Optional[date] = None) -> List[CorporateAction]:
        """
        Fetch corporate actions from NSE website.

        Args:
            symbol: Stock symbol
            start_date: Start date for fetching actions
            end_date: End date for fetching actions

        Returns:
            List of corporate actions
        """
        actions = []

        if start_date is None:
            start_date = date.today() - timedelta(days=365*2)  # 2 years back
        if end_date is None:
            end_date = date.today()

        try:
            # NSE Corporate Actions URL
            url = "https://www.nseindia.com/companies-listing/corporate-filings-actions"

            # Try to get corporate actions data
            # Note: NSE API structure changes frequently, this is a template
            params = {
                'symbol': symbol.upper(),
                'from_date': start_date.strftime('%d-%m-%Y'),
                'to_date': end_date.strftime('%d-%m-%Y')
            }

            response = self.session.get(url, params=params, timeout=30)

            if response.status_code == 200:
                # Parse the response - this would need to be adapted based on actual NSE API
                actions.extend(self._parse_nse_corporate_actions(response.text, symbol))

        except Exception as e:
            logger.warning(f"Failed to fetch NSE corporate actions for {symbol}: {e}")

        return actions

    def _parse_nse_corporate_actions(self, html_content: str, symbol: str) -> List[CorporateAction]:
        """Parse NSE corporate actions from HTML."""
        actions = []

        try:
            soup = BeautifulSoup(html_content, 'html.parser')

            # This would need to be customized based on actual NSE HTML structure
            # The following is a template/placeholder

            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')[1:]  # Skip header

                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 4:
                        try:
                            ex_date_str = cols[0].text.strip()
                            action_type = cols[1].text.strip().lower()
                            details = cols[2].text.strip()

                            ex_date = datetime.strptime(ex_date_str, '%d-%m-%Y').date()

                            # Parse action details to extract ratio/amount
                            ratio, amount, adj_factor = self._parse_action_details(action_type, details)

                            action = CorporateAction(
                                symbol=symbol,
                                ex_date=ex_date,
                                action_type=action_type,
                                ratio=ratio,
                                amount=amount,
                                adjustment_factor=adj_factor,
                                details=details
                            )

                            actions.append(action)

                        except Exception as e:
                            logger.debug(f"Failed to parse corporate action row: {e}")
                            continue

        except Exception as e:
            logger.warning(f"Failed to parse NSE corporate actions HTML: {e}")

        return actions

    def _process_dividend_action(self, row: pd.Series, symbol: str, action_date: date) -> Optional[CorporateAction]:
        """Process dividend action from Yahoo Finance data."""
        if DIVIDENDS_COLUMN in row and row[DIVIDENDS_COLUMN] > 0:
            return CorporateAction(
                symbol=symbol.replace(NS_SUFFIX, ''),
                ex_date=action_date,
                action_type='dividend',
                ratio=None,
                amount=float(row[DIVIDENDS_COLUMN]),
                adjustment_factor=1.0,  # Dividends don't affect price adjustment in Yahoo
                details=f"Dividend: ₹{row[DIVIDENDS_COLUMN]}"
            )
        return None

    def _process_split_action(self, row: pd.Series, symbol: str, action_date: date) -> Optional[CorporateAction]:
        """Process stock split action from Yahoo Finance data."""
        if (STOCK_SPLITS_COLUMN in row and
            row[STOCK_SPLITS_COLUMN] != 0 and
            row[STOCK_SPLITS_COLUMN] != 1):
            split_ratio = float(row[STOCK_SPLITS_COLUMN])
            return CorporateAction(
                symbol=symbol.replace(NS_SUFFIX, ''),
                ex_date=action_date,
                action_type='split',
                ratio=split_ratio,
                amount=None,
                adjustment_factor=1.0 / split_ratio,
                details=f"Stock Split: {split_ratio}:1"
            )
        return None

    def _get_yahoo_actions_data(self, ticker, start_date: Optional[date], end_date: Optional[date]):
        """Get actions data from Yahoo Finance ticker."""
        if start_date and end_date:
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            return ticker.actions.loc[start_str:end_str]
        return ticker.actions

    def fetch_corporate_actions_yahoo(self, symbol: str,
                                     start_date: Optional[date] = None,
                                     end_date: Optional[date] = None) -> List[CorporateAction]:
        """
        Fetch corporate actions from Yahoo Finance.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date

        Returns:
            List of corporate actions
        """
        actions = []

        try:
            # Ensure symbol has .NS suffix
            if not symbol.endswith(NS_SUFFIX):
                symbol = f"{symbol}{NS_SUFFIX}"

            ticker = yf.Ticker(symbol)

            # Get actions (dividends and splits)
            actions_data = self._get_yahoo_actions_data(ticker, start_date, end_date)

            if not actions_data.empty:
                for date_idx, row in actions_data.iterrows():
                    action_date = date_idx.date()

                    # Process dividend actions
                    dividend_action = self._process_dividend_action(row, symbol, action_date)
                    if dividend_action:
                        actions.append(dividend_action)

                    # Process split actions
                    split_action = self._process_split_action(row, symbol, action_date)
                    if split_action:
                        actions.append(split_action)

        except Exception as e:
            logger.warning(f"Failed to fetch Yahoo corporate actions for {symbol}: {e}")

        return actions

    def _parse_action_details(self, action_type: str, details: str) -> Tuple[Optional[float], Optional[float], float]:
        """
        Parse action details to extract ratio, amount, and adjustment factor.

        Args:
            action_type: Type of corporate action
            details: Details string

        Returns:
            Tuple of (ratio, amount, adjustment_factor)
        """
        ratio = None
        amount = None
        adjustment_factor = 1.0

        try:
            if 'split' in action_type:
                # Parse split ratio (e.g., "1:2", "2:1")
                import re
                split_match = re.search(r'(\d+):(\d+)', details)
                if split_match:
                    old_shares = float(split_match.group(1))
                    new_shares = float(split_match.group(2))
                    ratio = new_shares / old_shares
                    adjustment_factor = old_shares / new_shares

            elif 'dividend' in action_type:
                # Parse dividend amount
                import re
                amount_match = re.search(r'₹?(\d+\.?\d*)', details)
                if amount_match:
                    amount = float(amount_match.group(1))
                    # Dividend adjustment factor (simplified)
                    adjustment_factor = 1.0

            elif 'bonus' in action_type:
                # Parse bonus ratio (e.g., "1:1", "1:2")
                import re
                bonus_match = re.search(r'(\d+):(\d+)', details)
                if bonus_match:
                    bonus_shares = float(bonus_match.group(1))
                    held_shares = float(bonus_match.group(2))
                    ratio = (held_shares + bonus_shares) / held_shares
                    adjustment_factor = held_shares / (held_shares + bonus_shares)

        except Exception as e:
            logger.debug(f"Failed to parse action details '{details}': {e}")

        return ratio, amount, adjustment_factor

    def save_corporate_actions(self, actions: List[CorporateAction], source: str = 'api') -> int:
        """
        Save corporate actions to database.

        Args:
            actions: List of corporate actions
            source: Data source identifier

        Returns:
            Number of actions saved
        """
        saved_count = 0

        with sqlite3.connect(self.db_path) as conn:
            for action in actions:
                try:
                    conn.execute("""
                        INSERT OR REPLACE INTO corporate_actions
                        (symbol, ex_date, action_type, ratio, amount, adjustment_factor,
                         record_date, announcement_date, details, source, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        action.symbol,
                        action.ex_date.isoformat(),
                        action.action_type,
                        action.ratio,
                        action.amount,
                        action.adjustment_factor,
                        action.record_date.isoformat() if action.record_date else None,
                        action.announcement_date.isoformat() if action.announcement_date else None,
                        action.details,
                        source,
                        datetime.now().isoformat()
                    ))
                    saved_count += 1

                except Exception as e:
                    logger.warning(f"Failed to save corporate action for {action.symbol}: {e}")

            conn.commit()

        logger.info(f"Saved {saved_count} corporate actions from {source}")
        return saved_count

    def get_corporate_actions(self, symbol: str,
                             start_date: Optional[date] = None,
                             end_date: Optional[date] = None) -> List[CorporateAction]:
        """
        Get corporate actions for a symbol from database.

        Args:
            symbol: Stock symbol
            start_date: Filter start date
            end_date: Filter end date

        Returns:
            List of corporate actions
        """
        actions = []

        query = "SELECT * FROM corporate_actions WHERE symbol = ?"
        params = [symbol]

        if start_date:
            query += " AND ex_date >= ?"
            params.append(start_date.isoformat())

        if end_date:
            query += " AND ex_date <= ?"
            params.append(end_date.isoformat())

        query += " ORDER BY ex_date ASC"

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)

            for row in cursor.fetchall():
                action = CorporateAction(
                    symbol=row[1],
                    ex_date=datetime.fromisoformat(row[2]).date(),
                    action_type=row[3],
                    ratio=row[4],
                    amount=row[5],
                    adjustment_factor=row[6],
                    record_date=datetime.fromisoformat(row[7]).date() if row[7] else None,
                    announcement_date=datetime.fromisoformat(row[8]).date() if row[8] else None,
                    details=row[9]
                )
                actions.append(action)

        return actions

    def update_corporate_actions(self, symbol: str, force_refresh: bool = False) -> int:
        """
        Update corporate actions for a symbol.

        Args:
            symbol: Stock symbol
            force_refresh: Force refresh even if recently updated

        Returns:
            Number of new actions found
        """
        # Check if we need to update
        if not force_refresh:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT MAX(updated_at) FROM corporate_actions WHERE symbol = ?
                """, (symbol,))

                last_update = cursor.fetchone()[0]
                if last_update:
                    last_update_dt = datetime.fromisoformat(last_update)
                    if datetime.now() - last_update_dt < timedelta(days=1):
                        logger.debug(f"Corporate actions for {symbol} updated recently, skipping")
                        return 0

        # Fetch from multiple sources
        all_actions = []

        # Try Yahoo Finance first (more reliable)
        yahoo_actions = self.fetch_corporate_actions_yahoo(symbol)
        all_actions.extend(yahoo_actions)

        # Try NSE as backup
        nse_actions = self.fetch_corporate_actions_nse(symbol)
        all_actions.extend(nse_actions)

        # Remove duplicates
        unique_actions = self._deduplicate_actions(all_actions)

        # Save to database
        new_count = self.save_corporate_actions(unique_actions, 'auto_update')

        return new_count

    def _deduplicate_actions(self, actions: List[CorporateAction]) -> List[CorporateAction]:
        """Remove duplicate corporate actions."""
        seen = set()
        unique_actions = []

        for action in actions:
            # Create a key for deduplication
            key = (action.symbol, action.ex_date, action.action_type, action.ratio, action.amount)

            if key not in seen:
                seen.add(key)
                unique_actions.append(action)

        return unique_actions

    def _get_price_columns(self) -> List[str]:
        """Get list of price columns to adjust."""
        return ['Open', 'High', 'Low', 'Close']

    def _apply_price_adjustment(self, data: pd.DataFrame, mask: pd.Series,
                               adjustment_factor: float) -> None:
        """Apply price adjustment to OHLC columns."""
        for col in self._get_price_columns():
            if col in data.columns:
                data.loc[mask, col] *= adjustment_factor

    def _apply_volume_adjustment(self, data: pd.DataFrame, mask: pd.Series,
                                adjustment_factor: float) -> None:
        """Apply volume adjustment (inverse of price adjustment)."""
        if 'Volume' in data.columns:
            data.loc[mask, 'Volume'] /= adjustment_factor

    def _apply_dividend_adjustment(self, data: pd.DataFrame, mask: pd.Series,
                                  amount: float) -> None:
        """Apply dividend adjustment to close price."""
        if 'Close' in data.columns:
            data.loc[mask, 'Close'] -= amount

    def apply_adjustments(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Apply corporate action adjustments to price data.

        Args:
            data: OHLCV DataFrame with DatetimeIndex
            symbol: Stock symbol

        Returns:
            Adjusted DataFrame
        """
        if data.empty:
            return data

        # Get corporate actions for the symbol
        start_date = data.index.min().date()
        end_date = data.index.max().date()

        actions = self.get_corporate_actions(symbol, start_date, end_date)

        if not actions:
            return data  # No adjustments needed

        adjusted_data = data.copy()

        # Apply adjustments in chronological order
        for action in sorted(actions, key=lambda x: x.ex_date):
            ex_date = pd.Timestamp(action.ex_date)

            # Apply adjustment to all dates before ex-date
            mask = adjusted_data.index < ex_date

            if action.action_type in ['split', 'bonus']:
                self._apply_price_adjustment(adjusted_data, mask, action.adjustment_factor)
                self._apply_volume_adjustment(adjusted_data, mask, action.adjustment_factor)
            elif action.action_type == 'dividend' and action.amount:
                self._apply_dividend_adjustment(adjusted_data, mask, action.amount)

        return adjusted_data

    def get_adjustment_summary(self, symbol: str) -> Dict[str, Any]:
        """Get summary of adjustments for a symbol."""
        actions = self.get_corporate_actions(symbol)

        summary = {
            'symbol': symbol,
            'total_actions': len(actions),
            'splits': len([a for a in actions if a.action_type == 'split']),
            'bonuses': len([a for a in actions if a.action_type == 'bonus']),
            'dividends': len([a for a in actions if a.action_type == 'dividend']),
            'date_range': {
                'earliest': min([a.ex_date for a in actions]).isoformat() if actions else None,
                'latest': max([a.ex_date for a in actions]).isoformat() if actions else None
            },
            'actions': [
                {
                    'ex_date': a.ex_date.isoformat(),
                    'type': a.action_type,
                    'ratio': a.ratio,
                    'amount': a.amount,
                    'adjustment_factor': a.adjustment_factor,
                    'details': a.details
                }
                for a in actions
            ]
        }

        return summary


# Create default instance
default_corporate_actions_manager = CorporateActionsManager()
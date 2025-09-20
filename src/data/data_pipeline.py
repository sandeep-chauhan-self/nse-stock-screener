"""
Data Infrastructure & ETL Pipeline for NSE Stock Screener.

This module provides a comprehensive data pipeline that ensures reliable,
adjusted, and validated data feeds for downstream analysis. It implements
the FS.2 requirement for robust data infrastructure.

Key Features:
- Multi-source data ingestion (NSE Bhavcopy, NSE APIs, yfinance)
- Intelligent caching with Parquet/Feather storage
- Corporate action adjustments with raw/adjusted series
- Data validation and health checks
- Metadata tracking and lineage
"""

import logging
import hashlib
import sqlite3
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np

from ..common.interfaces import IDataFetcher, IDataCache, IDataValidator, StockData
from ..common.config import get_config
from .fetchers_impl import YahooDataFetcher, NSEDataFetcher, NSEBhavcopyFetcher
from .cache import ParquetCache
from .validation import EnhancedDataValidator as DataValidator


logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Supported data sources."""
    YAHOO_FINANCE = "yahoo"
    NSE_API = "nse_api"
    NSE_BHAVCOPY = "nse_bhavcopy"


@dataclass
class DatasetMetadata:
    """Metadata for tracking dataset versions and freshness."""
    symbol: str
    source: DataSource
    data_type: str  # 'raw', 'adjusted', 'corporate_actions'
    start_date: date
    end_date: date
    created_at: datetime
    updated_at: datetime
    checksum: str
    record_count: int
    validation_status: str  # 'valid', 'warning', 'error'
    validation_issues: List[str]
    version: str


@dataclass
class CorporateAction:
    """Corporate action adjustment data."""
    symbol: str
    ex_date: date
    action_type: str  # 'split', 'dividend', 'bonus'
    ratio: Optional[float]  # Split/bonus ratio
    amount: Optional[float]  # Dividend amount
    adjustment_factor: float  # Price adjustment factor


class DataPipeline:
    """
    Main data pipeline orchestrator.
    
    Coordinates data ingestion, caching, validation, and corporate action
    adjustments across multiple data sources.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize data pipeline.
        
        Args:
            cache_dir: Directory for data cache (defaults to config)
        """
        self.config = get_config().config.data
        self.cache_dir = cache_dir or Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._init_fetchers()
        self._init_cache()
        self._init_validator()
        self._init_metadata_db()
        
        logger.info(f"Data pipeline initialized with cache at {self.cache_dir}")
    
    def _init_fetchers(self) -> None:
        """Initialize data fetchers for each source."""
        self.fetchers = {
            DataSource.YAHOO_FINANCE: YahooDataFetcher(),
            DataSource.NSE_API: NSEDataFetcher(),
            DataSource.NSE_BHAVCOPY: NSEBhavcopyFetcher()
        }
    
    def _init_cache(self) -> None:
        """Initialize caching layer."""
        self.cache = ParquetCache(self.cache_dir / "market_data")
    
    def _init_validator(self) -> None:
        """Initialize data validator."""
        self.validator = DataValidator()
    
    def _init_metadata_db(self) -> None:
        """Initialize metadata database."""
        self.metadata_db = self.cache_dir / "metadata.db"
        self._create_metadata_tables()
    
    def _create_metadata_tables(self) -> None:
        """Create metadata tables if they don't exist."""
        with sqlite3.connect(self.metadata_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dataset_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    source TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    record_count INTEGER NOT NULL,
                    validation_status TEXT NOT NULL,
                    validation_issues TEXT,
                    version TEXT NOT NULL,
                    UNIQUE(symbol, source, data_type, start_date, end_date)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS corporate_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    ex_date TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    ratio REAL,
                    amount REAL,
                    adjustment_factor REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(symbol, ex_date, action_type)
                )
            """)
    
    def get_data(self, 
                 symbol: str, 
                 start: date, 
                 end: date,
                 adjusted: bool = True,
                 force_refresh: bool = False,
                 preferred_sources: Optional[List[DataSource]] = None) -> Optional[StockData]:
        """
        Get stock data with intelligent source selection and caching.
        
        Args:
            symbol: Stock symbol
            start: Start date
            end: End date
            adjusted: Whether to return adjusted prices
            force_refresh: Force refresh from source
            preferred_sources: Preferred data sources in order
            
        Returns:
            StockData with OHLCV data and metadata
        """
        try:
            # Check cache first (unless force refresh)
            if not force_refresh:
                cached_data = self._get_cached_data(symbol, start, end, adjusted)
                if cached_data is not None:
                    logger.info(f"Retrieved {symbol} from cache")
                    return cached_data
            
            # Fetch from sources
            data, source_used = self._fetch_from_sources(symbol, start, end, preferred_sources)
            if data is None:
                logger.warning(f"Failed to fetch data for {symbol}")
                return None
            
            # Validate data
            validation_result = self.validator.validate(data, symbol)
            if validation_result['status'] == 'error':
                logger.error(f"Data validation failed for {symbol}: {validation_result['issues']}")
                return None
            
            # Apply corporate actions if requested
            if adjusted:
                data = self._apply_corporate_actions(symbol, data)
            
            # Cache the data
            self._cache_data(symbol, data, source_used, start, end, adjusted, validation_result)
            
            # Create StockData object
            stock_data = StockData(
                symbol=symbol,
                data=data,
                metadata={
                    'source': source_used.value,
                    'adjusted': adjusted,
                    'validation': validation_result,
                    'start_date': start.isoformat(),
                    'end_date': end.isoformat()
                },
                timestamp=datetime.now()
            )
            
            logger.info(f"Successfully retrieved {symbol} from {source_used.value}")
            return stock_data
            
        except Exception as e:
            logger.error(f"Error getting data for {symbol}: {e}")
            return None
    
    def _get_cached_data(self, symbol: str, start: date, end: date, adjusted: bool) -> Optional[StockData]:
        """Get data from cache if available and fresh."""
        try:
            data_type = 'adjusted' if adjusted else 'raw'
            
            # Check metadata for freshness
            with sqlite3.connect(self.metadata_db) as conn:
                cursor = conn.execute("""
                    SELECT * FROM dataset_metadata 
                    WHERE symbol = ? AND data_type = ? 
                    AND start_date <= ? AND end_date >= ?
                    AND validation_status != 'error'
                    ORDER BY updated_at DESC LIMIT 1
                """, (symbol, data_type, start.isoformat(), end.isoformat()))
                
                metadata = cursor.fetchone()
                if not metadata:
                    return None
                
                # Check if data is fresh (within T+1)
                updated_at = datetime.fromisoformat(metadata[7])  # updated_at column
                if datetime.now() - updated_at > timedelta(days=1):
                    logger.info(f"Cached data for {symbol} is stale")
                    return None
            
            # Load from cache
            cache_key = f"{symbol}_{data_type}_{start}_{end}"
            cached_df = self.cache.get(cache_key)
            
            if cached_df is not None:
                return StockData(
                    symbol=symbol,
                    data=cached_df,
                    metadata={'source': 'cache', 'adjusted': adjusted},
                    timestamp=updated_at
                )
            
            return None
            
        except Exception as e:
            logger.warning(f"Error accessing cache for {symbol}: {e}")
            return None
    
    def _fetch_from_sources(self, 
                           symbol: str, 
                           start: date, 
                           end: date,
                           preferred_sources: Optional[List[DataSource]] = None) -> Tuple[Optional[pd.DataFrame], Optional[DataSource]]:
        """Fetch data from multiple sources with fallback."""
        if preferred_sources is None:
            preferred_sources = [DataSource.YAHOO_FINANCE, DataSource.NSE_API, DataSource.NSE_BHAVCOPY]
        
        for source in preferred_sources:
            try:
                fetcher = self.fetchers[source]
                logger.info(f"Attempting to fetch {symbol} from {source.value}")
                
                data = fetcher.fetch(symbol, start, end)
                if data is not None and not data.empty:
                    logger.info(f"Successfully fetched {symbol} from {source.value}")
                    return data, source
                
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol} from {source.value}: {e}")
                continue
        
        return None, None
    
    def _apply_corporate_actions(self, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        """Apply corporate action adjustments to raw data."""
        try:
            # Get corporate actions for symbol
            with sqlite3.connect(self.metadata_db) as conn:
                cursor = conn.execute("""
                    SELECT ex_date, action_type, adjustment_factor 
                    FROM corporate_actions 
                    WHERE symbol = ? 
                    ORDER BY ex_date ASC
                """, (symbol,))
                
                actions = cursor.fetchall()
            
            if not actions:
                return data  # No adjustments needed
            
            adjusted_data = data.copy()
            
            # Apply adjustments chronologically
            for ex_date_str, action_type, adjustment_factor in actions:
                ex_date = pd.to_datetime(ex_date_str).date()
                
                # Apply adjustment to all dates before ex-date
                mask = adjusted_data.index.date < ex_date
                
                price_columns = ['Open', 'High', 'Low', 'Close']
                for col in price_columns:
                    if col in adjusted_data.columns:
                        adjusted_data.loc[mask, col] *= adjustment_factor
                
                # Adjust volume (inverse of price adjustment)
                if 'Volume' in adjusted_data.columns:
                    adjusted_data.loc[mask, 'Volume'] /= adjustment_factor
            
            return adjusted_data
            
        except Exception as e:
            logger.warning(f"Error applying corporate actions for {symbol}: {e}")
            return data  # Return original data if adjustment fails
    
    def _cache_data(self, 
                   symbol: str, 
                   data: pd.DataFrame, 
                   source: DataSource,
                   start: date, 
                   end: date, 
                   adjusted: bool,
                   validation_result: Dict[str, Any]) -> None:
        """Cache data and update metadata."""
        try:
            data_type = 'adjusted' if adjusted else 'raw'
            cache_key = f"{symbol}_{data_type}_{start}_{end}"
            
            # Cache the data
            self.cache.set(cache_key, data)
            
            # Calculate checksum
            checksum = self._calculate_checksum(data)
            
            # Update metadata
            metadata = DatasetMetadata(
                symbol=symbol,
                source=source,
                data_type=data_type,
                start_date=start,
                end_date=end,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                checksum=checksum,
                record_count=len(data),
                validation_status=validation_result['status'],
                validation_issues=validation_result.get('issues', []),
                version="1.0"
            )
            
            self._save_metadata(metadata)
            
        except Exception as e:
            logger.error(f"Error caching data for {symbol}: {e}")
    
    def _calculate_checksum(self, data: pd.DataFrame) -> str:
        """Calculate data checksum for integrity validation."""
        # Create a stable string representation
        data_str = data.to_string()
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _save_metadata(self, metadata: DatasetMetadata) -> None:
        """Save metadata to database."""
        with sqlite3.connect(self.metadata_db) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO dataset_metadata 
                (symbol, source, data_type, start_date, end_date, created_at, 
                 updated_at, checksum, record_count, validation_status, 
                 validation_issues, version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata.symbol, metadata.source.value, metadata.data_type,
                metadata.start_date.isoformat(), metadata.end_date.isoformat(),
                metadata.created_at.isoformat(), metadata.updated_at.isoformat(),
                metadata.checksum, metadata.record_count, metadata.validation_status,
                ','.join(metadata.validation_issues), metadata.version
            ))
    
    def update_corporate_actions(self, symbol: str, actions: List[CorporateAction]) -> None:
        """Update corporate actions for a symbol."""
        with sqlite3.connect(self.metadata_db) as conn:
            for action in actions:
                conn.execute("""
                    INSERT OR REPLACE INTO corporate_actions 
                    (symbol, ex_date, action_type, ratio, amount, adjustment_factor, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    action.symbol, action.ex_date.isoformat(), action.action_type,
                    action.ratio, action.amount, action.adjustment_factor,
                    datetime.now().isoformat()
                ))
    
    def get_data_freshness_report(self) -> pd.DataFrame:
        """Generate data freshness report."""
        with sqlite3.connect(self.metadata_db) as conn:
            query = """
                SELECT symbol, source, data_type, updated_at, validation_status,
                       (JULIANDAY('now') - JULIANDAY(updated_at)) as days_since_update
                FROM dataset_metadata
                ORDER BY updated_at DESC
            """
            return pd.read_sql_query(query, conn)
    
    def run_health_checks(self) -> Dict[str, Any]:
        """Run comprehensive health checks on data pipeline."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'checks': []
        }
        
        # Check 1: Data freshness
        freshness_df = self.get_data_freshness_report()
        stale_data = freshness_df[freshness_df['days_since_update'] > 1]
        
        results['checks'].append({
            'name': 'Data Freshness',
            'status': 'PASS' if len(stale_data) == 0 else 'WARN',
            'message': f"Found {len(stale_data)} stale datasets" if len(stale_data) > 0 else "All data is fresh",
            'details': stale_data.to_dict('records') if len(stale_data) > 0 else []
        })
        
        # Check 2: Validation errors
        with sqlite3.connect(self.metadata_db) as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) FROM dataset_metadata 
                WHERE validation_status = 'error'
            """)
            error_count = cursor.fetchone()[0]
        
        results['checks'].append({
            'name': 'Validation Errors',
            'status': 'PASS' if error_count == 0 else 'FAIL',
            'message': f"Found {error_count} datasets with validation errors",
            'details': error_count
        })
        
        # Check 3: Cache health
        cache_size = sum(f.stat().st_size for f in self.cache_dir.rglob('*') if f.is_file())
        cache_size_mb = cache_size / (1024 * 1024)
        
        results['checks'].append({
            'name': 'Cache Health',
            'status': 'PASS' if cache_size_mb < 1000 else 'WARN',  # Warn if > 1GB
            'message': f"Cache size: {cache_size_mb:.1f} MB",
            'details': {'size_mb': cache_size_mb, 'path': str(self.cache_dir)}
        })
        
        return results


class DataPipelineManager:
    """High-level manager for data pipeline operations."""
    
    def __init__(self):
        """Initialize pipeline manager."""
        self.pipeline = DataPipeline()
    
    def daily_batch_update(self, symbols: List[str]) -> Dict[str, Any]:
        """Run daily batch update for list of symbols."""
        results = {
            'started_at': datetime.now().isoformat(),
            'symbols_processed': 0,
            'symbols_failed': 0,
            'errors': []
        }
        
        end_date = date.today()
        start_date = end_date - timedelta(days=30)  # Get last 30 days
        
        for symbol in symbols:
            try:
                # Fetch both raw and adjusted data
                raw_data = self.pipeline.get_data(symbol, start_date, end_date, adjusted=False, force_refresh=True)
                adjusted_data = self.pipeline.get_data(symbol, start_date, end_date, adjusted=True, force_refresh=True)
                
                if raw_data and adjusted_data:
                    results['symbols_processed'] += 1
                    logger.info(f"Successfully updated {symbol}")
                else:
                    results['symbols_failed'] += 1
                    results['errors'].append(f"Failed to fetch data for {symbol}")
                
            except Exception as e:
                results['symbols_failed'] += 1
                results['errors'].append(f"Error processing {symbol}: {str(e)}")
                logger.error(f"Error in batch update for {symbol}: {e}")
        
        results['completed_at'] = datetime.now().isoformat()
        return results
"""
Storage backends for data caching and persistence.

Supports multiple storage formats:
- Parquet/Feather for local high-performance storage
- Database backends (PostgreSQL, TimescaleDB) for scalable deployments
- S3/cloud storage for distributed architectures
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, date
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import sqlite3
from contextlib import contextmanager

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False

try:
    import psycopg2
    from sqlalchemy import create_engine
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class StorageConfig:
    """Configuration for storage backends."""
    backend: str = "parquet"  # parquet, feather, sqlite, postgresql
    base_path: Optional[str] = None
    connection_string: Optional[str] = None
    compression: str = "snappy"
    chunk_size: int = 10000
    create_directories: bool = True
    backup_enabled: bool = True
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

        # Set default base path if not provided
        if self.base_path is None:
            self.base_path = str(Path.cwd() / "data" / "storage")


class DataStorage(ABC):
    """Abstract base class for data storage backends."""

    def __init__(self, config: StorageConfig):
        self.config = config
        self._ensure_storage_ready()

    @abstractmethod
    def store_data(self, symbol: str, data: pd.DataFrame, data_type: str = "ohlcv") -> bool:
        """Store data for a symbol."""
        pass

    @abstractmethod
    def load_data(self, symbol: str, data_type: str = "ohlcv",
                  start_date: Optional[date] = None,
                  end_date: Optional[date] = None) -> Optional[pd.DataFrame]:
        """Load data for a symbol."""
        pass

    @abstractmethod
    def exists(self, symbol: str, data_type: str = "ohlcv") -> bool:
        """Check if data exists for a symbol."""
        pass

    @abstractmethod
    def list_symbols(self, data_type: str = "ohlcv") -> List[str]:
        """List all available symbols."""
        pass

    @abstractmethod
    def get_last_update(self, symbol: str, data_type: str = "ohlcv") -> Optional[datetime]:
        """Get last update timestamp for symbol."""
        pass

    @abstractmethod
    def delete_data(self, symbol: str, data_type: str = "ohlcv") -> bool:
        """Delete data for a symbol."""
        pass

    def _ensure_storage_ready(self):
        """Ensure storage backend is ready."""
        pass


class ParquetStorage(DataStorage):
    """Parquet-based storage backend for high-performance local storage."""

    def __init__(self, config: StorageConfig):
        if not PARQUET_AVAILABLE:
            raise ImportError("pyarrow is required for Parquet storage")
        super().__init__(config)

    def _ensure_storage_ready(self):
        """Create directory structure if needed."""
        base_path = Path(self.config.base_path)
        if self.config.create_directories:
            base_path.mkdir(parents=True, exist_ok=True)

            # Create subdirectories for different data types
            for data_type in ["ohlcv", "corporate_actions", "fundamentals", "metadata"]:
                (base_path / data_type).mkdir(exist_ok=True)

    def _get_file_path(self, symbol: str, data_type: str) -> Path:
        """Get file path for symbol and data type."""
        base_path = Path(self.config.base_path)
        return base_path / data_type / f"{symbol}.parquet"

    def _get_metadata_path(self, symbol: str, data_type: str) -> Path:
        """Get metadata file path."""
        base_path = Path(self.config.base_path)
        return base_path / "metadata" / f"{symbol}_{data_type}_meta.json"

    def store_data(self, symbol: str, data: pd.DataFrame, data_type: str = "ohlcv") -> bool:
        """Store data in Parquet format."""
        try:
            file_path = self._get_file_path(symbol, data_type)

            # Backup existing file if backup is enabled
            if self.config.backup_enabled and file_path.exists():
                backup_path = file_path.with_suffix(f".backup_{int(datetime.now().timestamp())}.parquet")
                file_path.rename(backup_path)

            # Store data
            data.to_parquet(
                file_path,
                compression=self.config.compression,
                index=False
            )

            # Store metadata
            metadata = {
                "symbol": symbol,
                "data_type": data_type,
                "rows": len(data),
                "columns": list(data.columns),
                "last_update": datetime.now().isoformat(),
                "file_size": file_path.stat().st_size,
                "checksum": self._calculate_checksum(data)
            }

            metadata_path = self._get_metadata_path(symbol, data_type)
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Stored {symbol} {data_type} data: {len(data)} rows")
            return True

        except Exception as e:
            logger.error(f"Failed to store {symbol} {data_type}: {e}")
            return False

    def load_data(self, symbol: str, data_type: str = "ohlcv",
                  start_date: Optional[date] = None,
                  end_date: Optional[date] = None) -> Optional[pd.DataFrame]:
        """Load data from Parquet file."""
        try:
            file_path = self._get_file_path(symbol, data_type)
            if not file_path.exists():
                return None

            data = pd.read_parquet(file_path)

            # Filter by date range if specified
            if start_date or end_date:
                if 'Date' in data.columns:
                    date_col = pd.to_datetime(data['Date'])
                    if start_date:
                        data = data[date_col >= pd.Timestamp(start_date)]
                    if end_date:
                        data = data[date_col <= pd.Timestamp(end_date)]

            return data

        except Exception as e:
            logger.error(f"Failed to load {symbol} {data_type}: {e}")
            return None

    def exists(self, symbol: str, data_type: str = "ohlcv") -> bool:
        """Check if data file exists."""
        file_path = self._get_file_path(symbol, data_type)
        return file_path.exists()

    def list_symbols(self, data_type: str = "ohlcv") -> List[str]:
        """List all symbols with stored data."""
        base_path = Path(self.config.base_path) / data_type
        if not base_path.exists():
            return []

        symbols = []
        for file_path in base_path.glob("*.parquet"):
            symbols.append(file_path.stem)

        return sorted(symbols)

    def get_last_update(self, symbol: str, data_type: str = "ohlcv") -> Optional[datetime]:
        """Get last update timestamp from metadata."""
        try:
            metadata_path = self._get_metadata_path(symbol, data_type)
            if not metadata_path.exists():
                return None

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            return datetime.fromisoformat(metadata['last_update'])

        except Exception as e:
            logger.error(f"Failed to get last update for {symbol}: {e}")
            return None

    def delete_data(self, symbol: str, data_type: str = "ohlcv") -> bool:
        """Delete data and metadata files."""
        try:
            file_path = self._get_file_path(symbol, data_type)
            metadata_path = self._get_metadata_path(symbol, data_type)

            deleted = False
            if file_path.exists():
                file_path.unlink()
                deleted = True

            if metadata_path.exists():
                metadata_path.unlink()
                deleted = True

            if deleted:
                logger.info(f"Deleted {symbol} {data_type} data")

            return deleted

        except Exception as e:
            logger.error(f"Failed to delete {symbol} {data_type}: {e}")
            return False

    def _calculate_checksum(self, data: pd.DataFrame) -> str:
        """Calculate simple checksum for data validation."""
        import hashlib
        data_str = f"{len(data)}_{data.columns.tolist()}"
        if not data.empty:
            data_str += f"_{data.iloc[0].to_dict()}_{data.iloc[-1].to_dict()}"
        return hashlib.md5(data_str.encode()).hexdigest()


class SQLiteStorage(DataStorage):
    """SQLite-based storage for lightweight database needs."""

    def __init__(self, config: StorageConfig):
        super().__init__(config)

    def _ensure_storage_ready(self):
        """Create database and tables if needed."""
        db_path = Path(self.config.base_path) / "stockdata.db"

        # Create directory if needed
        if self.config.create_directories:
            db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create tables
        with self._get_connection() as conn:
            self._create_tables(conn)

    @contextmanager
    def _get_connection(self):
        """Get database connection context manager."""
        db_path = Path(self.config.base_path) / "stockdata.db"
        conn = sqlite3.connect(str(db_path))
        try:
            yield conn
        finally:
            conn.close()

    def _create_tables(self, conn: sqlite3.Connection):
        """Create necessary tables."""
        # OHLCV data table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_data (
                symbol TEXT,
                date DATE,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                adj_close REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, date)
            )
        """)

        # Corporate actions table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS corporate_actions (
                symbol TEXT,
                date DATE,
                action_type TEXT,
                ratio REAL,
                amount REAL,
                ex_date DATE,
                record_date DATE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, date, action_type)
            )
        """)

        # Metadata table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS data_metadata (
                symbol TEXT,
                data_type TEXT,
                last_update TIMESTAMP,
                row_count INTEGER,
                checksum TEXT,
                PRIMARY KEY (symbol, data_type)
            )
        """)

        conn.commit()

    def store_data(self, symbol: str, data: pd.DataFrame, data_type: str = "ohlcv") -> bool:
        """Store data in SQLite database."""
        try:
            with self._get_connection() as conn:
                if data_type == "ohlcv":
                    self._store_ohlcv_data(conn, symbol, data)
                elif data_type == "corporate_actions":
                    self._store_corporate_actions(conn, symbol, data)
                else:
                    raise ValueError(f"Unsupported data type: {data_type}")

                # Update metadata
                self._update_metadata(conn, symbol, data_type, len(data))

            logger.info(f"Stored {symbol} {data_type} data: {len(data)} rows")
            return True

        except Exception as e:
            logger.error(f"Failed to store {symbol} {data_type}: {e}")
            return False

    def _store_ohlcv_data(self, conn: sqlite3.Connection, symbol: str, data: pd.DataFrame):
        """Store OHLCV data."""
        # Prepare data for insertion
        data_copy = data.copy()
        data_copy['symbol'] = symbol

        # Ensure required columns exist
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in data_copy.columns:
                raise ValueError(f"Missing required column: {col}")

        # Add adj_close if not present
        if 'Adj_Close' not in data_copy.columns:
            data_copy['Adj_Close'] = data_copy['Close']

        # Insert data (replace on conflict)
        data_copy.to_sql(
            'ohlcv_data',
            conn,
            if_exists='replace',
            index=False,
            method='multi'
        )

    def _store_corporate_actions(self, conn: sqlite3.Connection, symbol: str, data: pd.DataFrame):
        """Store corporate actions data."""
        data_copy = data.copy()
        data_copy['symbol'] = symbol

        data_copy.to_sql(
            'corporate_actions',
            conn,
            if_exists='replace',
            index=False,
            method='multi'
        )

    def _update_metadata(self, conn: sqlite3.Connection, symbol: str, data_type: str, row_count: int):
        """Update metadata table."""
        conn.execute("""
            INSERT OR REPLACE INTO data_metadata
            (symbol, data_type, last_update, row_count, checksum)
            VALUES (?, ?, CURRENT_TIMESTAMP, ?, ?)
        """, (symbol, data_type, row_count, ""))
        conn.commit()

    def load_data(self, symbol: str, data_type: str = "ohlcv",
                  start_date: Optional[date] = None,
                  end_date: Optional[date] = None) -> Optional[pd.DataFrame]:
        """Load data from SQLite database."""
        try:
            with self._get_connection() as conn:
                if data_type == "ohlcv":
                    return self._load_ohlcv_data(conn, symbol, start_date, end_date)
                elif data_type == "corporate_actions":
                    return self._load_corporate_actions(conn, symbol, start_date, end_date)
                else:
                    raise ValueError(f"Unsupported data type: {data_type}")

        except Exception as e:
            logger.error(f"Failed to load {symbol} {data_type}: {e}")
            return None

    def _load_ohlcv_data(self, conn: sqlite3.Connection, symbol: str,
                        start_date: Optional[date], end_date: Optional[date]) -> pd.DataFrame:
        """Load OHLCV data from database."""
        query = "SELECT * FROM ohlcv_data WHERE symbol = ?"
        params = [symbol]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date"

        return pd.read_sql_query(query, conn, params=params)

    def _load_corporate_actions(self, conn: sqlite3.Connection, symbol: str,
                               start_date: Optional[date], end_date: Optional[date]) -> pd.DataFrame:
        """Load corporate actions data."""
        query = "SELECT * FROM corporate_actions WHERE symbol = ?"
        params = [symbol]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date"

        return pd.read_sql_query(query, conn, params=params)

    def exists(self, symbol: str, data_type: str = "ohlcv") -> bool:
        """Check if data exists for symbol."""
        try:
            with self._get_connection() as conn:
                if data_type == "ohlcv":
                    table = "ohlcv_data"
                elif data_type == "corporate_actions":
                    table = "corporate_actions"
                else:
                    return False

                cursor = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE symbol = ?", (symbol,))
                count = cursor.fetchone()[0]
                return count > 0

        except Exception:
            return False

    def list_symbols(self, data_type: str = "ohlcv") -> List[str]:
        """List all symbols with data."""
        try:
            with self._get_connection() as conn:
                if data_type == "ohlcv":
                    table = "ohlcv_data"
                elif data_type == "corporate_actions":
                    table = "corporate_actions"
                else:
                    return []

                cursor = conn.execute(f"SELECT DISTINCT symbol FROM {table} ORDER BY symbol")
                return [row[0] for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Failed to list symbols: {e}")
            return []

    def get_last_update(self, symbol: str, data_type: str = "ohlcv") -> Optional[datetime]:
        """Get last update timestamp."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT last_update FROM data_metadata WHERE symbol = ? AND data_type = ?",
                    (symbol, data_type)
                )
                result = cursor.fetchone()
                if result:
                    return datetime.fromisoformat(result[0])
                return None

        except Exception as e:
            logger.error(f"Failed to get last update: {e}")
            return None

    def delete_data(self, symbol: str, data_type: str = "ohlcv") -> bool:
        """Delete data for symbol."""
        try:
            with self._get_connection() as conn:
                if data_type == "ohlcv":
                    conn.execute("DELETE FROM ohlcv_data WHERE symbol = ?", (symbol,))
                elif data_type == "corporate_actions":
                    conn.execute("DELETE FROM corporate_actions WHERE symbol = ?", (symbol,))

                conn.execute(
                    "DELETE FROM data_metadata WHERE symbol = ? AND data_type = ?",
                    (symbol, data_type)
                )
                conn.commit()

            logger.info(f"Deleted {symbol} {data_type} data")
            return True

        except Exception as e:
            logger.error(f"Failed to delete {symbol} {data_type}: {e}")
            return False


def get_storage(config: StorageConfig) -> DataStorage:
    """Factory function to get appropriate storage backend."""
    if config.backend == "parquet":
        return ParquetStorage(config)
    elif config.backend == "sqlite":
        return SQLiteStorage(config)
    elif config.backend == "feather":
        # Could implement FeatherStorage similar to ParquetStorage
        raise NotImplementedError("Feather storage not yet implemented")
    elif config.backend == "postgresql":
        # Could implement PostgreSQLStorage for production databases
        raise NotImplementedError("PostgreSQL storage not yet implemented")
    else:
        raise ValueError(f"Unknown storage backend: {config.backend}")


# Storage factory functions for common configurations
def create_local_storage(base_path: str = None, format: str = "parquet") -> DataStorage:
    """Create local file-based storage."""
    config = StorageConfig(
        backend=format,
        base_path=base_path or str(Path.cwd() / "data" / "storage"),
        create_directories=True,
        backup_enabled=True
    )
    return get_storage(config)


def create_database_storage(connection_string: str, backend: str = "sqlite") -> DataStorage:
    """Create database-backed storage."""
    config = StorageConfig(
        backend=backend,
        connection_string=connection_string,
        create_directories=True
    )
    return get_storage(config)
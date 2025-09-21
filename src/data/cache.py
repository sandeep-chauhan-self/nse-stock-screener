"""
Disk-based caching system for financial data.

Provides efficient caching with TTL, compression, and cache management features.
Supports multiple cache storage formats for different use cases.
"""

from datetime import datetime, timedelta
from pathlib import Path
import gzip
import hashlib
import json
import logging
import os
import pickle
import sqlite3
import time

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Union
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Metadata for cached data"""
    key: str
    created_at: float
    ttl_hours: int
    size_bytes: int
    format: str  # 'parquet', 'pickle', 'json'
    compressed: bool = False

    @property
    def expires_at(self) -> float:
        return self.created_at + (self.ttl_hours * 3600)

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    @property
    def age_hours(self) -> float:
        return (time.time() - self.created_at) / 3600


class CacheStorage(ABC):
    """Abstract base class for cache storage backends"""

    @abstractmethod
    def store(self, key: str, data: Any, metadata: CacheEntry) -> bool:
        """Store data with metadata"""
        pass

    @abstractmethod
    def retrieve(self, key: str) -> tuple[Any, CacheEntry]:
        """Retrieve data and metadata"""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete cached entry"""
        pass

    @abstractmethod
    def list_keys(self) -> List[str]:
        """List all cached keys"""
        pass

    @abstractmethod
    def cleanup(self) -> int:
        """Remove expired entries, return count deleted"""
        pass


class FileSystemCacheStorage(CacheStorage):
    """File system based cache storage with SQLite metadata tracking"""

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # SQLite database for metadata
        self.db_path = self.cache_dir / "cache_metadata.db"
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database for cache metadata"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    created_at REAL,
                    ttl_hours INTEGER,
                    size_bytes INTEGER,
                    format TEXT,
                    compressed BOOLEAN,
                    file_path TEXT
                )
            """)
            conn.commit()

    def _get_file_path(self, key: str, format: str) -> Path:
        """Get file path for cached data"""
        # Use hash to avoid filesystem issues with special characters
        key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
        filename = f"{key_hash}.{format}"
        return self.cache_dir / filename

    def store(self, key: str, data: Any, metadata: CacheEntry) -> bool:
        """Store data to filesystem with metadata in SQLite"""
        try:
            file_path = self._get_file_path(key, metadata.format)

            # Store data based on format
            if metadata.format == "parquet" and isinstance(data, pd.DataFrame):
                if metadata.compressed:
                    data.to_parquet(file_path, compression='gzip')
                else:
                    data.to_parquet(file_path)

            elif metadata.format == "pickle":
                if metadata.compressed:
                    with gzip.open(f"{file_path}.gz", 'wb') as f:
                        pickle.dump(data, f)
                    file_path = Path(f"{file_path}.gz")
                else:
                    with open(file_path, 'wb') as f:
                        pickle.dump(data, f)

            elif metadata.format == "json":
                json_data = json.dumps(data, default=str)
                if metadata.compressed:
                    with gzip.open(f"{file_path}.gz", 'wt') as f:
                        f.write(json_data)
                    file_path = Path(f"{file_path}.gz")
                else:
                    with open(file_path, 'w') as f:
                        f.write(json_data)

            else:
                raise ValueError(f"Unsupported format: {metadata.format}")

            # Update metadata with actual file size
            metadata.size_bytes = file_path.stat().st_size

            # Store metadata in SQLite
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO cache_entries
                    (key, created_at, ttl_hours, size_bytes, format, compressed, file_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    key, metadata.created_at, metadata.ttl_hours,
                    metadata.size_bytes, metadata.format, metadata.compressed,
                    str(file_path)
                ))
                conn.commit()

            logger.debug(f"Cached {key} ({metadata.size_bytes} bytes, TTL: {metadata.ttl_hours}h)")
            return True

        except Exception as e:
            logger.error(f"Failed to store cache entry {key}: {e}")
            return False

    def retrieve(self, key: str) -> tuple[Any, CacheEntry]:
        """Retrieve data and metadata from cache"""
        try:
            # Get metadata from SQLite
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute("""
                    SELECT created_at, ttl_hours, size_bytes, format, compressed, file_path
                    FROM cache_entries WHERE key = ?
                """, (key,)).fetchone()

            if not row:
                raise KeyError(f"Cache key not found: {key}")

            created_at, ttl_hours, size_bytes, format, compressed, file_path = row
            file_path = Path(file_path)

            if not file_path.exists():
                # File was deleted but metadata still exists - clean up
                self.delete(key)
                raise KeyError(f"Cache file not found: {file_path}")

            # Create metadata object
            metadata = CacheEntry(
                key=key,
                created_at=created_at,
                ttl_hours=ttl_hours,
                size_bytes=size_bytes,
                format=format,
                compressed=compressed
            )

            # Check if expired
            if metadata.is_expired:
                self.delete(key)
                raise KeyError(f"Cache entry expired: {key}")

            # Load data based on format
            if format == "parquet":
                data = pd.read_parquet(file_path)

            elif format == "pickle":
                if compressed:
                    with gzip.open(file_path, 'rb') as f:
                        data = pickle.load(f)
                else:
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)

            elif format == "json":
                if compressed:
                    with gzip.open(file_path, 'rt') as f:
                        data = json.load(f)
                else:
                    with open(file_path, 'r') as f:
                        data = json.load(f)

            else:
                raise ValueError(f"Unsupported format: {format}")

            logger.debug(f"Retrieved cached {key} (age: {metadata.age_hours:.1f}h)")
            return data, metadata

        except Exception as e:
            logger.debug(f"Cache miss for {key}: {e}")
            raise KeyError(f"Cache retrieval failed: {e}")

    def exists(self, key: str) -> bool:
        """Check if key exists and is not expired"""
        try:
            _, metadata = self.retrieve(key)
            return not metadata.is_expired
        except KeyError:
            return False

    def delete(self, key: str) -> bool:
        """Delete cached entry"""
        try:
            # Get file path from metadata
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute("""
                    SELECT file_path FROM cache_entries WHERE key = ?
                """, (key,)).fetchone()

                if row:
                    file_path = Path(row[0])

                    # Delete file if exists
                    if file_path.exists():
                        file_path.unlink()

                    # Delete metadata
                    conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                    conn.commit()

                    logger.debug(f"Deleted cache entry: {key}")
                    return True

            return False

        except Exception as e:
            logger.error(f"Failed to delete cache entry {key}: {e}")
            return False

    def list_keys(self) -> List[str]:
        """List all cached keys"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute("SELECT key FROM cache_entries").fetchall()
                return [row[0] for row in rows]
        except Exception as e:
            logger.error(f"Failed to list cache keys: {e}")
            return []

    def cleanup(self) -> int:
        """Remove expired entries"""
        deleted_count = 0
        current_time = time.time()

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Find expired entries
                expired_rows = conn.execute("""
                    SELECT key, file_path FROM cache_entries
                    WHERE ? > (created_at + (ttl_hours * 3600))
                """, (current_time,)).fetchall()

                for key, file_path in expired_rows:
                    try:
                        # Delete file
                        file_path_obj = Path(file_path)
                        if file_path_obj.exists():
                            file_path_obj.unlink()

                        # Delete metadata
                        conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                        deleted_count += 1

                    except Exception as e:
                        logger.warning(f"Failed to delete expired entry {key}: {e}")

                conn.commit()

                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} expired cache entries")

        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")

        return deleted_count


class DataCache:
    """
    High-level cache interface for financial data.

    Automatically handles different data types and provides convenient methods
    for caching stock data, symbol lists, and other financial information.
    """

    def __init__(self, cache_dir: Path = None, default_ttl_hours: int = 24):
        self.cache_dir = cache_dir or Path("data/cache")
        self.default_ttl_hours = default_ttl_hours
        self.storage = FileSystemCacheStorage(self.cache_dir)

        logger.info(f"DataCache initialized with cache_dir: {self.cache_dir}")

    def _create_metadata(self, key: str, data: Any, ttl_hours: int = None,
                        format: str = None, compressed: bool = None) -> CacheEntry:
        """Create cache metadata for data"""

        # Auto-detect format if not specified
        if format is None:
            if isinstance(data, pd.DataFrame):
                format = "parquet"
            elif isinstance(data, (dict, list)):
                format = "json"
            else:
                format = "pickle"

        # Auto-detect compression (compress large data)
        if compressed is None:
            if isinstance(data, pd.DataFrame):
                compressed = len(data) > 1000  # Compress large DataFrames
            else:
                compressed = False

        return CacheEntry(
            key=key,
            created_at=time.time(),
            ttl_hours=ttl_hours or self.default_ttl_hours,
            size_bytes=0,  # Will be updated during storage
            format=format,
            compressed=compressed
        )

    def set(self, key: str, data: Any, ttl_hours: int = None, **kwargs) -> bool:
        """
        Store data in cache.

        Args:
            key: Cache key
            data: Data to cache
            ttl_hours: Time to live in hours
            **kwargs: Additional options (format, compressed)

        Returns:
            True if successful, False otherwise
        """
        metadata = self._create_metadata(key, data, ttl_hours, **kwargs)
        return self.storage.store(key, data, metadata)

    def get(self, key: str) -> Any:
        """
        Retrieve data from cache.

        Args:
            key: Cache key

        Returns:
            Cached data

        Raises:
            KeyError: If key not found or expired
        """
        data, _ = self.storage.retrieve(key)
        return data

    def get_with_metadata(self, key: str) -> tuple[Any, CacheEntry]:
        """Retrieve data with metadata"""
        return self.storage.retrieve(key)

    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        return self.storage.exists(key)

    def delete(self, key: str) -> bool:
        """Delete cached entry"""
        return self.storage.delete(key)

    def clear_all(self) -> int:
        """Clear all cached entries"""
        keys = self.storage.list_keys()
        deleted = 0
        for key in keys:
            if self.storage.delete(key):
                deleted += 1
        return deleted

    def cleanup_expired(self) -> int:
        """Remove expired entries"""
        return self.storage.cleanup()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            keys = self.storage.list_keys()
            total_entries = len(keys)

            total_size = 0
            expired_count = 0
            format_counts = {}

            for key in keys:
                try:
                    _, metadata = self.storage.retrieve(key)
                    total_size += metadata.size_bytes

                    format_counts[metadata.format] = format_counts.get(metadata.format, 0) + 1

                    if metadata.is_expired:
                        expired_count += 1

                except KeyError:
                    expired_count += 1

            return {
                'total_entries': total_entries,
                'expired_entries': expired_count,
                'active_entries': total_entries - expired_count,
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'format_distribution': format_counts,
                'cache_directory': str(self.cache_dir)
            }

        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {'error': str(e)}

    # Convenience methods for specific data types

    def cache_stock_data(self, symbol: str, period: str, data: pd.DataFrame,
                        ttl_hours: int = 24) -> bool:
        """Cache historical stock data"""
        key = f"stock_data_{symbol}_{period}"
        return self.set(key, data, ttl_hours=ttl_hours, format="parquet", compressed=True)

    def get_stock_data(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """Retrieve cached stock data"""
        key = f"stock_data_{symbol}_{period}"
        try:
            return self.get(key)
        except KeyError:
            return None

    def cache_symbol_list(self, source: str, symbols: List[str],
                         ttl_hours: int = 12) -> bool:
        """Cache symbol list (e.g., NSE symbols)"""
        key = f"symbols_{source}"
        return self.set(key, symbols, ttl_hours=ttl_hours, format="json")

    def get_symbol_list(self, source: str) -> Optional[List[str]]:
        """Retrieve cached symbol list"""
        key = f"symbols_{source}"
        try:
            return self.get(key)
        except KeyError:
            return None

    def cache_symbol_info(self, symbol: str, info: Dict[str, Any],
                         ttl_hours: int = 168) -> bool:  # 1 week
        """Cache symbol information"""
        key = f"symbol_info_{symbol}"
        return self.set(key, info, ttl_hours=ttl_hours, format="json")

    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached symbol information"""
        key = f"symbol_info_{symbol}"
        try:
            return self.get(key)
        except KeyError:
            return None


# Create default cache instance
default_cache = DataCache()


class ParquetCache:
    """
    Specialized cache for OHLCV data using Parquet format.

    Optimized for financial time series data with intelligent
    partitioning and compression.
    """

    def __init__(self, cache_dir: Path):
        """Initialize Parquet cache."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Use the DataCache backend for metadata management
        self.data_cache = DataCache(cache_dir, default_ttl_hours=24)

        logger.info(f"ParquetCache initialized at {cache_dir}")

    def get(self, key: str) -> Optional[pd.DataFrame]:
        """Get cached DataFrame."""
        return self.data_cache.get_stock_data("cache", key)

    def set(self, key: str, data: pd.DataFrame, ttl_seconds: Optional[int] = None) -> None:
        """Cache DataFrame with TTL."""
        ttl_hours = ttl_seconds // 3600 if ttl_seconds else 24
        self.data_cache.cache_stock_data("cache", key, data, ttl_hours)

    def invalidate(self, pattern: str) -> None:
        """Invalidate cached data matching pattern."""
        keys = self.data_cache.storage.list_keys()
        deleted_count = 0

        for key in keys:
            if pattern in key:
                if self.data_cache.delete(key):
                    deleted_count += 1

        logger.info(f"Invalidated {deleted_count} cache entries matching pattern: {pattern}")


def cache_maintenance():
    """Utility function to perform cache maintenance"""
    logger.info("Starting cache maintenance...")

    # Cleanup expired entries
    deleted = default_cache.cleanup_expired()

    # Get cache stats
    stats = default_cache.get_cache_stats()

    logger.info(f"Cache maintenance completed: {deleted} expired entries removed")
    logger.info(f"Cache stats: {stats['active_entries']} active entries, {stats['total_size_mb']} MB")

    return stats
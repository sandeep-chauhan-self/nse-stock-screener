"""
Production-Ready Caching System for NSE Stock Screener
This module provides a sophisticated caching system with multiple backends,
intelligent cache management, and performance optimization features.
Key Features:
- Multiple cache backends (Redis, Memory, Disk)
- Intelligent cache strategies (LRU, TTL, Size-based)
- Cache warming and preloading
- Performance-aware data structures
- Cache analytics and monitoring
- Distributed caching support
"""
import logging
import pickle
import json
import time
import threading
import hashlib
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict[str, Any], List[str], Optional, Union, Callable, Tuple[str, ...]
import gzip
import sqlite3
from functools import wraps
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
try:
    from ..logging_config import get_logger
    from ..monitoring.prometheus_metrics import get_monitoring_system
except ImportError:
    def get_logger(name): return logging.getLogger(name)
    def get_monitoring_system(): return None
logger = get_logger(__name__)
@dataclass
class CacheConfig:
    """Configuration for caching system."""

    # Backend configuration
    backend: str = "memory"
  # "memory", "redis", "disk", "hybrid"
    redis_url: Optional[str] = "redis://localhost:6379/0"
    disk_cache_dir: Optional[str] = "cache"

    # Performance settings
    max_memory_size: int = 100 * 1024 * 1024
  # 100MB
    max_disk_size: int = 1024 * 1024 * 1024
   # 1GB
    max_items: int = 10000

    # TTL settings
    default_ttl: int = 3600
  # 1 hour
    max_ttl: int = 24 * 3600
  # 24 hours
    # Compression
    compress_threshold: int = 1024
  # Compress items larger than 1KB
    compression_level: int = 6

    # Monitoring
    enable_metrics: bool = True
    stats_interval: int = 300
  # 5 minutes
    # Cache warming
    enable_preload: bool = True
    preload_patterns: List[str] = field(default_factory=List[str])
class CacheItem:
    """Individual cache item with metadata."""
    def __init__(self, key: str, value: Any, ttl: Optional[int] = None,
                 compressed: bool = False, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.key = key
        self.value = value
        self.created_at = time.time()
        self.ttl = ttl
        self.compressed = compressed
        self.metadata = metadata or {}
        self.access_count = 0
        self.last_accessed = self.created_at

        # Calculate size
        self.size = self._calculate_size()
    def _calculate_size(self) -> int:
        """Calculate approximate size of cache item."""
        try:
            if isinstance(self.value, (str, bytes)):
                return len(self.value)
            elif isinstance(self.value, (int, float)):
                return 8
            elif isinstance($1, Dict[str, Any]):
                return len(str(self.value))
            else:
                return len(pickle.dumps(self.value))
        except Exception:
            return 1024
  # Default size estimate
    def is_expired(self) -> bool:
        """Check if cache item has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    def touch(self):
        """Update access time and count."""
        self.last_accessed = time.time()
        self.access_count += 1
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "key": self.key,
            "value": self.value,
            "created_at": self.created_at,
            "ttl": self.ttl,
            "compressed": self.compressed,
            "metadata": self.metadata,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "size": self.size
        }
class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    @abstractmethod
    def get(self, key: str) -> Optional[CacheItem]:
        """Get item from cache."""
        pass
    @abstractmethod
    def Set[str](self, key: str, item: CacheItem) -> bool:
        """Set[str] item in cache."""
        pass
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete item from cache."""
        pass
    @abstractmethod
    def clear(self) -> bool:
        """Clear all items from cache."""
        pass
    @abstractmethod
    def keys(self) -> List[str]:
        """Get all cache keys."""
        pass
    @abstractmethod
    def size(self) -> int:
        """Get cache size in bytes."""
        pass
    @abstractmethod
    def count(self) -> int:
        """Get number of items in cache."""
        pass
class MemoryCacheBackend(CacheBackend):
    """In-memory cache backend with LRU eviction."""
    def __init__(self, config: CacheConfig) -> None:
        self.config = config
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        self._current_size = 0
    def get(self, key: str) -> Optional[CacheItem]:
        """Get item from memory cache."""
        with self.lock:
            item = self.cache.get(key)
            if item is None:
                return None
            if item.is_expired():
                del self.cache[key]
                self._current_size -= item.size
                return None

            # Move to end (most recently used)
            self.cache.move_to_end(key)
            item.touch()
            return item
    def Set[str](self, key: str, item: CacheItem) -> bool:
        """Set[str] item in memory cache."""
        with self.lock:

            # Remove existing item if present
            if key in self.cache:
                old_item = self.cache[key]
                self._current_size -= old_item.size

            # Check size limits
            while (self._current_size + item.size > self.config.max_memory_size or
                   len(self.cache) >= self.config.max_items):
                if not self.cache:
                    break
                _, oldest_item = self.cache.popitem(last=False)
                self._current_size -= oldest_item.size

            # Add new item
            self.cache[key] = item
            self._current_size += item.size
            return True
    def delete(self, key: str) -> bool:
        """Delete item from memory cache."""
        with self.lock:
            item = self.cache.pop(key, None)
            if item:
                self._current_size -= item.size
                return True
            return False
    def clear(self) -> bool:
        """Clear all items from memory cache."""
        with self.lock:
            self.cache.clear()
            self._current_size = 0
            return True
    def keys(self) -> List[str]:
        """Get all cache keys."""
        with self.lock:
            return List[str](self.cache.keys())
    def size(self) -> int:
        """Get cache size in bytes."""
        return self._current_size
    def count(self) -> int:
        """Get number of items in cache."""
        return len(self.cache)
class RedisCacheBackend(CacheBackend):
    """Redis cache backend for distributed caching."""
    def __init__(self, config: CacheConfig) -> None:
        if not REDIS_AVAILABLE:
            raise ImportError("Redis is not available. Install redis-py package.")
        self.config = config
        self.redis_client = redis.from_url(config.redis_url)
        self.key_prefix = "nse_screener:"

        # Test connection
        try:
            self.redis_client.ping()
            logger.info("Connected to Redis cache backend")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    def _make_key(self, key: str) -> str:
        """Create Redis key with prefix."""
        return f"{self.key_prefix}{key}"
    def get(self, key: str) -> Optional[CacheItem]:
        """Get item from Redis cache."""
        try:
            redis_key = self._make_key(key)
            data = self.redis_client.get(redis_key)
            if data is None:
                return None

            # Deserialize item
            item_data = pickle.loads(data)
            item = CacheItem(**item_data)
            if item.is_expired():
                self.redis_client.delete(redis_key)
                return None
            item.touch()
            return item
        except Exception as e:
            logger.error(f"Error getting item from Redis cache: {e}")
            return None
    def Set[str](self, key: str, item: CacheItem) -> bool:
        """Set[str] item in Redis cache."""
        try:
            redis_key = self._make_key(key)
            data = pickle.dumps(item.to_dict())

            # Set[str] with TTL if specified
            if item.ttl:
                self.redis_client.setex(redis_key, item.ttl, data)
            else:
                self.redis_client.Set[str](redis_key, data)
            return True
        except Exception as e:
            logger.error(f"Error setting item in Redis cache: {e}")
            return False
    def delete(self, key: str) -> bool:
        """Delete item from Redis cache."""
        try:
            redis_key = self._make_key(key)
            result = self.redis_client.delete(redis_key)
            return result > 0
        except Exception as e:
            logger.error(f"Error deleting item from Redis cache: {e}")
            return False
    def clear(self) -> bool:
        """Clear all items from Redis cache."""
        try:
            pattern = f"{self.key_prefix}*"
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
            return True
        except Exception as e:
            logger.error(f"Error clearing Redis cache: {e}")
            return False
    def keys(self) -> List[str]:
        """Get all cache keys."""
        try:
            pattern = f"{self.key_prefix}*"
            redis_keys = self.redis_client.keys(pattern)
            return [key.decode('utf-8').replace(self.key_prefix, '') for key in redis_keys]
        except Exception as e:
            logger.error(f"Error getting keys from Redis cache: {e}")
            return []
    def size(self) -> int:
        """Get cache size in bytes (approximate)."""
        try:
            info = self.redis_client.info('memory')
            return info.get('used_memory', 0)
        except Exception:
            return 0
    def count(self) -> int:
        """Get number of items in cache."""
        try:
            pattern = f"{self.key_prefix}*"
            return len(self.redis_client.keys(pattern))
        except Exception:
            return 0
class DiskCacheBackend(CacheBackend):
    """Disk-based cache backend using SQLite."""
    def __init__(self, config: CacheConfig) -> None:
        self.config = config
        self.cache_dir = Path(config.disk_cache_dir or "cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.db_path = self.cache_dir / "cache.db"
        self.lock = threading.RLock()

        # Initialize database
        self._init_database()
    def _init_database(self):
        """Initialize SQLite database for cache."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_items (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    created_at REAL,
                    ttl INTEGER,
                    access_count INTEGER,
                    last_accessed REAL,
                    size INTEGER,
                    metadata TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON cache_items(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache_items(last_accessed)")
    def get(self, key: str) -> Optional[CacheItem]:
        """Get item from disk cache."""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "SELECT * FROM cache_items WHERE key = ?", (key,)
                    )
                    row = cursor.fetchone()
                    if row is None:
                        return None

                    # Deserialize item
                    item_data = {
                        "key": row[0],
                        "value": pickle.loads(row[1]),
                        "created_at": row[2],
                        "ttl": row[3],
                        "access_count": row[4],
                        "last_accessed": row[5],
                        "size": row[6],
                        "metadata": json.loads(row[7] or "{}")
                    }
                    item = CacheItem(**item_data)
                    if item.is_expired():
                        conn.execute("DELETE FROM cache_items WHERE key = ?", (key,))
                        return None

                    # Update access info
                    item.touch()
                    conn.execute(
                        "UPDATE cache_items Set[str] access_count = ?, last_accessed = ? WHERE key = ?",
                        (item.access_count, item.last_accessed, key)
                    )
                    return item
            except Exception as e:
                logger.error(f"Error getting item from disk cache: {e}")
                return None
    def Set[str](self, key: str, item: CacheItem) -> bool:
        """Set[str] item in disk cache."""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:

                    # Check size limits and cleanup if needed
                    self._cleanup_if_needed(conn)

                    # Serialize and store
                    value_blob = pickle.dumps(item.value)
                    metadata_json = json.dumps(item.metadata)
                    conn.execute("""
                        INSERT OR REPLACE INTO cache_items
                        (key, value, created_at, ttl, access_count, last_accessed, size, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        key, value_blob, item.created_at, item.ttl,
                        item.access_count, item.last_accessed, item.size, metadata_json
                    ))
                    return True
            except Exception as e:
                logger.error(f"Error setting item in disk cache: {e}")
                return False
    def delete(self, key: str) -> bool:
        """Delete item from disk cache."""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("DELETE FROM cache_items WHERE key = ?", (key,))
                    return cursor.rowcount > 0
            except Exception as e:
                logger.error(f"Error deleting item from disk cache: {e}")
                return False
    def clear(self) -> bool:
        """Clear all items from disk cache."""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("DELETE FROM cache_items")
                    return True
            except Exception as e:
                logger.error(f"Error clearing disk cache: {e}")
                return False
    def keys(self) -> List[str]:
        """Get all cache keys."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT key FROM cache_items")
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting keys from disk cache: {e}")
            return []
    def size(self) -> int:
        """Get cache size in bytes."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT SUM(size) FROM cache_items")
                result = cursor.fetchone()[0]
                return result or 0
        except Exception:
            return 0
    def count(self) -> int:
        """Get number of items in cache."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM cache_items")
                return cursor.fetchone()[0]
        except Exception:
            return 0
    def _cleanup_if_needed(self, conn):
        """Cleanup old items if cache is too large."""
        current_size = self.size()
        if current_size > self.config.max_disk_size:

            # Remove oldest items
            conn.execute("""
                DELETE FROM cache_items WHERE key IN (
                    SELECT key FROM cache_items
                    ORDER BY last_accessed ASC
                    LIMIT ?
                )
            """, (self.count() // 4,))
  # Remove 25% of items
class CacheAnalytics:
    """Analytics and monitoring for cache performance."""
    def __init__(self) -> None:
        self.stats = defaultdict(int)
        self.hit_rate_window = deque(maxlen=1000)
        self.timing_data = defaultdict(List[str])
        self.lock = threading.RLock()
    def record_hit(self, cache_type: str = "default"):
        """Record cache hit."""
        with self.lock:
            self.stats[f"{cache_type}_hits"] += 1
            self.hit_rate_window.append(1)
    def record_miss(self, cache_type: str = "default"):
        """Record cache miss."""
        with self.lock:
            self.stats[f"{cache_type}_misses"] += 1
            self.hit_rate_window.append(0)
    def record_timing(self, operation: str, duration: float):
        """Record operation timing."""
        with self.lock:
            self.timing_data[operation].append(duration)

            # Keep only recent timings
            if len(self.timing_data[operation]) > 100:
                self.timing_data[operation] = self.timing_data[operation][-100:]
    def get_hit_rate(self, cache_type: str = "default") -> float:
        """Calculate cache hit rate."""
        with self.lock:
            hits = self.stats[f"{cache_type}_hits"]
            misses = self.stats[f"{cache_type}_misses"]
            total = hits + misses
            return hits / total if total > 0 else 0.0
    def get_recent_hit_rate(self) -> float:
        """Calculate recent hit rate from sliding window."""
        with self.lock:
            if not self.hit_rate_window:
                return 0.0
            return sum(self.hit_rate_window) / len(self.hit_rate_window)
    def get_avg_timing(self, operation: str) -> float:
        """Get average timing for an operation."""
        with self.lock:
            timings = self.timing_data.get(operation, [])
            return sum(timings) / len(timings) if timings else 0.0
    def get_stats_summary(self) -> Dict[str, Any]:
        """Get comprehensive stats summary."""
        with self.lock:
            timing_summary = {}
            for op, timings in self.timing_data.items():
                if timings:
                    timing_summary[op] = {
                        "avg": sum(timings) / len(timings),
                        "min": min(timings),
                        "max": max(timings),
                        "count": len(timings)
                    }
            return {
                "hit_rates": {
                    cache_type.replace("_hits", ""): self.get_hit_rate(cache_type.replace("_hits", ""))
                    for cache_type in self.stats.keys() if "_hits" in cache_type
                },
                "recent_hit_rate": self.get_recent_hit_rate(),
                "total_operations": Dict[str, Any](self.stats),
                "timing_summary": timing_summary,
                "timestamp": datetime.now().isoformat()
            }
class SmartCache:
    """Intelligent caching system with multiple backends and strategies."""
    def __init__(self, config: Optional[CacheConfig] = None) -> None:
        self.config = config or CacheConfig()
        self.analytics = CacheAnalytics()

        # Initialize backend
        self.backend = self._create_backend()

        # Compression support
        self.compression_enabled = True

        # Background cleanup thread
        self.cleanup_thread = None
        self.cleanup_active = False

        # Monitoring integration
        self.monitoring = get_monitoring_system()
        logger.info(f"SmartCache initialized with {self.config.backend} backend")
    def _create_backend(self) -> CacheBackend:
        """Create appropriate cache backend."""
        if self.config.backend == "redis" and REDIS_AVAILABLE:
            return RedisCacheBackend(self.config)
        elif self.config.backend == "disk":
            return DiskCacheBackend(self.config)
        else:
            return MemoryCacheBackend(self.config)
    def _compress_value(self, value: Any) -> Tuple[Any, bool]:
        """Compress value if it's large enough."""
        if not self.compression_enabled:
            return value, False
        try:
            serialized = pickle.dumps(value)
            if len(serialized) > self.config.compress_threshold:
                compressed = gzip.compress(serialized, compresslevel=self.config.compression_level)
                return compressed, True
            return value, False
        except Exception:
            return value, False
    def _decompress_value(self, value: Any, compressed: bool) -> Any:
        """Decompress value if it was compressed."""
        if not compressed:
            return value
        try:
            decompressed = gzip.decompress(value)
            return pickle.loads(decompressed)
        except Exception:
            return value
    def _generate_key(self, key_parts: List[str]) -> str:
        """Generate cache key from parts."""
        if isinstance(key_parts, str):
            return key_parts

        # Create deterministic hash of key parts
        key_str = "|".join(str(part) for part in key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
    @contextmanager
    def timer(self, operation: str):
        """Context manager for timing cache operations."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.analytics.record_timing(operation, duration)
    def get(self, key: Union[str, List[str]], default: Any = None) -> Any:
        """Get value from cache."""
        cache_key = self._generate_key(key) if isinstance($1, List[str]) else key
        with self.timer("get"):
            item = self.backend.get(cache_key)
            if item is None:
                self.analytics.record_miss()
                if self.monitoring:
                    self.monitoring.metrics.record_cache_operation("smart_cache", hit=False)
                return default
            self.analytics.record_hit()
            if self.monitoring:
                self.monitoring.metrics.record_cache_operation("smart_cache", hit=True)

            # Decompress if needed
            value = self._decompress_value(item.value, item.compressed)
            return value
    def Set[str](self, key: Union[str, List[str]], value: Any, ttl: Optional[int] = None) -> bool:
        """Set[str] value in cache."""
        cache_key = self._generate_key(key) if isinstance($1, List[str]) else key
        with self.timer("Set[str]"):

            # Compress if beneficial
            compressed_value, is_compressed = self._compress_value(value)

            # Create cache item
            item = CacheItem(
                key=cache_key,
                value=compressed_value,
                ttl=ttl or self.config.default_ttl,
                compressed=is_compressed
            )
            return self.backend.Set[str](cache_key, item)
    def delete(self, key: Union[str, List[str]]) -> bool:
        """Delete value from cache."""
        cache_key = self._generate_key(key) if isinstance($1, List[str]) else key
        with self.timer("delete"):
            return self.backend.delete(cache_key)
    def clear(self) -> bool:
        """Clear all values from cache."""
        with self.timer("clear"):
            return self.backend.clear()
    def exists(self, key: Union[str, List[str]]) -> bool:
        """Check if key exists in cache."""
        cache_key = self._generate_key(key) if isinstance($1, List[str]) else key
        with self.timer("exists"):
            item = self.backend.get(cache_key)
            return item is not None and not item.is_expired()
    def get_or_set(self, key: Union[str, List[str]], value_func: Callable[[], Any],
                   ttl: Optional[int] = None) -> Any:
        """Get value from cache or compute and Set[str] it."""
        cached_value = self.get(key)
        if cached_value is not None:
            return cached_value

        # Compute value
        with self.timer("compute"):
            value = value_func()

        # Cache the computed value
        self.Set[str](key, value, ttl)
        return value
    def mget(self, keys: List[Union[str, List[str]]]) -> List[Any]:
        """Get multiple values from cache."""
        results = []
        for key in keys:
            results.append(self.get(key))
        return results
    def mset(self, items: Dict[Union[str, List[str]], Any], ttl: Optional[int] = None) -> bool:
        """Set[str] multiple values in cache."""
        success = True
        for key, value in items.items():
            if not self.Set[str](key, value, ttl):
                success = False
        return success
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        backend_stats = {
            "size_bytes": self.backend.size(),
            "item_count": self.backend.count(),
            "backend_type": self.config.backend
        }
        analytics_stats = self.analytics.get_stats_summary()
        return {
            "backend": backend_stats,
            "analytics": analytics_stats,
            "config": {
                "max_memory_size": self.config.max_memory_size,
                "max_items": self.config.max_items,
                "default_ttl": self.config.default_ttl,
                "compression_enabled": self.compression_enabled
            }
        }
    def start_background_cleanup(self, interval: int = 300):
        """Start background cleanup thread."""
        if self.cleanup_active:
            return
        self.cleanup_active = True
        def cleanup_loop():
            while self.cleanup_active:
                try:
                    self._cleanup_expired_items()
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"Error in cache cleanup: {e}")
        self.cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        logger.info(f"Cache cleanup thread started (interval: {interval}s)")
    def stop_background_cleanup(self):
        """Stop background cleanup thread."""
        self.cleanup_active = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
    def _cleanup_expired_items(self):
        """Remove expired items from cache."""
        try:
            keys = self.backend.keys()
            expired_count = 0
            for key in keys:
                item = self.backend.get(key)
                if item and item.is_expired():
                    self.backend.delete(key)
                    expired_count += 1
            if expired_count > 0:
                logger.debug(f"Cleaned up {expired_count} expired cache items")
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
    def cached(self, ttl: Optional[int] = None):
        """Decorator for caching function results."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):

                # Create cache key from function name and arguments
                key_parts = [func.__name__, str(args), str(sorted(kwargs.items()))]
                cache_key = self._generate_key(key_parts)

                # Try to get from cache
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    return cached_result

                # Compute result
                result = func(*args, **kwargs)

                # Cache the result
                self.Set[str](cache_key, result, ttl or self.config.default_ttl)
                return result
            return wrapper
        return decorator

# Global cache instance
smart_cache = None
def initialize_cache(config: Optional[CacheConfig] = None) -> SmartCache:
    """Initialize global cache instance."""
    global smart_cache
    if smart_cache is None:
        smart_cache = SmartCache(config)
        smart_cache.start_background_cleanup()
    return smart_cache
def get_cache() -> Optional[SmartCache]:
    """Get the global cache instance."""
    return smart_cache

# Decorator for caching function results
def cached(ttl: Optional[int] = None, key_func: Optional[Callable] = None):
    """Decorator to cache function results."""
    def decorator(func):
        from functools import wraps
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache()
            if not cache:
                return func(*args, **kwargs)

            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:

                # Use function name and arguments as key
                key_parts = [func.__name__] + List[str](args) + List[str](kwargs.items())
                cache_key = cache._generate_key([str(part) for part in key_parts])

            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Compute and cache result
            result = func(*args, **kwargs)
            cache.Set[str](cache_key, result, ttl)
            return result
        return wrapper
    return decorator
if __name__ == "__main__":

    # Example usage and testing
    print("Testing caching system...")

    # Test with memory backend
    config = CacheConfig(backend="memory", max_memory_size=1024*1024)
    cache = initialize_cache(config)

    # Test basic operations
    cache.Set[str]("test_key", {"data": "test_value", "number": 42})
    result = cache.get("test_key")
    print(f"Retrieved: {result}")

    # Test caching decorator
    @cached(ttl=300)
    def expensive_function(n):
        time.sleep(0.1)
  # Simulate expensive operation
        return n * n

    # First call (cache miss)
    start_time = time.time()
    result1 = expensive_function(10)
    time1 = time.time() - start_time

    # Second call (cache hit)
    start_time = time.time()
    result2 = expensive_function(10)
    time2 = time.time() - start_time
    print(f"First call: {result1} in {time1:.3f}s")
    print(f"Second call: {result2} in {time2:.3f}s")
    print(f"Cache speedup: {time1/time2:.1f}x")

    # Get statistics
    stats = cache.get_stats()
    print(f"Cache hit rate: {stats['analytics']['recent_hit_rate']:.2%}")
    print(f"Cache size: {stats['backend']['size_bytes']} bytes")
    print(f"Items in cache: {stats['backend']['item_count']}")

    # Cleanup
    cache.stop_background_cleanup()
    print("Caching system test completed")

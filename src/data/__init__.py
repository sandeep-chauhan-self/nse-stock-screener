"""
Data ingestion package for NSE Stock Screener

Provides robust, production-ready data fetching with:
- Retry logic and exponential backoff
- Local caching to reduce API calls
- Rate limiting and user-agent rotation
- Corporate action handling
- Data validation and monitoring
"""

from .cache import DataCache
from .fetchers import DataFetcher, YahooFinanceFetcher, NSEFetcher
from .fetchers_impl import YahooDataFetcher, AlphaVantageDataFetcher, DataFetcherFactory
from .validation import EnhancedDataValidator as DataValidator

__all__ = [
    'DataFetcher',
    'YahooFinanceFetcher',
    'NSEFetcher',
    'YahooDataFetcher',
    'AlphaVantageDataFetcher',
    'DataFetcherFactory',
    'DataCache',
    'DataValidator'
]
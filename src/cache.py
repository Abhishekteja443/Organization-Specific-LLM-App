"""
Caching utilities for query results and embeddings
"""
import hashlib
import functools
from typing import Callable, Any, Tuple
import threading
import time
from src import logger

class CacheEntry:
    def __init__(self, value: Any, ttl: int = 3600):
        self.value = value
        self.timestamp = time.time()
        self.ttl = ttl
    
    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl


class QueryCache:
    """Thread-safe cache for query results."""
    
    def __init__(self, max_size: int = 500, ttl: int = 3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl
        self.lock = threading.Lock()
    
    def _hash_query(self, query: str) -> str:
        """Create a hash key for the query."""
        return hashlib.md5(query.lower().encode()).hexdigest()
    
    def get(self, query: str) -> Tuple[bool, Any]:
        """Get cached result if available and not expired."""
        try:
            key = self._hash_query(query)
            with self.lock:
                if key in self.cache:
                    entry = self.cache[key]
                    if not entry.is_expired():
                        logger.info(f"Cache hit for query: {query[:50]}...")
                        return True, entry.value
                    else:
                        del self.cache[key]
                        logger.info(f"Cache expired for query: {query[:50]}...")
            return False, None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return False, None
    
    def set(self, query: str, value: Any):
        """Store result in cache."""
        try:
            key = self._hash_query(query)
            with self.lock:
                # Simple LRU: remove oldest if at capacity
                if len(self.cache) >= self.max_size:
                    oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].timestamp)
                    del self.cache[oldest_key]
                
                self.cache[key] = CacheEntry(value, self.ttl)
                logger.info(f"Cached query: {query[:50]}... (cache size: {len(self.cache)})")
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    def clear(self):
        """Clear entire cache."""
        with self.lock:
            size = len(self.cache)
            self.cache.clear()
            logger.info(f"Cleared cache ({size} entries)")


# Global cache instance
query_cache = QueryCache(max_size=500, ttl=3600)


def cached_result(ttl: int = 3600):
    """Decorator to cache function results."""
    def decorator(func: Callable) -> Callable:
        cache = QueryCache(ttl=ttl)
        
        @functools.wraps(func)
        def wrapper(query: str, *args, **kwargs):
            # Only cache if first argument is a string query
            hit, cached_value = cache.get(query)
            if hit:
                return cached_value
            
            result = func(query, *args, **kwargs)
            cache.set(query, result)
            return result
        
        wrapper.cache_clear = cache.clear
        return wrapper
    
    return decorator

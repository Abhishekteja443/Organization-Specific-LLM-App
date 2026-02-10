"""
Unit and Integration Tests for Organization-Specific LLM App
"""
import unittest
import json
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.validators import InputValidator
from src.rate_limiter import RateLimiter, RequestMonitor
from src.cache import QueryCache
from src.chat_engine import count_tokens_approximate, manage_conversation_history


class TestInputValidator(unittest.TestCase):
    """Test input validation utilities."""
    
    def test_validate_query_empty(self):
        """Test empty query validation."""
        is_valid, _, error = InputValidator.validate_query("")
        self.assertFalse(is_valid)
        self.assertIn("empty", error.lower())
    
    def test_validate_query_valid(self):
        """Test valid query."""
        is_valid, sanitized, error = InputValidator.validate_query("What are the admission requirements?")
        self.assertTrue(is_valid)
        self.assertEqual(sanitized, "What are the admission requirements?")
    
    def test_validate_query_too_long(self):
        """Test query exceeding max length."""
        long_query = "a" * 10000
        is_valid, _, error = InputValidator.validate_query(long_query)
        self.assertFalse(is_valid)
        self.assertIn("exceeds", error.lower())
    
    def test_validate_query_malicious(self):
        """Test query with malicious patterns."""
        malicious = "What is <script>alert('xss')</script> admission?"
        is_valid, _, error = InputValidator.validate_query(malicious)
        self.assertFalse(is_valid)
    
    def test_validate_url_valid(self):
        """Test valid URL."""
        is_valid, error = InputValidator.validate_url("https://example.com/page")
        self.assertTrue(is_valid)
        self.assertEqual(error, "")
    
    def test_validate_url_invalid_format(self):
        """Test invalid URL format."""
        is_valid, error = InputValidator.validate_url("not a url")
        self.assertFalse(is_valid)
        self.assertIn("Invalid", error)
    
    def test_validate_urls_list(self):
        """Test URL list validation."""
        urls = [
            "https://example.com/page1",
            "https://example.com/page2"
        ]
        all_valid, valid_urls, errors = InputValidator.validate_urls_list(urls)
        self.assertTrue(all_valid)
        self.assertEqual(len(valid_urls), 2)
        self.assertEqual(len(errors), 0)


class TestRateLimiter(unittest.TestCase):
    """Test rate limiting."""
    
    def setUp(self):
        self.limiter = RateLimiter(max_requests=5, window_seconds=60)
    
    def test_initial_request_allowed(self):
        """Test first request is allowed."""
        allowed, limits = self.limiter.is_allowed("192.168.1.1")
        self.assertTrue(allowed)
        self.assertEqual(limits["remaining"], 4)
    
    def test_rate_limit_exceeded(self):
        """Test rate limit is enforced."""
        ip = "192.168.1.1"
        
        # Make 5 requests (limit)
        for i in range(5):
            allowed, _ = self.limiter.is_allowed(ip)
            self.assertTrue(allowed)
        
        # 6th request should fail
        allowed, limits = self.limiter.is_allowed(ip)
        self.assertFalse(allowed)
        self.assertEqual(limits["remaining"], 0)
    
    def test_reset_ip(self):
        """Test IP reset."""
        ip = "192.168.1.1"
        
        # Exhaust limit
        for i in range(5):
            self.limiter.is_allowed(ip)
        
        # Reset
        self.limiter.reset_ip(ip)
        
        # Should allow again
        allowed, _ = self.limiter.is_allowed(ip)
        self.assertTrue(allowed)


class TestQueryCache(unittest.TestCase):
    """Test query caching."""
    
    def setUp(self):
        self.cache = QueryCache(max_size=10, ttl=3600)
    
    def test_cache_miss(self):
        """Test cache miss."""
        hit, value = self.cache.get("test query")
        self.assertFalse(hit)
        self.assertIsNone(value)
    
    def test_cache_hit(self):
        """Test cache hit."""
        query = "test query"
        expected_result = ("retrieved text", "source_url")
        
        self.cache.set(query, expected_result)
        hit, result = self.cache.get(query)
        
        self.assertTrue(hit)
        self.assertEqual(result, expected_result)
    
    def test_cache_lru(self):
        """Test LRU eviction when cache is full."""
        cache = QueryCache(max_size=3, ttl=3600)
        
        # Fill cache
        for i in range(3):
            cache.set(f"query{i}", f"result{i}")
        
        # Add one more (should evict oldest)
        cache.set("query3", "result3")
        
        # Oldest should be gone
        hit, _ = cache.get("query0")
        self.assertFalse(hit)
        
        # Newest should be there
        hit, _ = cache.get("query3")
        self.assertTrue(hit)


class TestRequestMonitor(unittest.TestCase):
    """Test request monitoring."""
    
    def setUp(self):
        self.monitor = RequestMonitor()
    
    def test_log_request(self):
        """Test logging requests."""
        self.monitor.log_request("/test", 200, 0.5, "127.0.0.1")
        self.monitor.log_request("/test", 404, 0.3, "127.0.0.1")
        
        stats = self.monitor.get_stats()
        self.assertEqual(stats["total_requests"], 2)
        self.assertEqual(stats["failed_requests"], 1)
    
    def test_success_rate(self):
        """Test success rate calculation."""
        self.monitor.log_request("/test", 200, 0.5, "127.0.0.1")
        self.monitor.log_request("/test", 200, 0.3, "127.0.0.1")
        self.monitor.log_request("/test", 500, 0.2, "127.0.0.1")
        
        stats = self.monitor.get_stats()
        self.assertAlmostEqual(stats["success_rate"], 66.666, places=1)


class TestTokenCounting(unittest.TestCase):
    """Test token counting utilities."""
    
    def test_count_tokens(self):
        """Test token approximation."""
        text = "This is a test"
        tokens = count_tokens_approximate(text)
        # Should be roughly len(text) / 4
        self.assertGreater(tokens, 0)
        self.assertLess(tokens, len(text))


class TestConversationHistory(unittest.TestCase):
    """Test conversation history management."""
    
    def test_manage_history_within_limit(self):
        """Test that history is maintained within token limits."""
        global conversation_history
        from src import chat_engine
        
        # Create messages within limit
        chat_engine.conversation_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
        
        # Should not remove anything
        manage_conversation_history(max_tokens=1000)
        
        self.assertEqual(len(chat_engine.conversation_history), 2)
    
    def test_manage_history_exceeds_limit(self):
        """Test that old messages are removed when exceeding limit."""
        from src import chat_engine
        
        # Create many large messages
        chat_engine.conversation_history = [
            {"role": "user", "content": "a" * 2000},
            {"role": "assistant", "content": "b" * 2000},
            {"role": "user", "content": "c" * 2000},
            {"role": "assistant", "content": "d" * 2000}
        ]
        
        # Manage with small token limit
        manage_conversation_history(max_tokens=500)
        
        # Should have removed oldest messages
        self.assertLess(len(chat_engine.conversation_history), 4)


# ==================== Integration Tests ====================

class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    @patch('src.helper.web_scrape_url')
    def test_url_processing_flow(self, mock_scrape):
        """Test URL processing pipeline."""
        mock_scrape.return_value = {
            "text": "Sample content for testing",
            "metadata": {
                "page_title": "Test Page",
                "page_type": "course",
                "department": "CS",
                "heading_path": ["Heading 1"],
                "last_modified": "2026-01-25",
                "content_length": 28
            }
        }
        
        # Test would go through full pipeline
        # This is a placeholder for comprehensive integration testing
        self.assertTrue(True)


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == "__main__":
    run_tests()

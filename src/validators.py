import re
from typing import Dict, List, Any
from src import logger

class InputValidator:
    
    MAX_QUERY_LENGTH = 5000
    MAX_URL_LENGTH = 2048
    MAX_URLS_PER_REQUEST = 100
    
    FORBIDDEN_PATTERNS = [
        r'<script',
        r'javascript:',
        r'onerror=',
        r'onclick=',
        r'<iframe',
        r'eval\(',
        r'__import__',
        r'exec\(',
        r'subprocess',
    ]
    
    @staticmethod
    def validate_query(query: str) -> tuple[bool, str, str]:
        if not query:
            return False, "", "Query cannot be empty"
        
        if not isinstance(query, str):
            return False, "", "Query must be a string"
        
        if len(query) > InputValidator.MAX_QUERY_LENGTH:
            return False, "", f"Query exceeds maximum length of {InputValidator.MAX_QUERY_LENGTH}"
        
        query_lower = query.lower()
        for pattern in InputValidator.FORBIDDEN_PATTERNS:
            if re.search(pattern, query_lower):
                logger.warning(f"Malicious pattern detected in query: {pattern}")
                return False, "", "Invalid query content"
        
        sanitized = query.strip()
        
        return True, sanitized, ""
    
    @staticmethod
    def validate_url(url: str) -> tuple[bool, str]:
        if not url or not isinstance(url, str):
            return False, "URL must be a non-empty string"
        
        if len(url) > InputValidator.MAX_URL_LENGTH:
            return False, f"URL exceeds maximum length of {InputValidator.MAX_URL_LENGTH}"
        
        url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        if not re.match(url_pattern, url, re.IGNORECASE):
            return False, "Invalid URL format"
        
        url_lower = url.lower()
        for pattern in InputValidator.FORBIDDEN_PATTERNS:
            if re.search(pattern, url_lower):
                return False, f"Malicious pattern detected in URL"
        
        return True, ""
    
    @staticmethod
    def validate_urls_list(urls: List[str]) -> tuple[bool, List[str], List[str]]:
        if not isinstance(urls, list):
            return False, [], ["URLs must be a list"]
        
        if len(urls) > InputValidator.MAX_URLS_PER_REQUEST:
            return False, [], [f"Too many URLs. Maximum is {InputValidator.MAX_URLS_PER_REQUEST}"]
        
        valid_urls = []
        errors = []
        
        for i, url in enumerate(urls):
            is_valid, error = InputValidator.validate_url(url)
            if is_valid:
                valid_urls.append(url)
            else:
                errors.append(f"URL {i+1}: {error}")
        
        return len(errors) == 0, valid_urls, errors


def sanitize_string(text: str, max_length: int = 5000) -> str:
    if not isinstance(text, str):
        return ""
    
    text = text.strip()
    if len(text) > max_length:
        text = text[:max_length]
    
    return text


def validate_json_request(data: Dict[str, Any], required_fields: List[str]) -> tuple[bool, str]:
    if not isinstance(data, dict):
        return False, "Request body must be JSON"
    
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
    
    return True, ""

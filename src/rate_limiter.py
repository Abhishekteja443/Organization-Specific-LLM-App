import threading
import time
from typing import Dict, Tuple
from collections import defaultdict
from src import logger


class RateLimiter:
    
    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests =max_requests
        self.window_seconds =window_seconds
        self.requests =defaultdict(list) 
        self.lock =threading.Lock()
    
    def is_allowed(self, ip: str) -> Tuple[bool, Dict]:
        try:
            with self.lock:
                now =time.time()
                
                if ip in self.requests:
                    self.requests[ip] =[
                        ts for ts in self.requests[ip] 
                        if now - ts < self.window_seconds
                    ]
                if len(self.requests[ip]) < self.max_requests:
                    self.requests[ip].append(now)
                    remaining = self.max_requests - len(self.requests[ip])
                    return True, {
                        "remaining": remaining,
                        "reset_in": self.window_seconds
                    }
                else:
                    oldest =self.requests[ip][0]
                    reset_in =int(self.window_seconds - (now - oldest))
                    logger.warning(f"Rate limit exceeded for IP: {ip}")
                    return False, {
                        "remaining": 0,
                        "reset_in": reset_in
                    }
        except Exception as e:
            logger.error(f"Rate limiter error: {e}")
            return True, {"remaining": -1, "reset_in": -1}
    
    def reset_ip(self, ip: str):
        with self.lock:
            if ip in self.requests:
                del self.requests[ip]
                logger.info(f"Rate limit reset for IP: {ip}")


class RequestMonitor:
    def __init__(self):
        self.total_requests = self.failed_requests = self.avg_response_time = 0
        self.response_times = []
        self.lock = threading.Lock()
        self.max_history = 1000
    
    def log_request(self, endpoint: str, status_code: int, response_time: float, ip: str):
        try:
            with self.lock:
                self.total_requests += 1
                if status_code >= 400:
                    self.failed_requests += 1
                
                self.response_times.append(response_time)
                if len(self.response_times) > self.max_history:
                    self.response_times.pop(0)
                
                self.avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
                if status_code >= 400:
                    logger.warning(
                        f"Request failed: {endpoint} | Status: {status_code} | "
                        f"Time: {response_time:.2f}s | IP: {ip}"
                    )
        except Exception as e:
            logger.error(f"Monitor logging error: {e}")
    
    def get_stats(self) -> Dict:
        with self.lock:
            return {
                "total_requests": self.total_requests,
                "failed_requests": self.failed_requests,
                "success_rate": (
                    (self.total_requests - self.failed_requests) / self.total_requests * 100
                ) if self.total_requests > 0 else 0,
                "avg_response_time": round(self.avg_response_time, 3)
            }
    
    def reset(self):
        with self.lock:
            self.total_requests = 0
            self.failed_requests = 0
            self.response_times = []
            self.avg_response_time = 0
            logger.info("Monitor statistics reset")


rate_limiter = RateLimiter(max_requests=100, window_seconds=60)
request_monitor = RequestMonitor()

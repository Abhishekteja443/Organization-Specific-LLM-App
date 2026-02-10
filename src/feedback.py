"""
Feedback and analytics system for tracking response quality
"""
import json
import os
from datetime import datetime, timezone
from typing import Dict, List
from threading import Lock
from src import logger

FEEDBACK_DIR = "feedback"
FEEDBACK_FILE = os.path.join(FEEDBACK_DIR, "responses_feedback.json")


class FeedbackCollector:
    """Collect and store user feedback on responses."""
    
    def __init__(self):
        self.lock = Lock()
        os.makedirs(FEEDBACK_DIR, exist_ok=True)
        self._load_feedback()
    
    def _load_feedback(self):
        """Load existing feedback from disk."""
        if os.path.exists(FEEDBACK_FILE):
            try:
                with open(FEEDBACK_FILE, 'r') as f:
                    self.feedback_data = json.load(f)
                logger.info(f"Loaded {len(self.feedback_data)} feedback entries")
            except Exception as e:
                logger.error(f"Error loading feedback: {e}")
                self.feedback_data = []
        else:
            self.feedback_data = []
    
    def add_feedback(self, query: str, response: str, source_url: str, 
                     rating: int, comment: str = "", user_id: str = "anonymous") -> bool:
        """
        Add feedback entry.
        
        Args:
            query: User's original query
            response: Generated response
            source_url: Source URL for the response
            rating: Rating 1-5 (1=poor, 5=excellent)
            comment: Optional user comment
            user_id: Anonymous or user identifier
        
        Returns:
            bool: Success status
        """
        try:
            if not 1 <= rating <= 5:
                logger.warning(f"Invalid rating: {rating}")
                return False
            
            with self.lock:
                feedback_entry = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "query": query[:500],  # Limit length
                    "response": response[:1000],
                    "source_url": source_url,
                    "rating": rating,
                    "comment": comment[:500] if comment else "",
                    "user_id": user_id
                }
                
                self.feedback_data.append(feedback_entry)
                self._save_feedback()
                
                logger.info(f"Feedback recorded: rating={rating}, user={user_id}")
                return True
        
        except Exception as e:
            logger.error(f"Error adding feedback: {e}", exc_info=True)
            return False
    
    def _save_feedback(self):
        """Save feedback to disk."""
        try:
            with open(FEEDBACK_FILE, 'w') as f:
                json.dump(self.feedback_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
    
    def get_analytics(self) -> Dict:
        """Get feedback analytics."""
        try:
            with self.lock:
                if not self.feedback_data:
                    return {
                        "total_feedback": 0,
                        "average_rating": 0,
                        "rating_distribution": {}
                    }
                
                ratings = [f["rating"] for f in self.feedback_data]
                
                rating_dist = {}
                for r in range(1, 6):
                    rating_dist[str(r)] = sum(1 for rating in ratings if rating == r)
                
                return {
                    "total_feedback": len(self.feedback_data),
                    "average_rating": sum(ratings) / len(ratings),
                    "rating_distribution": rating_dist,
                    "latest_feedback": self.feedback_data[-10:]  # Last 10
                }
        except Exception as e:
            logger.error(f"Error getting analytics: {e}")
            return {}
    
    def export_feedback(self, format: str = "json") -> str:
        """Export feedback data."""
        try:
            with self.lock:
                if format == "json":
                    return json.dumps(self.feedback_data, indent=2)
                else:
                    logger.warning(f"Unsupported export format: {format}")
                    return ""
        except Exception as e:
            logger.error(f"Error exporting feedback: {e}")
            return ""


# Global instance
feedback_collector = FeedbackCollector()

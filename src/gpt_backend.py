from src.faiss_manager import faiss_manager
from src.cache import cached_result
from src import logger

@cached_result(ttl=3600)
def retrieve_relevant_chunks(query, top_n=4):
    try:
        return faiss_manager.retrieve_relevant_chunks(query, top_n)
    except Exception as e:
        logger.error(f"Error retrieving chunks: {e}", exc_info=True)
        return "", "Unknown"

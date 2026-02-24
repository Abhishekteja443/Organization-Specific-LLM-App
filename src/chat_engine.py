import json
import ollama
import threading
from src.query_router import route_query
from src.prompt_builder import build_system_prompt
from src import gpt_backend
from src import logger

conversation_history = []
history_lock = threading.Lock()
MAX_TOKENS_PER_MESSAGE = 500
MAX_CONTEXT_TOKENS = 4000


def count_tokens_approximate(text: str) -> int:
    return len(text) // 4


def manage_conversation_history(max_tokens: int = MAX_CONTEXT_TOKENS):
    global conversation_history
    
    total_tokens = sum(count_tokens_approximate(msg.get("content", "")) for msg in conversation_history)
    
    while total_tokens > max_tokens and len(conversation_history) > 1:
        if len(conversation_history) >= 2:
            removed_msg = conversation_history.pop(0)
            total_tokens -= count_tokens_approximate(removed_msg.get("content", ""))
            logger.info(f"Removed message from history to manage tokens. Total: {total_tokens}")


def stream_chat_response(query: str):
    global conversation_history
    
    try:
        if not query or not isinstance(query, str):
            logger.warning("Invalid query received")
            yield "Error: Invalid query", None
            return
        
        if len(query) > 5000:
            logger.warning("Query exceeds maximum length")
            yield "Error: Query is too long", None
            return
        
        route = route_query(query)
        
        if route == "meta":
            response = (
                "I do not have persistent memory or awareness of documents. "
                "I only see information provided to me during this conversation."
            )
            with history_lock:
                conversation_history.append({"role": "user", "content": query})
                conversation_history.append({"role": "assistant", "content": response})
                manage_conversation_history()
            yield response, None
            return
        
        relevant_text = ""
        source_url = None
        
        if route != "chitchat":
            try:
                relevant_text, source_url = gpt_backend.retrieve_relevant_chunks(query,5)
                if relevant_text:
                    logger.info(f"Retrieved context from: {source_url}")
            except Exception as e:
                logger.error(f"Error retrieving chunks: {e}")
                relevant_text = ""
                source_url = "Unknown"
        
        system_prompt = build_system_prompt(route)
        
        with history_lock:
            manage_conversation_history()
            
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(conversation_history)
            
            if relevant_text:
                messages.append({
                    "role": "system",
                    "content": f"Context:\n{relevant_text}"
                })
            
            messages.append({"role": "user", "content": query})

        response_text = ""
        
        try:
            for chunk in ollama.chat(
                model="llama3.2:3b",
                messages=messages,
                options={"temperature": 0.5},
                stream=True
            ):
                if "message" in chunk and "content" in chunk["message"]:
                    content = chunk["message"]["content"]
                    response_text += content
                    yield content, source_url
        except Exception as e:
            logger.error(f"LLM streaming error: {e}", exc_info=True)
            error_msg = "Error generating response. Please try again."
            yield error_msg, None
            response_text = error_msg

        with history_lock:
            conversation_history.append({"role": "user", "content": query})
            conversation_history.append({"role": "assistant", "content": response_text})
            manage_conversation_history()
    
    except Exception as e:
        logger.error(f"Unexpected error in stream_chat_response: {e}", exc_info=True)
        yield "An unexpected error occurred. Please try again.", None


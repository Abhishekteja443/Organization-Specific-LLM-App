import json
import ollama
from src.query_router import route_query
from src.prompt_builder import build_system_prompt
from src import gpt_backend

conversation_history = []

def stream_chat_response(query: str):
    global conversation_history

    route = route_query(query)
    if route == "meta":
        response = (
            "I do not have persistent memory or awareness of documents. "
            "I only see information provided to me during this conversation."
        )
        yield response, None
        return

    # Retrieve only if needed
    if route == "chitchat":
        relevant_text = ""
        source_url = None
    else:
        relevant_text, source_url = gpt_backend.retrieve_relevant_chunks(query)

    system_prompt = build_system_prompt(route)

    # Trim history
    if len(conversation_history) > 6:
        conversation_history = conversation_history[-6:]

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)

    if relevant_text:
        messages.append({
            "role": "system",
            "content": f"Context:\n{relevant_text}"
        })

    messages.append({"role": "user", "content": query})

    response_text = ""

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

    # Save history
    conversation_history.append({"role": "user", "content": query})
    conversation_history.append({"role": "assistant", "content": response_text})

from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from src import helper,gpt_backend
import ollama
import json

app = Flask(__name__)
conversation_history=[]

# Original routes
@app.route("/")
def admin_panel():
    return render_template("index.html")

@app.route("/submit-urls", methods=["POST"])
def receive_urls():
    stored_urls = set()
    data = request.json
    base_urls = data.get("base_urls", [])
    extra_urls = set(data.get("extra_urls", []))
    stored_urls = stored_urls | extra_urls

    for i in base_urls:
        stored_urls = stored_urls | helper.fetch_urls_from_sitemap(i)
    
    unscraped_urls = helper.process_urls(stored_urls)

    print(f"Received Base URLs: {base_urls}")
    print(f"Received Extra URLs: {extra_urls}")

    result_message = f"Received {len(base_urls)} Base URLs and {len(extra_urls)} Additional URLs."
    
    return {
        "message": result_message, 
        "submitted_urls": list(stored_urls),
        "unscraped_urls": list(unscraped_urls)
    }

# New routes for Organization-GPT
@app.route("/organization-gpt")
def org_gpt_interface():
    return render_template("organization-gpt.html")

@app.route("/chat", methods=["POST"])
@app.route("/chat-stream", methods=["GET"])
def handle_chat_stream():
    query = request.args.get("query", "")
    
    if not query:
        return jsonify({"error": "Empty query received"})
    
    def generate():
        relevant_text, source_url = gpt_backend.retrieve_relevant_chunks(query)
        
        # Get conversation history and system prompt ready
        global conversation_history
        system_prompt = """
            You are an AI assistant. 
            1. **Prioritize conversation history first**. 
            2. **Only use the retrieved FAISS context if needed**.
            3. **If the history contains relevant context, continue the conversation naturally**.
            4. If the history does not contain relevant information, then refer to FAISS context.

            Conversation history will be provided first.  
            FAISS-retrieved information will be clearly separated.
            """
        
        # Keep only the last N exchanges to avoid excessive memory use
        if len(conversation_history) > 3:  
            conversation_history = conversation_history[-3:]
        
        # Construct the chat history for Ollama
        messages = [{"role": "system", "content": system_prompt}]

        messages.append({"role": "user", "content": f"Previous Conversation:\n{json.dumps(conversation_history, indent=2)}"})

        messages.append({"role": "user", "content": f"FAISS Retrieved Context (Use Only If Needed):\n{relevant_text} source Url: {source_url}"})

        messages.append({"role": "user", "content": f"User Query:\n{query}"})

        
        try:
            # Use Ollama's streaming capability
            response_text = ""
            for chunk in ollama.chat(
                model="llama3.2:3b",
                messages=messages,
                options={"temperature": 0.5},
                stream=True  # Enable streaming
            ):
                if 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    response_text += content
                    # Send each chunk as an SSE event
                    yield f"data: {json.dumps({'content': content, 'source_url': source_url})}\n\n"
            
            # Add user query and full assistant response to conversation history
            conversation_history.append({"role": "user", "content": query})
            conversation_history.append({"role": "assistant", "content": response_text})
            
            # Send a final empty chunk to signal completion
            yield f"data: {json.dumps({'content': '', 'source_url': source_url, 'done': True})}\n\n"
        
        except Exception as e:
            error_msg = f"Could not generate a response from the LLM. {str(e)}"
            yield f"data: {json.dumps({'content': error_msg, 'error': True})}\n\n"
    
    return Response(stream_with_context(generate()), content_type='text/event-stream')
if __name__ == "__main__":
    app.run(debug=True)
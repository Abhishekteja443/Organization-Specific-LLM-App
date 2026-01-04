from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from src import helper
from src.chat_engine import stream_chat_response
import json

app = Flask(__name__)

@app.route("/")
def admin_panel():
    return render_template("index.html")

@app.route("/submit-urls", methods=["POST"])
def receive_urls():
    stored_urls = set()
    data = request.json

    base_urls = data.get("base_urls", [])
    extra_urls = set(data.get("extra_urls", []))
    stored_urls |= extra_urls

    for url in base_urls:
        stored_urls |= helper.fetch_urls_from_sitemap(url)

    unscraped_urls = helper.process_urls(stored_urls)

    return {
        "message": f"Received {len(base_urls)} Base URLs and {len(extra_urls)} Additional URLs.",
        "submitted_urls": list(stored_urls),
        "unscraped_urls": list(unscraped_urls)
    }

@app.route("/organization-gpt")
def org_gpt_interface():
    return render_template("organization-gpt.html")

@app.route("/chat-stream", methods=["GET"])
def chat_stream():
    query = request.args.get("query", "")
    if not query:
        return jsonify({"error": "Empty query"})

    def generate():
        for content, source_url in stream_chat_response(query):
            yield f"data: {json.dumps({'content': content, 'source_url': source_url})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

    return Response(stream_with_context(generate()), content_type="text/event-stream")

if __name__ == "__main__":
    app.run(debug=True)

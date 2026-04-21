from dotenv import load_dotenv

load_dotenv()
from flask import Flask, render_template, request, jsonify, Response, stream_with_context, redirect, url_for
from flask_cors import CORS
import os
import time
import json
from datetime import datetime
from functools import wraps
import threading
import hashlib
from src import helper, logger
from src.chat_engine import stream_chat_response
from src.validators import InputValidator, validate_json_request
from src.faiss_manager import faiss_manager
import redis

redis_client = redis.Redis(
    host="my-rag-valkey-urievg.serverless.use2.cache.amazonaws.com",
    port=6379,
    decode_responses=True,
    ssl = True
)
print(redis_client)
#Initialize Flask app

app = Flask(__name__)

# Security configuration
app.config['JSON_SORT_KEYS'] = False
app.config['MAX_CONTENT_LENGTH'] =16 * 1024 * 1024

#Enable CORS with restricted origins
CORS(app, resources={
    r"/api/*": {
        "origins": os.getenv("CORS_ORIGINS", "http://localhost:5000").split(","),
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }})

#Debug mode from environment
DEBUG_MODE =os.getenv("FLASK_DEBUG","False").lower() =="true"




def log_request(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time =time.time()
        endpoint =request.endpoint or "unknown"
        
        try:
            response =f(*args, **kwargs)
            response_time =time.time() - start_time
            status_code =response.status_code if hasattr(response, 'status_code') else 200
            return response
        except Exception as e:
            response_time = time.time() -start_time
            logger.error(f"Unhandled error in {endpoint}: {e}",exc_info=True)
            raise
    
    return decorated_function


@app.route("/", methods=["GET"])
@log_request
def admin_panel():
    try:
        return redirect(url_for("login"))
    except Exception as e:
        logger.error(f"Error redirecting to login:{e}")
        return jsonify({"error": "Failed to redirect to login"}),500
    
@app.route("/index", methods=["GET"])
@log_request
def index_panel():
    try:
        return render_template("index.html")
    except Exception as e:
        logger.error(f"Error rendering admin panel:{e}")
        return jsonify({"error": "Failed to load admin panel"}),500


@app.route("/submit-urls", methods=["POST"])
@log_request
def receive_urls():
    try:
        data =request.json or {}
        
        is_valid, error =validate_json_request(data, ["base_urls", "extra_urls"])
        if not is_valid:
            return jsonify({"error": error}), 400
        
        base_urls =data.get("base_urls", [])
        extra_urls= data.get("extra_urls", [])
        
        if not isinstance(base_urls, list) or not isinstance(extra_urls, list):
            return jsonify({"error": "base_urls and extra_urls must be lists"}), 400
        
        stored_urls = set()
        
        for url in base_urls:
            is_valid, error_msg = InputValidator.validate_url(url)
            if not is_valid:
                logger.warning(f"Invalid base URL:{url} - {error_msg}")
                continue
            
            sitemap_urls =helper.fetch_urls_from_sitemap(url)
            stored_urls.update(sitemap_urls)
        
        all_valid, valid_urls, errors =InputValidator.validate_urls_list(extra_urls)
        if errors:
            logger.warning(f"Invalid extra URLs: {errors}")

        # Single URL pass
        stored_urls.update(valid_urls)

        # Nested URL pass for fetching URLs in URL domains
        # for url in valid_urls:
        #     try:
        #         domain_urls, graph =helper.fetch_urls_from_domain(url)
        #         print(graph)
        #         stored_urls.update(domain_urls)
        #     except Exception as e:
        #         logger.warning(f"Failed to fetch URLs from domain {url}:{e}")
        
        if not stored_urls:
            return jsonify({
                "error": "No valid URLs to process",
                "validation_errors": errors
            }), 400
        
        logger.info(f"Processing {len(stored_urls)} URLs")
        
        unscraped_urls = helper.process_urls(stored_urls)
        
        return jsonify({
            "message": f"Successfully processed {len(stored_urls)} URLs for indexing!",
            "total_urls": len(stored_urls),
            "unscraped_urls": list(unscraped_urls),
            "unscraped_count": len(unscraped_urls),
            "next_action": "Open the chat interface to start asking questions",
            "chat_url": "/organization-gpt",
            "success": True
        }), 202
    
    except Exception as e:
        logger.error(f"Error in receive_urls: {e}",exc_info=True)
        return jsonify({"error": "Failed to process URLs"}), 500


@app.route("/organization-gpt", methods=["GET"])
@log_request
def org_gpt_interface():
    try:
        return render_template("organization-gpt.html")
    except Exception as e:
        logger.error(f"Error rendering chat interface: {e}")
        return jsonify({"error": "Failed to load chat interface"}), 500


@app.route("/login", methods=["GET", "POST"])
@log_request
def login():
    try:
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "").strip()

            valid_users = {
                "admin": "admin",
                "w868axg": "56452312"
            }

            if username in valid_users and valid_users[username] == password:
                if username == "admin":
                    return redirect(url_for("index_panel"))
                return redirect(url_for("org_gpt_interface"))

            error = "Invalid username or password."
            return render_template("login.html", error=error, username=username)

        return render_template("login.html")
    except Exception as e:
        logger.error(f"Error rendering login page: {e}")
        return jsonify({"error": "Failed to load login page"}), 500


@app.route("/chat-stream", methods=["GET"])
@log_request
def chat_stream():
    try:
        query =request.args.get("query", "").strip()

        is_valid, sanitized_query, error =InputValidator.validate_query(query)
        if not is_valid:
            return jsonify({"error": error}), 400

        logger.info(f"Chat query received:{sanitized_query[:100]}...")

        def generate():
            try:

                cache_key = "llm:" + hashlib.md5(sanitized_query.encode()).hexdigest()

                try:
                    cached = redis_client.get(cache_key)
                except Exception as e:
                    logger.warning(f"Cache read failed: {e}")
                    cached = None

                if cached:
                    logger.info(f"Cache HIT: {sanitized_query[:50]}")
                    yield f"data: {json.dumps({'content': cached, 'source_url': 'cache'})}\n\n"
                    yield f"data: {json.dumps({'done': True})}\n\n"
                    return
                
                logger.info(f"Cache MISS: {sanitized_query[:50]}")
                full_response = ""
                last_source_url = None

                for content, source_url in stream_chat_response(sanitized_query):
                    full_response += content
                    last_source_url = source_url
                    yield f"data: {json.dumps({'content': content, 'source_url': source_url})}\n\n"

                yield f"data: {json.dumps({'done': True})}\n\n"

                def save_to_cache():
                    try:
                        redis_client.setex(cache_key, 3600, full_response)
                        logger.info(f"Cached response for: {sanitized_query[:50]}")
                    except Exception as e:
                        logger.warning(f"Cache write failed: {e}")

                threading.Thread(target=save_to_cache, daemon=True).start()

            except Exception as e:
                logger.error(f"Error during streaming: {e}", exc_info=True)
                yield f"data: {json.dumps({'error': 'Stream error occurred'})}\n\n"
        
        # def generate():
        #     try:
        #         for content, source_url in stream_chat_response(sanitized_query):
        #             event_data ={
        #                 'content': content,
        #                 'source_url': source_url
        #             }
        #             yield f"data: {json.dumps(event_data)}\n\n"

        #         yield f"data: {json.dumps({'done': True})}\n\n"
        #     except Exception as e:
        #         logger.error(f"Error during streaming: {e}", exc_info=True)
        #         yield f"data: {json.dumps({'error': 'Stream error occurred'})}\n\n"

        # return Response(stream_with_context(generate()), content_type="text/event-stream")

    except Exception as e:
        logger.error(f"Error in chat_stream: {e}", exc_info=True)
        return jsonify({"error": "Failed to process chat request"}), 500

@app.route("/api/health", methods=["GET"])
@log_request
def health_check():
    try:
        faiss_stats =faiss_manager.get_index_stats()
        
        return jsonify({
            "status": "healthy",
            "faiss_index": faiss_stats,
            "version": "1.0.0"
        }), 200
    
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 503



# @app.route("/api/reindex", methods=["POST"])
# @log_request
# def reindex():
#     try:
#         data = request.json or {}
#         urls = data.get("urls", [])
#         clear_existing = data.get("clear_existing", False)
        
#         if not urls:
#             return jsonify({"error": "No URLs provided"}), 400
        
#         # Validate URLs
#         all_valid, valid_urls, errors = InputValidator.validate_urls_list(urls)
#         if not all_valid:
#             return jsonify({
#                 "error": "Some URLs are invalid",
#                 "validation_errors": errors
#             }), 400
        
#         if clear_existing:
#             logger.info("Clearing existing index")
#             faiss_manager.all_documents = []
#             faiss_manager.all_embeddings = []
#             faiss_manager.all_metadatas = []
#             faiss_manager.all_ids = []
#             faiss_manager.url_to_chunks = {}
#             faiss_manager.save_metadata()
        
#         logger.info(f"Starting re-indexing of {len(valid_urls)} URLs")
#         unscraped = helper.process_urls(set(valid_urls))
        
#         return jsonify({
#             "message": "Re-indexing started",
#             "processed_urls": len(valid_urls),
#             "unscraped_urls": list(unscraped),
#             "unscraped_count": len(unscraped)
#         }), 202
    
#     except Exception as e:
#         logger.error(f"Error in reindex: {e}", exc_info=True)
#         return jsonify({"error": "Failed to start re-indexing"}), 500


@app.errorhandler(400)
def bad_request(e):
    """Handle 400 errors."""
    logger.warning(f"Bad request: {e}")
    return jsonify({"error": "Bad request"}), 400


@app.errorhandler(404)
def not_found(e):
    request_path = request.path
    if request_path in ['/favicon.ico', '/.well-known/appspecific/com.chrome.devtools.json']:
        return "", 404
    
    logger.warning(f"Not found: {request_path} - {e}")
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal error: {e}", exc_info=True)
    return jsonify({"error": "Internal server error"}), 500



if __name__ == "__main__":
    logger.info("Starting Organization-Specific LLM App")
    logger.info(f"Debug mode: {DEBUG_MODE}")
    logger.info(f"CORS origins: {os.getenv('CORS_ORIGINS', 'http://localhost:5000')}")
    app.run(debug=DEBUG_MODE, host="0.0.0.0", port=5000)


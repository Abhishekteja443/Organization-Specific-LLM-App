from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import os
import time
import json
from datetime import datetime
from functools import wraps

from src import helper, logger
from src.chat_engine import stream_chat_response
from src.validators import InputValidator, validate_json_request
from src.rate_limiter import rate_limiter, request_monitor
from src.faiss_manager import faiss_manager
from src.cache import query_cache

#Initialize Flask app
app = Flask(__name__)

# Security configuration
app.config['JSON_SORT_KEYS'] = False
app.config['MAX_CONTENT_LENGTH'] =16 * 1024 * 1024  # 16MB max request size

#Enable CORS with restricted origins
CORS(app, resources={
    r"/api/*": {
        "origins": os.getenv("CORS_ORIGINS", "http://localhost:5000").split(","),
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }})

#Debug mode from environment
DEBUG_MODE =os.getenv("FLASK_DEBUG","False").lower() =="true"


def require_rate_limit(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        ip =request.remote_addr
        allowed, limits= rate_limiter.is_allowed(ip)
        
        if not allowed:
            logger.warning(f"Rate limit exceeded for IP: {ip}")
            response = jsonify({
                "error": "Rate limit exceeded",
                "reset_in": limits["reset_in"]})
            response.status_code =429
            return response
        
        return f(*args,**kwargs)
    
    return decorated_function


def log_request(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time =time.time()
        endpoint =request.endpoint or "unknown"
        
        try:
            response =f(*args, **kwargs)
            response_time =time.time() - start_time
            status_code =response.status_code if hasattr(response, 'status_code') else 200
            request_monitor.log_request(endpoint, status_code, response_time, request.remote_addr)
            return response
        except Exception as e:
            response_time = time.time() -start_time
            request_monitor.log_request(endpoint, 500, response_time, request.remote_addr)
            logger.error(f"Unhandled error in {endpoint}: {e}",exc_info=True)
            raise
    
    return decorated_function


@app.route("/", methods=["GET"])
@log_request
def admin_panel():
    try:
        return render_template("index.html")
    except Exception as e:
        logger.error(f"Error rendering admin panel:{e}")
        return jsonify({"error": "Failed to load admin panel"}),500


@app.route("/submit-urls", methods=["POST"])
@require_rate_limit
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


@app.route("/chat-stream", methods=["GET"])
@require_rate_limit
@log_request
def chat_stream():
    try:
        query =request.args.get("query", "").strip()
        
        #Validate query
        is_valid, sanitized_query, error =InputValidator.validate_query(query)
        if not is_valid:
            return jsonify({"error": error}), 400
        
        logger.info(f"Chat query received:{sanitized_query[:100]}...")
        
        def generate():
            try:
                for content, source_url in stream_chat_response(sanitized_query):
                    event_data ={
                        'content': content,
                        'source_url': source_url
                    }
                    yield f"data: {json.dumps(event_data)}\n\n"
                
                yield f"data: {json.dumps({'done': True})}\n\n"
            except Exception as e:
                logger.error(f"Error during streaming: {e}", exc_info=True)
                yield f"data: {json.dumps({'error': 'Stream error occurred'})}\n\n"
        
        return Response(stream_with_context(generate()), content_type="text/event-stream")
    
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


@app.route("/api/stats", methods=["GET"])
@log_request
def get_stats():
    try:
        faiss_stats =faiss_manager.get_index_stats()
        request_stats= request_monitor.get_stats()
        
        return jsonify({
            "faiss": faiss_stats,
            "requests": request_stats
        }), 200
    
    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        return jsonify({"error": "Failed to retrieve stats"}), 500


@app.route("/api/cache-stats", methods=["GET"])
@log_request
def cache_stats():
    try:
        return jsonify({
            "cache_size": len(query_cache.cache),
            "max_size": query_cache.max_size,
            "ttl_seconds": query_cache.ttl
        }), 200
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        return jsonify({"error": "Failed to retrieve cache stats"}), 500


@app.route("/api/cache-clear", methods=["POST"])
@log_request
def clear_cache():
    try:
        query_cache.clear()
        return jsonify({"message": "Cache cleared"}), 200
    except Exception as e:
        logger.error(f"Error clearing cache: {e}",exc_info=True)
        return jsonify({"error": "Failed to clear cache"}), 500



@app.route("/api/reindex", methods=["POST"])
@log_request
def reindex():
    try:
        data = request.json or {}
        urls = data.get("urls", [])
        clear_existing = data.get("clear_existing", False)
        
        if not urls:
            return jsonify({"error": "No URLs provided"}), 400
        
        # Validate URLs
        all_valid, valid_urls, errors = InputValidator.validate_urls_list(urls)
        if not all_valid:
            return jsonify({
                "error": "Some URLs are invalid",
                "validation_errors": errors
            }), 400
        
        if clear_existing:
            logger.info("Clearing existing index")
            faiss_manager.all_documents = []
            faiss_manager.all_embeddings = []
            faiss_manager.all_metadatas = []
            faiss_manager.all_ids = []
            faiss_manager.url_to_chunks = {}
            faiss_manager.save_metadata()
        
        logger.info(f"Starting re-indexing of {len(valid_urls)} URLs")
        unscraped = helper.process_urls(set(valid_urls))
        
        return jsonify({
            "message": "Re-indexing started",
            "processed_urls": len(valid_urls),
            "unscraped_urls": list(unscraped),
            "unscraped_count": len(unscraped)
        }), 202
    
    except Exception as e:
        logger.error(f"Error in reindex: {e}", exc_info=True)
        return jsonify({"error": "Failed to start re-indexing"}), 500


@app.route("/api/index-status", methods=["GET"])
@log_request
def index_status():
    try:
        stats = faiss_manager.get_index_stats()
        
        return jsonify({
            **stats,
            "timestamp": datetime.now().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Error getting index status: {e}")
        return jsonify({"error": "Failed to get index status"}), 500


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


@app.errorhandler(429)
def rate_limit_exceeded(e):
    return jsonify({"error": "Rate limit exceeded"}), 429


@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal error: {e}", exc_info=True)
    return jsonify({"error": "Internal server error"}), 500



if __name__ == "__main__":
    logger.info("Starting Organization-Specific LLM App")
    logger.info(f"Debug mode: {DEBUG_MODE}")
    logger.info(f"CORS origins: {os.getenv('CORS_ORIGINS', 'http://localhost:5000')}")
    app.run(debug=DEBUG_MODE, host="0.0.0.0", port=5000)


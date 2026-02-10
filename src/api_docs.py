from flask import Blueprint, jsonify
import json

api_docs_bp = Blueprint('api_docs', __name__, url_prefix='/api')

OPENAPI_SPEC = {
    "openapi": "3.0.0",
    "info": {
        "title": "Organization-Specific LLM API",
        "description": "Private LLM system for organizations with RAG capabilities",
        "version": "1.0.0",
        "contact": {
            "name": "Organization LLM Team"
        }
    },
    "servers": [
        {
            "url": "http://localhost:5000",
            "description": "Development server"
        }
    ],
    "paths": {
        "/submit-urls": {
            "post": {
                "summary": "Submit URLs for scraping and indexing",
                "tags": ["Data Ingestion"],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "base_urls": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "List of sitemap URLs to crawl",
                                        "example": ["https://example.com/sitemap.xml"]
                                    },
                                    "extra_urls": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "Additional URLs to include",
                                        "example": ["https://example.com/page1"]
                                    }
                                },
                                "required": ["base_urls", "extra_urls"]
                            }
                        }
                    }
                },
                "responses": {
                    "202": {
                        "description": "URLs accepted for processing",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "message": {"type": "string"},
                                        "total_urls": {"type": "integer"},
                                        "unscraped_urls": {"type": "array", "items": {"type": "string"}},
                                        "unscraped_count": {"type": "integer"}
                                    }
                                }
                            }
                        }
                    },
                    "400": {"description": "Invalid request"},
                    "429": {"description": "Rate limit exceeded"},
                    "500": {"description": "Server error"}
                }
            }
        },
        "/chat-stream": {
            "get": {
                "summary": "Stream chat responses",
                "tags": ["Chat"],
                "parameters": [
                    {
                        "name": "query",
                        "in": "query",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "User query"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Server-Sent Events stream",
                        "content": {
                            "text/event-stream": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "content": {"type": "string"},
                                        "source_url": {"type": "string"},
                                        "done": {"type": "boolean"}
                                    }
                                }
                            }
                        }
                    },
                    "400": {"description": "Invalid query"},
                    "429": {"description": "Rate limit exceeded"},
                    "500": {"description": "Server error"}
                }
            }
        },
        "/health": {
            "get": {
                "summary": "Health check",
                "tags": ["System"],
                "responses": {
                    "200": {
                        "description": "System is healthy",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "status": {"type": "string"},
                                        "faiss_index": {
                                            "type": "object",
                                            "properties": {
                                                "total_documents": {"type": "integer"},
                                                "total_chunks": {"type": "integer"},
                                                "total_urls": {"type": "integer"},
                                                "index_size": {"type": "integer"}
                                            }
                                        },
                                        "version": {"type": "string"}
                                    }
                                }
                            }
                        }
                    },
                    "503": {"description": "Service unavailable"}
                }
            }
        },
        "/stats": {
            "get": {
                "summary": "Get system statistics",
                "tags": ["System"],
                "responses": {
                    "200": {
                        "description": "Statistics",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "faiss": {"type": "object"},
                                        "requests": {"type": "object"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/cache-stats": {
            "get": {
                "summary": "Get cache statistics",
                "tags": ["System"],
                "responses": {
                    "200": {
                        "description": "Cache statistics",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object"
                                }
                            }
                        }
                    }
                }
            }
        },
        "/cache-clear": {
            "post": {
                "summary": "Clear query cache",
                "tags": ["System"],
                "responses": {
                    "200": {"description": "Cache cleared"}
                }
            }
        }
    },
    "components": {
        "securitySchemes": {
            "rate_limit": {
                "type": "apiKey",
                "in": "header",
                "name": "X-Rate-Limit-Remaining"
            }
        }
    },
    "tags": [
        {
            "name": "Data Ingestion",
            "description": "Endpoints for submitting URLs and data"
        },
        {
            "name": "Chat",
            "description": "Chat and query endpoints"
        },
        {
            "name": "System",
            "description": "System health and statistics"
        }
    ]
}


@api_docs_bp.route('/docs/openapi.json')
def openapi_json():
    """Serve OpenAPI specification."""
    return jsonify(OPENAPI_SPEC)


@api_docs_bp.route('/docs')
def swagger_ui():
    """Serve Swagger UI documentation."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Organization-LLM API Docs</title>
        <meta charset="utf-8"/>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link href="https://fonts.googleapis.com/css?family=Montserrat:300,400,700|Roboto:300,400,700" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/4.15.5/swagger-ui.min.css">
        <style>
          html{ box-sizing: border-box; overflow: -moz-scrollbars-vertical; overflow-y: scroll; }
          *, *:before, *:after { box-sizing: inherit; }
          body { margin:0; background: #fafafa; }
        </style>
    </head>
    <body>
        <div id="swagger-ui"></div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/4.15.5/swagger-ui.min.js"></script>
        <script>
          SwaggerUIBundle({
            url: "/api/docs/openapi.json",
            dom_id: '#swagger-ui',
            presets: [
              SwaggerUIBundle.presets.apis,
              SwaggerUIBundle.SwaggerUIStandalonePreset
            ],
            layout: "StandaloneLayout"
          })
        </script>
    </body>
    </html>
    """


def register_api_docs(app):
    """Register API documentation blueprint with Flask app."""
    app.register_blueprint(api_docs_bp)

# Organization Specific LLM APP🤖

## Description
This project is built to ensure data security and confidentiality by allowing organizations to create their own LLM (Large Language Model) instead of relying on online services like ChatGPT or Claude. The application processes various multimedia data formats such as SQL databases, URLs, Excel sheets, Word documents, and PDFs to make them usable for GPT-based interactions.

### Features:
- **Retrieval-Augmented Generation (RAG)** with semantic search
- **Query caching** for 50-80% performance improvement
- **Rate limiting** and comprehensive security
- **Chunk deduplication** and intelligent indexing
- **Token-based conversation** history management
- **Ollama Nomic Embeddings** for semantic embeddings
- **FAISS** for efficient vector storage
- **Llama 3.2** for streaming responses
- **Request monitoring** and analytics
- **Batch re-indexing** capability

## Setup and Installation

### 1. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use 'venv\Scripts\activate'
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download and Install Models
Install **Ollama** and download the required models:
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 4. Configure Environment
```bash
cp .env.example .env
# Edit .env with your settings
```

Key variables:
```
FLASK_DEBUG=False  
FAISS_INDEX_PATH=./faiss_store
CORS_ORIGINS=http://localhost:5000
RATE_LIMIT_REQUESTS=100 
```

### 5. Run the Application
```bash
python app.py
```

Access the app at:
- **Admin Panel**: `http://localhost:5000/`
- **Chat Interface**: `http://localhost:5000/organization-gpt`

## API Endpoints

### Data Ingestion
- `POST /submit-urls` - Submit URLs for indexing
- `POST /api/reindex` - Re-index specific URLs

### Chat & Retrieval
- `GET /chat-stream` - Stream chat responses with sources
- `POST /api/feedback` - Submit response feedback

### System & Monitoring
- `GET /api/health` - Health check
- `GET /api/stats` - System statistics
- `GET /api/index-status` - Detailed index information
- `GET /api/cache-stats` - Cache statistics
- `POST /api/cache-clear` - Clear query cache

## Configuration

### Rate Limiting
Default: 100 requests per minute per IP address

Adjust in `.env`:
```
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
```

### Query Caching
Default: 500 queries cached for 1 hour

Adjust in `.env`:
```
CACHE_MAX_SIZE=500
CACHE_TTL=3600
```

### Conversation History
Default: 4000 tokens max context

Adjust in `.env`:
```
MAX_CONTEXT_TOKENS=4000
```
### FAISS Index Issues
```bash
rm -rf faiss_store/
python app.py
# Resubmit URLs through admin panel
```

### Model Not Loading
Ensure Ollama is running:
```bash
ollama serve
# In another terminal:
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### Rate Limit Exceeded
Reduce request frequency or adjust `RATE_LIMIT_REQUESTS` in `.env`

## Contributing
Feel free to fork this repository and contribute enhancements via pull requests.

## License
This project is licensed under the **MIT License**.




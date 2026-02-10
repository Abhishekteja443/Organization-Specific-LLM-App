import os
import json
import numpy as np
import faiss
import threading
from typing import Tuple, List, Dict
from src import logger
import src

class FAISSManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.METADATA_PATH = src.METADATA_PATH
        self.INDEX_DATA_FILE = src.INDEX_DATA_FILE
        self.INDEX_FAISS_FILE = src.INDEX_FAISS_FILE
        
        self.index = None
        self.all_documents = []
        self.all_embeddings = []
        self.all_metadatas = []
        self.all_ids = []
        self.url_to_chunks = {}
        
        self.data_lock = threading.RLock()  # Use RLock to allow reentrant locking
        self._initialized = True
        
        # Ensure FAISS index directory exists and create empty metadata file if missing
        try:
            index_dir = os.path.dirname(self.METADATA_PATH) or "."
            os.makedirs(index_dir, exist_ok=True)
            if not os.path.exists(self.METADATA_PATH):
                with open(self.METADATA_PATH, 'w') as mf:
                    json.dump({}, mf)
                logger.info(f"Created empty metadata file at {self.METADATA_PATH}")
        except Exception as e:
            logger.error(f"Error ensuring FAISS index directory/metadata: {e}", exc_info=True)

        self._load_metadata()
        self._load_index()
    
    def _load_metadata(self):
        """Load metadata tracking URL to chunk mappings."""
        if os.path.exists(self.METADATA_PATH):
            try:
                with open(self.METADATA_PATH, 'r') as f:
                    self.url_to_chunks = json.load(f)
                logger.info(f"Loaded metadata with {len(self.url_to_chunks)} URLs")
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading metadata: {e}")
                self.url_to_chunks = {}
        else:
            self.url_to_chunks = {}
    
    def _load_index(self):
        """Load FAISS index and embeddings from disk."""
        if not os.path.exists(self.INDEX_FAISS_FILE) or not os.path.exists(self.INDEX_DATA_FILE):
            logger.info("No existing FAISS index found. Will create on first save.")
            return
        
        try:
            with open(self.INDEX_DATA_FILE, 'r') as f:
                index_data = json.load(f)
            
            self.all_documents = index_data.get("documents", [])
            embeddings_list = index_data.get("embeddings", [])
            self.all_embeddings = [np.array(e, dtype=np.float32) for e in embeddings_list]
            self.all_metadatas = index_data.get("metadatas", [])
            self.all_ids = index_data.get("ids", [])
            
            if self.all_embeddings:
                dimension = len(self.all_embeddings[0])
                nlist = min(len(self.all_embeddings) // 10, 200)
                
                quantizer = faiss.IndexFlatL2(dimension)
                self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
                self.index = faiss.IndexIDMap(self.index)
                self.index.train(np.array(self.all_embeddings, dtype=np.float32))
                self.index.add(np.array(self.all_embeddings, dtype=np.float32))
                # Increase nprobe for better retrieval accuracy (trades speed for quality)
                self.index.nprobe = min(20, nlist // 2)
                
                logger.info(f"FAISS Index Loaded: {self.index.ntotal} vectors, {nlist} clusters, nprobe={self.index.nprobe}")
            else:
                logger.warning("No embeddings found in index data")
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}", exc_info=True)
            self.all_documents = []
            self.all_embeddings = []
            self.all_metadatas = []
            self.all_ids = []
            self.index = None
    
    def save_metadata(self):
        """Save metadata tracking URL to chunk mappings."""
        try:
            logger.info("Starting metadata save...")
            with self.data_lock:
                logger.info("Acquired data lock for metadata save")
                # Ensure directory exists before writing
                meta_dir = os.path.dirname(self.METADATA_PATH) or "."
                logger.info(f"Metadata directory: {meta_dir}")
                os.makedirs(meta_dir, exist_ok=True)
                logger.info(f"Directory created/verified: {meta_dir}")
                with open(self.METADATA_PATH, 'w') as f:
                    logger.info(f"Opening metadata file for writing: {self.METADATA_PATH}")
                    json.dump(self.url_to_chunks, f, indent=2)
                    logger.info(f"Metadata written: {len(self.url_to_chunks)} URLs")
            logger.info(f"Metadata saved successfully to {self.METADATA_PATH}")
        except IOError as e:
            logger.error(f"Error saving metadata: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error saving metadata: {e}", exc_info=True)
    
    def save_index(self, retrain_threshold=0.1):
        """Efficiently save FAISS index using Approximate Nearest Neighbor (ANN)."""
        try:
            with self.data_lock:
                if not self.all_embeddings:
                    logger.warning("No embeddings to save. Skipping index save.")
                    return
                
                index_data = {
                    "documents": self.all_documents,
                    "embeddings": [e.tolist() if isinstance(e, np.ndarray) else e for e in self.all_embeddings],
                    "metadatas": self.all_metadatas,
                    "ids": self.all_ids
                }
                # Ensure index directory exists before writing files
                index_dir = os.path.dirname(self.INDEX_DATA_FILE) or "."
                os.makedirs(index_dir, exist_ok=True)

                with open(self.INDEX_DATA_FILE, 'w') as f:
                    json.dump(index_data, f)

                embedding_array = np.array(self.all_embeddings, dtype=np.float32)
                dimension = embedding_array.shape[1]
                
                retrain_needed = len(embedding_array) > retrain_threshold * self.index.ntotal if self.index else True
                nlist = min(len(embedding_array) // 10, 200)
                
                if retrain_needed:
                    logger.info("Retraining FAISS index due to large data update.")
                    quantizer = faiss.IndexFlatL2(dimension)
                    self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
                    self.index.train(embedding_array)
                else:
                    logger.info("Adding new vectors without retraining.")
                
                self.index.add(embedding_array)
                # Increase nprobe for better retrieval accuracy
                self.index.nprobe = min(20, nlist // 2)
                # Ensure faiss file directory exists and write index
                faiss_dir = os.path.dirname(self.INDEX_FAISS_FILE) or "."
                os.makedirs(faiss_dir, exist_ok=True)
                faiss.write_index(self.index, self.INDEX_FAISS_FILE)

                logger.info(f"Saved FAISS index file to {self.INDEX_FAISS_FILE}")
                logger.info(f"Saved FAISS index with {len(self.all_documents)} documents and nlist={nlist}, nprobe={self.index.nprobe}")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}", exc_info=True)
    
    def check_and_delete_chunks(self, url: str):
        """Delete all chunks associated with a specific URL from memory & index."""
        try:
            with self.data_lock:
                if url in self.url_to_chunks and self.url_to_chunks[url]:
                    chunk_ids = self.url_to_chunks[url]
                    logger.info(f"Found {len(chunk_ids)} existing chunks for {url}. Deleting...")
                    
                    new_documents = []
                    new_embeddings = []
                    new_metadatas = []
                    new_ids = []
                    
                    for i, doc_id in enumerate(self.all_ids):
                        if doc_id not in chunk_ids:
                            new_documents.append(self.all_documents[i])
                            new_embeddings.append(self.all_embeddings[i])
                            new_metadatas.append(self.all_metadatas[i])
                            new_ids.append(doc_id)
                    
                    self.all_documents = new_documents
                    self.all_embeddings = new_embeddings
                    self.all_metadatas = new_metadatas
                    self.all_ids = new_ids
                    del self.url_to_chunks[url]
                    self.save_metadata()
                    
                    logger.info(f"Deleted all chunks for {url}.")
                else:
                    logger.info(f"No existing chunks found for {url}.")
        except Exception as e:
            logger.error(f"Error deleting chunks for {url}: {e}", exc_info=True)
    
    def retrieve_relevant_chunks(self, query: str, top_n: int = 10) -> Tuple[str, str]:
        """
        Retrieve relevant chunks from FAISS with distance-based filtering.
        
        Args:
            query: User query text
            top_n: Number of top candidates to retrieve (default 10, higher = more thorough)
        
        Returns:
            Tuple of (combined_chunks, source_url)
        """
        try:
            import ollama
            
            if self.index is None or len(self.all_documents) == 0:
                logger.warning("FAISS index is empty or not loaded")
                return "", "Unknown"
            
            # Generate query embedding
            query_embedding = ollama.embeddings(model="nomic-embed-text:latest", prompt=query)["embedding"]
            query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
            
            # Search FAISS index with expanded top_n for better filtering
            distances, indices = self.index.search(query_vector, min(top_n, len(self.all_documents)))
            
            # Filter by distance threshold (lower = more similar, L2 distance)
            # Typically < 1.5 indicates good semantic similarity
            distance_threshold = 1.5
            
            retrieved_chunks = []
            retrieved_metadatas = []
            retrieved_scores = []
            
            for idx, distance in zip(indices[0], distances[0]):
                print(f"Index: {idx}, Distance: {distance}")
                if idx < len(self.all_documents) and distance > distance_threshold:
                    relevance_score = 1.0 / (1.0 + distance)
                    retrieved_chunks.append(self.all_documents[idx])
                    retrieved_metadatas.append(self.all_metadatas[idx])
                    retrieved_scores.append(relevance_score)
            
            # If no chunks pass the threshold, return top 3 anyway
            if not retrieved_chunks:
                logger.info(f"No chunks passed distance threshold (< {distance_threshold}), returning top 3 by distance")
                for idx in indices[0][:3]:
                    if idx < len(self.all_documents):
                        retrieved_chunks.append(self.all_documents[idx])
                        retrieved_metadatas.append(self.all_metadatas[idx])
                        retrieved_scores.append(0.0)
            
            # Get most common source URL
            source_url_list = [meta.get("source_url", "Unknown") for meta in retrieved_metadatas]
            source_url = max(set(source_url_list), key=source_url_list.count) if source_url_list else "Unknown"
            print(retrieved_chunks)
            
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks (threshold: {distance_threshold}, top scores: {[f'{s:.2f}' for s in retrieved_scores[:3]]})")
            
            return "\n\n".join(retrieved_chunks), source_url
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}", exc_info=True)
            return "", "Unknown"
    
    def add_chunks(self, url: str, chunks: List[str], embeddings: List, metadatas: List[Dict]):
        """Add chunks to the index."""
        try:
            with self.data_lock:
                logger.info(f"Adding {len(chunks)} chunks for {url}")
                url_chunks = []
                for i, (chunk, embedding, metadata) in enumerate(zip(chunks, embeddings, metadatas)):
                    chunk_id = f"{url}_chunk_{len(self.all_ids) + i + 1}"
                    url_chunks.append(chunk_id)
                    
                    self.all_documents.append(chunk)
                    self.all_embeddings.append(np.array(embedding, dtype=np.float32))
                    self.all_metadatas.append(metadata)
                    self.all_ids.append(chunk_id)
                
                self.url_to_chunks[url] = url_chunks
                self.save_metadata()
                logger.info(f"Successfully added {len(url_chunks)} chunks for {url}")
                
                logger.info(f"Added {len(url_chunks)} chunks for {url}")
        except Exception as e:
            logger.error(f"Error adding chunks for {url}: {e}", exc_info=True)
    
    def get_index_stats(self) -> Dict:
        """Get current index statistics."""
        with self.data_lock:
            return {
                "total_documents": len(self.all_documents),
                "total_chunks": len(self.all_ids),
                "total_urls": len(self.url_to_chunks),
                "index_size": self.index.ntotal if self.index else 0
            }


faiss_manager = FAISSManager()

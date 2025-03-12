import ollama
import faiss
import numpy as np
import json
import os
import re
import threading
import src



# FAISS and LLM code
METADATA_PATH=src.METADATA_PATH
INDEX_DATA_FILE=src.INDEX_DATA_FILE
INDEX_FAISS_FILE=src.INDEX_FAISS_FILE

index_lock = threading.Lock()

# Initialize global variables
index = None
all_documents = []
all_metadatas = []
conversation_history = []

# Load FAISS index and metadata
def load_index():
    """Load FAISS index into memory once."""
    global index, all_documents, all_metadatas
    if index is None and os.path.exists(INDEX_FAISS_FILE) and os.path.exists(INDEX_DATA_FILE):
        with open(INDEX_DATA_FILE, 'r') as f:
            index_data = json.load(f)
        all_documents = index_data["documents"]
        all_embeddings = np.array(index_data["embeddings"], dtype=np.float32)
        all_metadatas = index_data["metadatas"]
        dimension = all_embeddings.shape[1]
        nlist = min(len(all_embeddings) // 10, 200)  # Adjust clusters dynamically
        # Load FAISS index
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
        index.train(all_embeddings)
        index.add(all_embeddings)
        index.nprobe = min(10, nlist // 2)  # Search within n clusters
        print(f"FAISS Index Loaded: {index.ntotal} vectors, {nlist} clusters")

def get_index():
    """Return the FAISS index, ensuring it's loaded."""
    global index
    with index_lock:  # Ensure thread safety
        if index is None:
            load_index()
    return index

def retrieve_relevant_chunks(query, top_n=4):
    """Retrieve relevant chunks from FAISS based on user query."""
    faiss_index = get_index()
    query_embedding = ollama.embeddings(model="nomic-embed-text:latest", prompt=query)["embedding"]
    query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
    distances, indices = faiss_index.search(query_vector, top_n)
    
    retrieved_chunks = [all_documents[i] for i in indices[0] if i < len(all_documents)]
    retrieved_metadatas = [all_metadatas[i] for i in indices[0] if i < len(all_metadatas)]
    
    source_url_list = [meta["source_url"] for meta in retrieved_metadatas]
    source_url = max(set(source_url_list), key=source_url_list.count) if source_url_list else "Unknown"
    return "\n\n".join(retrieved_chunks), source_url

# def remove_markdown_links(response_text):
#     """Converts Markdown links [text](URL) into plain text URLs."""
#     return re.sub(r"\[(.*?)\]\((https?://.*?)\)", r"\2", response_text)

# def ask_llm(query: str, context: str, source_url: str):
#     """Send the user query and retrieved chunks to the LLM and get a response."""
#     global conversation_history  # Use the session memory
#     system_prompt = """
#     You are an AI assistant answering questions based on the provided context.
#     Use clear and concise language. If context is insufficient, say so.
#     """
#     # Keep only the last N exchanges to avoid excessive memory use
#     if len(conversation_history) > 5:  
#         conversation_history = conversation_history[-5:]  
#     # Construct the chat history in the format Ollama expects
#     messages = [{"role": "system", "content": system_prompt}]
#     messages.extend(conversation_history)  # Add chat history
#     messages.append({"role": "user", "content": f"Context:\n{context}\n\nSource URL:\n{source_url}\n\nQuery: {query}"})
    
#     try:
#         response = ollama.chat(
#             model="llama3.2:3b",
#             messages=messages,
#             options={"temperature": 0.5}  # Control randomness
#         )
#         answer = remove_markdown_links(response.get("message", {}).get("content", "No response generated"))
#         # Add user query and assistant response to memory
#         conversation_history.append({"role": "user", "content": query})
#         conversation_history.append({"role": "assistant", "content": answer})
#         return answer
#     except Exception as e:
#         return f"Could not generate a response from the LLM. {e}"

# def generate_final_output(user_query):
#     """Retrieve relevant text and generate final output using Llama 3.2."""
#     relevant_text, source_url = retrieve_relevant_chunks(user_query)
#     return ask_llm(user_query, relevant_text, source_url), source_url
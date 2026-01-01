import requests
from bs4 import BeautifulSoup
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime, timezone
import time
from xml.etree import ElementTree
from typing import List
import psutil
import os
import json
import numpy as np
import faiss
import gc
import re
from concurrent.futures import ThreadPoolExecutor
from src import logger
import src

METADATA_PATH=src.METADATA_PATH
INDEX_DATA_FILE=src.INDEX_DATA_FILE
INDEX_FAISS_FILE=src.INDEX_FAISS_FILE

# Load metadata tracking
if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH, 'r') as f:
        try:
            url_to_chunks = json.load(f)
        except json.JSONDecodeError:
            url_to_chunks = {}
else:
    url_to_chunks = {}

# Initialize FAISS-related variables
index = None
all_documents = []
all_embeddings = []
all_metadatas = []
all_ids = []

session = requests.Session()

unscraped_urls=[]

def fetch_urls_from_sitemap(sitemap_url: str) -> List[str]:
    """
    Recursively fetches all URLs from a sitemap or nested sitemaps.
    
    Args:
        sitemap_url (str): The root or nested sitemap URL.
    
    Returns:
        List[str]: List of all URLs to be crawled.
    """
    try:
        logger.info(f"Fetching URLs from sitemap: {sitemap_url}")
        response = requests.get(sitemap_url)
        response.raise_for_status()  # Raise exception for non-2xx responses
        root = ElementTree.fromstring(response.content)
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        
        # Check for nested sitemaps
        sitemap_elements = root.findall('.//ns:sitemap/ns:loc', namespace)
        if sitemap_elements:
            nested_urls = [fetch_urls_from_sitemap(sitemap.text) for sitemap in sitemap_elements]
            return set([url for sublist in nested_urls for url in sublist])
        
        # Otherwise, extract all URLs
        url_elements = root.findall('.//ns:url/ns:loc', namespace)
        return set(url.text for url in url_elements)
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching sitemap {sitemap_url}: {e}")
        return set()
    except ElementTree.ParseError as e:
        logger.error(f"Error parsing sitemap XML {sitemap_url}: {e}")
        return set()
    except Exception as e:
        logger.error(f"Unexpected error while fetching sitemap {sitemap_url}: {e}")
        return set()

def save_metadata():
    """Save metadata tracking URL to chunk mappings."""
    with open(METADATA_PATH, 'w') as f:
        json.dump(url_to_chunks, f)



def save_index(retrain_threshold=0.1):
    """Efficiently save FAISS index using Approximate Nearest Neighbor (ANN)."""
    global index, all_documents, all_embeddings, all_metadatas, all_ids

    if not all_embeddings:
        logger.warning("No embeddings to save. Skipping index save.")
        return

    index_data = {
        "documents": all_documents,
        "embeddings": [embedding.tolist() if isinstance(embedding, np.ndarray) else embedding for embedding in all_embeddings],
        "metadatas": all_metadatas,
        "ids": all_ids
    }
    
    with open(INDEX_DATA_FILE, 'w') as f:
        json.dump(index_data, f)

    # Convert embeddings to NumPy array for FAISS
    embedding_array = np.array(all_embeddings, dtype=np.float32)
    dimension = embedding_array.shape[1]
    
    # Determine if retraining is needed
    retrain_needed = len(embedding_array) > retrain_threshold * index.ntotal if index else True
    nlist = min(len(embedding_array) // 10, 200)  

    if retrain_needed:
        logger.info("Retraining FAISS index due to large data update.")
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
        index.train(embedding_array)
    else:
        logger.info("Adding new vectors without retraining.")
    
    index.add(embedding_array)
    index.nprobe = min(10, nlist // 2)
    faiss.write_index(index, INDEX_FAISS_FILE)

    logger.info(f"Saved FAISS index with {len(all_documents)} documents and nlist={nlist}")



def check_and_delete_chunks(url):
    """Delete all chunks associated with a specific URL from memory & index."""
    global all_documents, all_embeddings, all_metadatas, all_ids

    if url in url_to_chunks and url_to_chunks[url]:
        chunk_ids = url_to_chunks[url]
        logger.info(f"Found {len(chunk_ids)} existing chunks for {url}. Deleting...")

        new_documents = []
        new_embeddings = []
        new_metadatas = []
        new_ids = []

        for i, doc_id in enumerate(all_ids):
            if doc_id not in chunk_ids:
                new_documents.append(all_documents[i])
                new_embeddings.append(all_embeddings[i])
                new_metadatas.append(all_metadatas[i])
                new_ids.append(doc_id)

        all_documents, all_embeddings, all_metadatas, all_ids = new_documents, new_embeddings, new_metadatas, new_ids
        del url_to_chunks[url]
        save_metadata()

        if len(all_embeddings) > 0:
            dimension = len(all_embeddings[0])
            index = faiss.IndexIVFFlat(faiss.IndexFlatL2(dimension), dimension, min(len(all_embeddings) // 10, 200) if len(all_embeddings) > 500 else 5 , faiss.METRIC_L2)
            index.train(np.array(all_embeddings, dtype=np.float32))
            index.add(np.array(all_embeddings, dtype=np.float32))
        else:
            index = None

        logger.info(f"Deleted all chunks for {url}.")
    else:
        logger.info(f"No existing chunks found for {url}. Proceeding with insertion.")



def web_scrape_url(url):
    global unscraped_urls
    try:
        logger.info(f"Scraping URL: {url}")
        response = session.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, "html.parser")

        scraped_data = []
        seen_texts = set()
        
        # Extract only relevant sections (modify the selector if needed)
        for tag in soup.find_all(["p", "span", "h1", "h2", "h3", "h4", "h5", "h6", "a", "li", "td"]):
            text = tag.get_text(strip=True)
        
            # Handling anchor tags separately for links
            if tag.name == "a" and tag.get("href"):
                href = tag.get("href")
                icon = tag.find("i")
                if icon:
                    text = icon.get("class", [])
                    text = " ".join(text)
                    pattern = r"fa-([a-zA-Z0-9\-]+)"
                    text = re.findall(pattern, text)
                    text = " ".join(text)
                if text and (text, href) not in seen_texts:
                    seen_texts.add((text, href))
                    scraped_data.append(f"{text}: {href}")
            else:
                if text and text not in seen_texts:
                    seen_texts.add(text)
                    scraped_data.append(text)
        scraped_data=" ".join(scraped_data)
        return scraped_data
    
    except requests.exceptions.RequestException as e:
        unscraped_urls.append(url)
        logger.error(f"Error scraping URL {url}: {e}")
        return ""
    

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """Split text into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)



def embed_texts(chunks):
    """Parallel embedding of text chunks using Ollama API."""
    with ThreadPoolExecutor() as executor:
        embeddings = list(executor.map(lambda chunk: ollama.embeddings(model="nomic-embed-text:latest", prompt=chunk)["embedding"], chunks))
    return embeddings





def process_single_url(url):
    """Scrape, chunk, embed, and store data for a single URL."""
    global all_documents, all_embeddings, all_metadatas, all_ids

    try:
        check_and_delete_chunks(url)
        logger.info(f"Processing URL: {url}")

        data = web_scrape_url(url)
        if not data:
            logger.warning(f"No data found for {url}. Skipping...")
            return

        chunks = chunk_text(data)
        if not chunks:
            logger.warning(f"No chunks created for {url}. Skipping...")
            return

        url_chunks = []
        embeddings = embed_texts(chunks)  # Use parallel embeddings

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{url}_chunk_{i+1}"
            url_chunks.append(chunk_id)

            all_documents.append(chunk)
            all_embeddings.append(np.array(embedding, dtype=np.float16))  # Reduce memory
            all_metadatas.append({
                "source_url": url,
                "chunk_size": len(chunk),
                "crawled_at": datetime.now(timezone.utc).isoformat()
            })
            all_ids.append(chunk_id)

        url_to_chunks[url] = url_chunks
        save_metadata()
        logger.info(f"Successfully processed {len(url_chunks)} chunks for {url}")

    except Exception as e:
        logger.error(f"Error processing URL {url}: {e}", exc_info=True)

def process_urls(urls):
    global unscraped_urls
    try:
        """Process URLs in parallel using multiple threads."""
        num_workers = min(10, os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:  # Adjust workers based on CPU cores
            executor.map(process_single_url, urls)
        gc.collect()
        save_index()
        print(unscraped_urls)

        return unscraped_urls
   

    except Exception as e:
                    logger.error(e, exc_info=True)
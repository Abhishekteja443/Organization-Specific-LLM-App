import requests
from bs4 import BeautifulSoup
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime, timezone
import time
from xml.etree import ElementTree
from typing import List, Set
import os
import json
import numpy as np
import gc
import re
from concurrent.futures import ThreadPoolExecutor
from src import logger
from src.faiss_manager import faiss_manager
from threading import Lock
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

data_lock = Lock()

session = requests.Session()

# Configure retry strategy for robustness
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)
session.headers.update({"User-Agent": "Mozilla/5.0 (Organization-LLM-App)"})

unscraped_urls = []


def fetch_urls_from_sitemap(sitemap_url: str, max_retries: int = 3) -> Set[str]:
    """
    Recursively fetches all URLs from a sitemap or nested sitemaps.
    Includes retry logic for robustness.
    
    Args:
        sitemap_url (str): The root or nested sitemap URL.
        max_retries (int): Number of retries for failed requests.
    
    Returns:
        Set[str]: Set of all URLs to be crawled.
    """
    try:
        logger.info(f"Fetching URLs from sitemap: {sitemap_url}")
        
        response = session.get(sitemap_url, timeout=10)
        response.raise_for_status()
        
        root = ElementTree.fromstring(response.content)
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        
        # Check for nested sitemaps
        sitemap_elements = root.findall('.//ns:sitemap/ns:loc', namespace)
        if sitemap_elements:
            nested_urls = [fetch_urls_from_sitemap(sitemap.text) for sitemap in sitemap_elements]
            return set([url for sublist in nested_urls for url in sublist])
        
        # Otherwise, extract all URLs
        url_elements = root.findall('.//ns:url/ns:loc', namespace)
        urls = set(url.text for url in url_elements if url.text)
        logger.info(f"Extracted {len(urls)} URLs from sitemap")
        return urls
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching sitemap {sitemap_url}: {e}")
        return set()
    except ElementTree.ParseError as e:
        logger.error(f"Error parsing sitemap XML {sitemap_url}: {e}")
        return set()
    except Exception as e:
        logger.error(f"Unexpected error while fetching sitemap {sitemap_url}: {e}", exc_info=True)
        return set()




def web_scrape_url(url: str) -> dict:
    """
    Scrape content from URL with improved error handling and timeouts.
    
    Returns:
        dict: Contains 'text' and 'metadata' keys, or empty dict on failure
    """
    try:
        logger.info(f"Scraping URL: {url}")
        response = session.get(url, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        page_title = soup.title.string.strip() if soup.title else "Unknown"
        
        # Infer page type (heuristics)
        page_type = "generic"
        if any(x in url.lower() for x in ["admission", "enroll", "apply"]):
            page_type = "admissions"
        elif any(x in url.lower() for x in ["course", "curriculum", "catalog"]):
            page_type = "course"
        elif any(x in url.lower() for x in ["policy", "rule", "regulation"]):
            page_type = "policy"
        
        # Infer department
        department = "Unknown"
        if any(x in url.lower() for x in ["cs", "computer", "engineering"]):
            department = "Computer Science"
        elif "admission" in url.lower():
            department = "Admissions"
        
        # Extract headings
        headings = [tag.get_text(strip=True) for tag in soup.find_all(["h1", "h2", "h3"])[:5]]
        
        # Extract text with deduplication
        seen_texts = set()
        scraped_data = []
        
        for tag in soup.find_all(["p", "li", "td"]):
            text = tag.get_text(strip=True)
            if text and len(text) > 3 and text not in seen_texts:
                seen_texts.add(text)
                scraped_data.append(text)
        
        # Add meaningful anchor text
        for tag in soup.find_all("a"):
            anchor_text = tag.get_text(strip=True)
            if (anchor_text and len(anchor_text) > 3 and 
                anchor_text.lower() not in ["click here", "read more", "learn more"] and
                anchor_text not in seen_texts):
                seen_texts.add(anchor_text)
                scraped_data.append(anchor_text)
        
        text_content = " ".join(scraped_data)
        
        if not text_content:
            logger.warning(f"No text content extracted from {url}")
            return {}
        
        return {
            "text": text_content,
            "metadata": {
                "page_title": page_title,
                "page_type": page_type,
                "department": department,
                "heading_path": headings,
                "last_modified": response.headers.get("Last-Modified", "Unknown"),
                "content_length": len(text_content)
            }
        }
    
    except requests.exceptions.Timeout:
        logger.warning(f"Timeout scraping URL {url}")
        with data_lock:
            unscraped_urls.append(url)
        return {}
    except requests.exceptions.RequestException as e:
        logger.error(f"Error scraping URL {url}: {e}")
        with data_lock:
            unscraped_urls.append(url)
        return {}
    except Exception as e:
        logger.error(f"Unexpected error scraping URL {url}: {e}", exc_info=True)
        with data_lock:
            unscraped_urls.append(url)
        return {}


def chunk_text(text: str, chunk_size: int = 600, chunk_overlap: int = 150) -> List[str]:
    """Split text into smaller chunks with overlap for better retrieval accuracy."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(text)
    
    # Remove any empty chunks
    chunks = [c.strip() for c in chunks if c.strip()]
    
    logger.info(f"Created {len(chunks)} chunks from text")
    return chunks


def deduplicate_chunks(chunks: List[str]) -> List[str]:
    """Remove duplicate chunks."""
    seen = set()
    unique_chunks = []
    
    for chunk in chunks:
        # Use hash of chunk to detect duplicates
        chunk_hash = hash(chunk.lower())
        if chunk_hash not in seen:
            seen.add(chunk_hash)
            unique_chunks.append(chunk)
    
    if len(unique_chunks) < len(chunks):
        logger.info(f"Deduplicated {len(chunks) - len(unique_chunks)} chunks")
    
    return unique_chunks


def embed_texts(chunks: List[str]) -> List:
    """Parallel embedding of text chunks using Ollama API."""
    try:
        with ThreadPoolExecutor(max_workers=5) as executor:
            embeddings = list(executor.map(
                lambda chunk: ollama.embeddings(
                    model="nomic-embed-text:latest", 
                    prompt=chunk
                )["embedding"],
                chunks
            ))
        return embeddings
    except Exception as e:
        logger.error(f"Error embedding texts: {e}", exc_info=True)
        return []


def process_single_url(url: str):
    """Scrape, chunk, embed, and store data for a single URL."""
    try:
        faiss_manager.check_and_delete_chunks(url)
        logger.info(f"Processing URL: {url}")
        
        result = web_scrape_url(url)
        if not result or not result.get("text"):
            logger.warning(f"No data found for {url}. Skipping...")
            return
        
        text = result["text"]
        page_metadata = result["metadata"]
        chunks = chunk_text(text)
        
        if not chunks:
            logger.warning(f"No chunks created for {url}. Skipping...")
            return
        
        # Deduplicate chunks
        chunks = deduplicate_chunks(chunks)
        embeddings = embed_texts(chunks)

        if len(embeddings) != len(chunks):
            logger.error(f"Embedding count mismatch for {url}")
            return
        
        # Build metadata for each chunk
        chunk_metadatas = [
            {
                "source_url": url,
                "chunk_id": f"{url}_chunk_{i+1}",
                "page_title": page_metadata["page_title"],
                "page_type": page_metadata["page_type"],
                "department": page_metadata["department"],
                "heading_path": page_metadata["heading_path"],
                "chunk_size": len(chunk),
                "crawled_at": datetime.now(timezone.utc).isoformat(),
                "last_modified": page_metadata["last_modified"]
            }
            for i, chunk in enumerate(chunks)
        ]
        
        faiss_manager.add_chunks(url, chunks, embeddings, chunk_metadatas)

        logger.info(f"Successfully processed {len(chunks)} chunks for {url}")
    
    except Exception as e:
        logger.error(f"Error processing URL {url}: {e}", exc_info=True)


def process_urls(urls: Set[str]) -> List[str]:
    """Process URLs in parallel using multiple threads."""
    global unscraped_urls
    
    try:
        logger.info(f"Starting to process {len(urls)} URLs")
        num_workers = min(5, os.cpu_count() or 2)
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Wait for all futures to complete to ensure chunks/embeddings are added
            list(executor.map(process_single_url, urls))

        gc.collect()
        logger.info("All URL processing threads completed. Saving FAISS index.")
        faiss_manager.save_index()
        logger.info("FAISS index save attempted.")
        
        logger.info(f"Processing complete. Unscraped URLs: {len(unscraped_urls)}")
        return unscraped_urls
    
    except Exception as e:
        logger.error(f"Error in process_urls: {e}", exc_info=True)
        return []

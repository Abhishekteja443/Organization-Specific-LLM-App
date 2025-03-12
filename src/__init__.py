import os
import sys
import logging
from dotenv import load_dotenv, dotenv_values
load_dotenv()

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_dir = "logs"
log_filepath = os.path.join(log_dir,"running_logs.log")
os.makedirs(log_dir, exist_ok=True)


logging.basicConfig(
    level= logging.INFO,
    format= logging_str,

    handlers=[
        logging.FileHandler(log_filepath)
        # logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("src")

FAISS_INDEX_PATH=os.getenv("FAISS_INDEX_PATH")
METADATA_PATH = os.path.join(FAISS_INDEX_PATH, "metadata.json")
INDEX_DATA_FILE = os.path.join(FAISS_INDEX_PATH, "index_data.json")
INDEX_FAISS_FILE = os.path.join(FAISS_INDEX_PATH, "index.faiss")

# Create directory if it doesn't exist
os.makedirs(FAISS_INDEX_PATH, exist_ok=True)

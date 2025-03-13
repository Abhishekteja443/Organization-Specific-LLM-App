# Organization Specific LLM APP🤖

## Description
This project is built to ensure data security and confidentiality by allowing organizations to create their own LLM (Large Language Model) instead of relying on online services like ChatGPT or Claude. The application processes various multimedia data formats such as SQL databases, URLs, Excel sheets, Word documents, and PDFs to make them usable for GPT-based interactions.

### Features:
- Retrieval-Augmented Generation (RAG) implementation for URLs.
- Full GPT-based application functionality.
- Uses **Ollama Nomic Embeddings** for embeddings.
- Utilizes **FAISS** for efficient vector storage.
- Employs **Llama 3.2** for model responses.

## Setup and Installation
Follow the steps below to install and run the application:

### 1. Create a Virtual Environment
It is recommended to use a virtual environment to manage dependencies. Run the following command:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use 'venv\Scripts\activate'
```

### 2. Set Up Environment
Ensure you have Python installed (preferably Python 3.8 or later). Install necessary dependencies by running:
```bash
pip install -r requirements.txt
```

### 3. Run Setup Script
Execute the setup script to initialize the environment:
```bash
python setup.py
```

### 4. Configure FAISS Index Path
Modify the `.env` file to set the FAISS index path:
```
FAISS_INDEX_PATH=/path/to/your/index
```

### 5. Run the Application
Start the application by running:
```bash
python app.py
```

## Contributing
Feel free to fork this repository and contribute enhancements via pull requests.

## License
This project is licensed under the **MIT License**.


# Offline Code Assistant (RAG Starter)

A project that was built using Google Gemini Chat. This project is a scalable, offline-first code assistant designed to answer questions about local codebases. It uses a Retrieval-Augmented Generation (RAG) pipeline to provide relevant, context-aware answers by combining semantic vector search with a local LLM — all running on your machine without requiring an internet connection.

This repository is built to be a robust starting point, focusing on performance, scalability, and maintainability.

## ✨ Features

* **Offline First:** All components (embedding model, LLM, vector store) run 100% locally.
* **Scalable RAG Pipeline:** Uses an `sqlite` database for efficient text chunk storage, avoiding high memory usage and allowing it to scale to millions of documents.
* **High-Performance Search:** Employs a `faiss-cpu` (HNSW) index for fast and accurate semantic search, even with large numbers of vectors.
* **Local LLM Inference:** Powered by `llama-cpp-python` to run GGUF models (like Mistral, Llama, etc.) on-device (CPU or GPU).
* **OpenAI-Compatible API:** The `FastAPI` backend provides `/v1/query` and `/v1/query/stream` endpoints, mimicking the OpenAI API for easy integration with existing tools.
* **Parallelized Indexing:** Uses `multiprocessing` to quickly scan, filter, and chunk large codebases, making the indexing process significantly faster.
* **Smart Chunking & Filtering:** Uses a code-aware text splitter and filters out common "junk" directories (like `node_modules`, `.git`, etc.) for a clean, relevant index.
* **Live Index Reloading:** Includes a `POST /v1/admin/reload-index` endpoint to load a new index without restarting the server.
* **Svelte Frontend:** Includes a simple, pre-configured web interface for chatting.

## ⚙️ Architecture Overview

The system is split into two main processes: **Indexing** and **Querying**.

1.  **Indexing (`indexer.py`)**
    * The script recursively scans your `CODEBASE_PATH`, obeying exclusion rules from `config.py`.
    * Files are processed in parallel: read, normalized, and split into intelligent chunks using `langchain_text_splitters`.
    * Each chunk is passed through the `sentence-transformers/all-mpnet-base-v2` embedding model.
    * The text content of the chunks is saved to `data/docstore.db` (SQLite).
    * The vector embeddings are saved to `data/vector_index.faiss` (a Faiss HNSW index).
    * An ID map (`data/id_map.json`) is saved to link the Faiss index to the SQLite IDs.

2.  **Querying (`server.py`)**
    * The FastAPI server loads the Faiss index, the ID map, and the LLM into memory on startup. It **does not** load the text chunks, saving RAM.
    * When you send a query to `/v1/query`:
        1.  Your question is embedded using the same `sentence-transformers` model.
        2.  The Faiss HNSW index performs a fast semantic search and returns the `top_k` most relevant chunk IDs.
        3.  The server queries the `docstore.db` to retrieve the text for *only* those `top_k` chunks.
        4.  This context, along with your question, is formatted into a prompt and sent to the local `llama-cpp` model.
        5.  The LLM's response is streamed back to you.

## 🚀 Getting Started

### 1. Prerequisites

* Python 3.10+
* A C++ Compiler (required by `faiss-cpu` and `llama-cpp-python`)
    * **macOS:** `xcode-select --install`
    * **Linux (Ubuntu):** `sudo apt-get install build-essential`
    * **Windows:** Install "C++ build tools" from the Visual Studio Installer.
* [Node.js](https://nodejs.org/) (Optional, for the frontend)

### 2. Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/your-username/rag-starter.git](https://github.com/your-username/rag-starter.git)
    cd rag-starter
    ```

2.  **Create a virtual environment:**
    ```sh
    python -m venv .venv
    source .venv/bin/activate
    # On Windows: .venv\Scripts\activate
    ```

3.  **Install Python dependencies:**
    ```sh
    pip install "uvicorn[standard]" fastapi "pydantic-settings" sse-starlette llama-cpp-python faiss-cpu sentence-transformers langchain-text-splitters
    ```
    *Note: For Metal (Apple Silicon) GPU support, install `llama-cpp-python` with Metal flags. See the `llama-cpp-python` docs for details.*

### 3. Configuration & Model Setup

Before running, you must provide the models.

1.  **Download a GGUF LLM:**
    The app is configured by default to look for a model at `models/mistral-7b-instruct-v0.2.Q4_K_M.gguf`.

    * Create the `models` directory: `mkdir models`
    * Download your preferred GGUF model (e.g., [Mistral 7B Instruct v0.2 Q4_K_M](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/blob/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf)) and place it in that directory.

2.  **Pre-cache the Embedding Model (Optional):**
    The embedding model (`sentence-transformers/all-mpnet-base-v2`) will be downloaded automatically on the first run. To download it ahead of time, run this command:
    ```sh
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"
    ```

3.  **Review Configuration (Optional):**
    All settings are in `src/config.py`. You can edit this file or override any setting with environment variables.
    * `CODEBASE_PATH`: The path to the code you want to index (default: `sample_code/`).
    * `LLAMA_MODEL_PATH`: The path to your GGUF LLM file.

## 🖥️ Usage

### Step 1: Index Your Codebase

Place the code you want to query into the `sample_code/` directory, or change the `CODEBASE_PATH` in `src/config.py` to point to your code.

Run the indexer:
```sh
python -m src.indexer
```

This will scan the files, generate embeddings, and create the data/ directory containing your vector index and document database.

### Step 2: Run the Backend Server
Start the FastAPI server using Uvicorn:


```sh
uvicorn src.server:app --host 0.0.0.0 --port 8000
```

or

```sh
python -m src.server
```

The server will load the models and index. Once you see "Application startup complete," it's ready.

### Step 3: Query the API Directly
You can also use curl or any HTTP client to interact with the API.

#### Standard Query (Non-streaming)

```sh
curl -X POST http://localhost:8000/v1/query \
-H "Content-Type: application/json" \
-d '{
  "query": "What is the function of build_chat_messages?"
}'
```
**Example API Response**

```json
{
  "id": "chatcmpl-849d3fd258db",
  "object": "chat.completion",
  "created": 1762825986,
  "model": "local-llama-3-8b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The function `build_chat_messages` is used to build the 'messages' list for the Llama chat completion API"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 1058,
    "completion_tokens": 131,
    "total_tokens": 1189
  }
}
```

#### Streaming Query

```sh
curl -X POST http://localhost:8000/v1/query/stream \
-H "Content-Type: application/json" \
-d '{
  "query": "Show me the chunk_text function from utils.py"
}'
```
**Example streaming data response**

```
data: {"id": "chatcmpl-433393e7-0f8f-4af3-a72f-eebac75e01d9", "model": "/home/user/projects/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf", "created": 1762826551, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant"}, "logprobs": null, "finish_reason": null}]}

data: {"id": "chatcmpl-433393e7-0f8f-4af3-a72f-eebac75e01d9", "model": "/home/user/projects/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf", "created": 1762826551, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "Hello"}, "logprobs": null, "finish_reason": null}]}

data: {"id": "chatcmpl-433393e7-0f8f-4af3-a72f-eebac75e01d9", "model": "/home/user/projects/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf", "created": 1762826551, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "words"}, "logprobs": null, "finish_reason": null}]}

data: {"id": "chatcmpl-433393e7-0f8f-4af3-a72f-eebac75e01d9", "model": "/home/user/projects/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf", "created": 1762826551, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "logprobs": null, "finish_reason": "stop"}]}

data: [DONE]

```

📁 Project Structure
.
├── data/                 # Generated by the indexer
│   ├── docstore.db       # SQLite DB for text chunks
│   ├── id_map.json       # Maps Faiss IDs to DB IDs
│   └── vector_index.faiss  # Faiss HNSW index
├── frontend/             # Svelte frontend
├── models/               # Directory for your GGUF LLM
├── sample_code/          # Example codebase to index
└── src/
    ├── server.py         # FastAPI server, API endpoints
    ├── indexer.py        # Script to index the codebase
    ├── embedding_service.py # Handles embeddings, Faiss, and SQLite
    ├── config.py         # Pydantic settings
    ├── utils.py          # File/text utilities, chunking
    └── models.py         # Pydantic API models

⚖️ License
This project is open-sourced. Please feel free to adapt it to your needs.

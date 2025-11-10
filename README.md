# Offline Code Assistant

An offline semantic code assistant for locating functions and configurable settings inside a local **Objective-C / Ruby** codebase. It uses local models:

- **Embeddings:** `all-MiniLM-L6-v2` (sentence-transformers) — *for vector embeddings*
- **LLM:** `llama-3-8b-Instruct` (GGUF via `llama-cpp-python`) — *for answer generation*

> **Paths used in this project** (must be absolute on your machine):
>
> - Codebase path: `~/projects/codebase`
> - Embedding model dir: `~/projects/llm-models/all-MiniLM-L6-v2/`
> - Llama model file:
>   `~/projects/llm-models/backyardai_llama-3-8b-Instruct-GGUF_llama-3-8b-Instruct.Q4_K_M.gguf`
>
> **Note:** These paths can now be overridden by environment variables (e.g., `CODEBASE_PATH=...`)

---

## Features

- FastAPI backend with REST JSON endpoint and SSE streaming endpoint
- Local inference only — no network calls
- Sentence-Transformers + Faiss vector index for semantic retrieval
- Llama via `llama-cpp-python` for text generation
- Indexer for scanning `.m`, `.h`, `.rb`, `.xml` files
- Unit tests with mocks (pytest)
- Structured logging and detailed error messages

---

## Quick Start (Development)

### 1. Clone and Install

```bash
git clone <this-repo-url> offline-code-assistant
cd offline-code-assistant
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> If you use pip inside a conda env or system Python, install `faiss-cpu` (or your GPU build) as needed.

---

### 2. Prepare the Models

Place your models at the exact absolute paths listed above, or update `src/config.py` defaults, or set environment variables.

- Put the `all-MiniLM-L6-v2` directory under:
  `~/projects/llm-models/all-MiniLM-L6-v2/`

- Put the Llama GGUF file at:
  `~/projects/llm-models/backyardai_llama-3-8b-Instruct-GGUF_llama-3-8b-Instruct.Q4_K_M.gguf`

The app will validate the presence of models at startup and fail gracefully with helpful logs if missing.

---

#### Compiling llama-cpp with Intel GPU (SYCL) support

If you wish to compile `llama-cpp` with Intel GPU acceleration, use these flags:

```bash
-DGGML_SYCL=on -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
```

Example build steps:

```bash
# clone
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# create build dir
mkdir build && cd build

# configure: ensure icx/icpx (Intel compilers) are in PATH
cmake .. -DGGML_SYCL=on -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx

# build
cmake --build . -j
```

After building, follow instructions to convert the model to GGUF and copy it to your models directory.
**Important:** Building with Intel compilers requires Intel oneAPI or appropriate toolchain installed.
If you cannot compile for SYCL, `llama-cpp-python` will still work on CPU.

---

### 3. Indexing the Codebase

Before querying, index your codebase into vectors:

```bash
python -m src.indexer
# or import and call index_codebase from a Python shell
```

The indexer:

- Scans `~/projects/codebase` (or `CODEBASE_PATH`) for `.m`, `.h`, `.rb`, `.xml`
- Chunks each file (512 tokens/words default, 64 overlap)
- Embeds chunks with MiniLM
- Stores a Faiss index at `data/vector_index.faiss`
- Saves metadata at `data/docs_chunks.json`

If indexing fails because models are missing, the indexer will raise a clear error.

---

### 4. Running the FastAPI Server

Start with uvicorn:

```bash
uvicorn src.server:app --host 0.0.0.0 --port 8000 --log-level info
```

On startup, the server will attempt to load embeddings and the prebuilt index (if present).
If the index is not present yet, you will get a **503** from `/v1/query` until you run the indexer.

---

## API

### POST /v1/query

**JSON body:**

```json
{
  "query": "Where is the API token configured?",
  "top_k": 5
}
```

**Response (example):**

```json
{
  "id": "chatcmpl-12345",
  "object": "chat.completion",
  "created": 1731052452,
  "model": "local-llama-3-8b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The configuration for API tokens is located in config/initializer/api_config.rb"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": { ... },
  "conversation_id": null,
  "context": {
    "retrieved": [
      "/home/user/projects/codebase/config/initializer/api_config.rb"
    ]
  }
}
```

---

### POST /v1/query/stream

Server-Sent Events (SSE) endpoint that streams tokens as they are produced.
Useful for UI that wants progressive results.

**JSON body:**

```json
{
  "query": "Where is the API token configured?",
  "top_k": 3
}
```

Example using curl:

```bash
curl -N -X POST "http://localhost:8000/v1/query/stream"   -H "Content-Type: application/json"   -d '{"query": "Where is the API token configured?", "top_k": 3}'
```

You will receive SSE events of token messages and a final `[DONE]` message.

---

## Tests

Run tests with:

```bash
pytest -q
```

The tests mock heavy models so they run quickly on CI / developer machines.

---

## Example Usage (Python Client)

```python
import requests
resp = requests.post("http://localhost:8000/v1/query", json={"query":"Find function that initializes Bluetooth","top_k":5})
print(resp.json())
```

**SSE streaming example (Python):**

```python
import sseclient
import requests
import json

payload = {"query": "show the processValue function", "top_k": 1}
r = requests.post("http://localhost:8000/v1/query/stream", json=payload, stream=True)

client = sseclient.SSEClient(r)
for event in client.events():
    if event.data == "[DONE]":
        break
    try:
        chunk = json.loads(event.data)
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        if "content" in delta:
            print(delta["content"], end="", flush=True)
    except json.JSONDecodeError:
        pass
```

---

## Implementation Notes and Limitations

- Uses Faiss (flat L2 index) and `all-MiniLM-L6-v2` embeddings.
  For production or very large corpora, use an IVF/HNSW Faiss index and persistent storage.
- Streaming depends on the `llama-cpp-python` streaming API.
- Codebase path and model paths are absolute and validated. Confirm exact paths on your host.

---

## Troubleshooting

- **Model not found:** Check environment variables or paths in `src/config.py`.
- **Index not found:** Run `python -m src.indexer`.
- **Llama streaming not working:** Check your `llama-cpp-python` version.

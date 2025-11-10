import time
import logging
import json
import uuid
import sqlite3
from typing import List, Optional, AsyncGenerator, Dict, Any, Tuple
import asyncio
import uvicorn

from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from llama_cpp import Llama
import faiss # Import faiss to use its types in hints

from .models import QueryRequest
from .embedding_service import EmbeddingService
from .config import settings
from .utils import normalize_whitespace

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("oca.server")

app = FastAPI(title="Offline Code Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["Content-Type", "Authorization"],
)

class AppState:
    """Holds the loaded models and index."""
    def __init__(self):
        self.embedding: Optional[EmbeddingService] = None
        self.index: Optional[faiss.Index] = None  # faiss.Index
        self.id_map: Optional[List[str]] = None # Maps Faiss index to chunk ID
        self.llama: Optional[Llama] = None
        # Lock to protect index/id_map during reloads
        self.reload_lock = asyncio.Lock()

app.state = AppState()

def build_chat_messages(query: str, contexts: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Builds the 'messages' list for the Llama chat completion API.
    """
    context_text = "\n\n----\n".join(
        [f"File: {c['path']}\nSnippet:\n{c['chunk']}" for c in contexts]
    )

    system_prompt = (
        "You are a helpful offline code assistant. "
        "Use the file snippets below to answer the user's question. "
        "Answer concisely, referencing file paths and function names."
    )

    user_prompt = (
        f"The following snippets come from the user's local codebase:\n\n"
        f"{context_text}\n\n"
        f"User question:\n{query}\n\n"
        "If the user asks to see code, show the exact code from the snippets above without rewording. "
        "If the function or code block is not found, respond with "
        "'I could not find that code in the provided context.'"
    )

    if len(user_prompt) > settings.LLAMA_MAX_PROMPT_CHARS:
        logger.warning("User prompt too long (%d chars). Truncating to %d chars.", len(user_prompt), settings.LLAMA_MAX_PROMPT_CHARS)
        user_prompt = user_prompt[:settings.LLAMA_MAX_PROMPT_CHARS] + "\n\n[Truncated for context length]\n"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    return messages


def _load_index_data() -> Tuple[Optional[faiss.Index], Optional[List[str]]]:
    """
    Tries to load the Faiss index and ID map using the
    embedding service on app.state.
    """
    if app.state.embedding is None:
        logger.error("Cannot load index, embedding service not available.")
        return None, None

    try:
        index, id_map = app.state.embedding.load_index()
        logger.info("Vector index and ID map re-loaded successfully.")
        return index, id_map
    except FileNotFoundError:
        logger.warning("Vector index not found. Please run indexing before using the API.")
        return None, None
    except Exception as e:
        logger.exception("Failed to load vector index: %s", e)
        return None, None


@app.on_event("startup")
async def startup_event():
    """Load embeddings, vector index, and Llama model at startup."""
    logger.info("Starting Offline Code Assistant server...")

    try:
        app.state.embedding = EmbeddingService(model_path=str(settings.EMBEDDING_MODEL_PATH))
        app.state.embedding.load()
    except Exception as e:
        logger.exception("FATAL: Failed to load embedding model at startup: %s", e)
        app.state.embedding = None

    if app.state.embedding:
        app.state.index, app.state.id_map = _load_index_data()

    if not settings.LLAMA_MODEL_PATH or not settings.LLAMA_MODEL_PATH.exists():
        logger.error("FATAL: Llama model path not found: %s", settings.LLAMA_MODEL_PATH)
        logger.error("Please set LLAMA_MODEL_PATH environment variable or update src/config.py")
        app.state.llama = None
    else:
        try:
            logger.info("Loading Llama model: %s", settings.LLAMA_MODEL_PATH)
            app.state.llama = Llama(
                model_path=str(settings.LLAMA_MODEL_PATH),
                n_ctx=settings.LLAMA_N_CTX,
                n_threads=settings.LLAMA_THREADS,
                n_gpu_layers=settings.LLM_N_GPU_LAYERS,
                verbose=settings.LLM_VERBOSE,
            )
            logger.info("Llama model loaded successfully.")
        except Exception as e:
            logger.exception("FATAL: Failed to initialize Llama: %s", e)
            app.state.llama = None

async def check_models_or_raise():
    """Check if all required models are loaded, or raise HTTPException."""
    # Acquire lock to ensure we read a consistent state
    async with app.state.reload_lock:
        if app.state.embedding is None:
            raise HTTPException(status_code=500, detail="Embedding model not loaded on server.")
        if app.state.llama is None:
            raise HTTPException(status_code=500, detail="Llama model not loaded on server.")
        if app.state.index is None or app.state.id_map is None:
            raise HTTPException(status_code=503, detail="Vector index not available. Run indexing first.")


async def retrieve_contexts(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve top-k code chunks for a given query.
    1. Search Faiss for top-k chunk IDs (under lock).
    2. Fetch chunk text from SQLite using IDs.
    """
    context_ids = []

    # Acquire lock to ensure index is not reloaded during retrieval
    async with app.state.reload_lock:
        if app.state.index is None or app.state.id_map is None or app.state.embedding is None:
            logger.error("Retrieval attempted but index or embedding model is not loaded.")
            raise HTTPException(status_code=503, detail="Vector index not available. Run indexing first.")

        # 1. Embed query and search Faiss for chunk IDs
        qvec = app.state.embedding.embed([normalize_whitespace(query)])[0]
        # Search now returns [{"score": float, "id": str}]
        results = app.state.embedding.search(app.state.index, app.state.id_map, qvec.reshape(1, -1), top_k=top_k)

        context_ids = [r["id"] for r in results]

    # --- Lock is released here ---

    if not context_ids:
        return []

    # 2. Fetch chunk text from SQLite (occurs outside the lock)
    contexts = []
    conn = None
    try:
        conn = sqlite3.connect(f"file:{settings.DB_PATH}?mode=ro", uri=True) # Read-only connection
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        placeholders = ",".join("?" * len(context_ids))
        query_sql = f"SELECT id, path, chunk FROM chunks WHERE id IN ({placeholders})"

        cursor.execute(query_sql, context_ids)

        # Map IDs to rows to preserve relevance order
        db_results = {row["id"]: {"path": row["path"], "chunk": row["chunk"]} for row in cursor.fetchall()}

        # Re-build contexts list in the correct order
        for r_id in context_ids:
            if r_id in db_results:
                contexts.append(db_results[r_id])

    except Exception as e:
        logger.exception("Failed to retrieve contexts from SQLite: %s", e)
        raise HTTPException(status_code=500, detail="Failed to retrieve context from database.")
    finally:
        if conn:
            conn.close()

    return contexts

# ---------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------

@app.post("/v1/query")
async def query_endpoint(req: QueryRequest):
    """
    Standard non-streaming query endpoint.
    Returns an OpenAPI-compliant ChatCompletionResponse.
    """
    try:
        await check_models_or_raise()
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": {"message": e.detail}})

    if not req.query:
        raise HTTPException(status_code=400, detail="Query text required.")

    contexts = await retrieve_contexts(req.query, top_k=req.top_k or 5)

    messages = build_chat_messages(req.query, contexts)

    try:
        start_time = time.time()

        resp = app.state.llama.create_chat_completion(
            messages=messages,
            **settings.LLAMA_DECODE_OPTIONS,
        )

        text = resp["choices"][0]["message"]["content"].strip()

        usage = resp.get("usage", {
            "prompt_tokens": 0,
            "completion_tokens": len(text.split()),
            "total_tokens": len(text.split())
        })

    except Exception as e:
        logger.exception("Llama generation error: %s", e)
        return JSONResponse(status_code=500, content={"error": {"message": "Model generation failed"}})

    # Build and return OpenAPI-compliant response
    response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    model_name = "local-llama-3-8b"
    created_time = int(start_time)

    openapi_response = {
        "id": response_id,
        "object": "chat.completion",
        "created": created_time,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text
                },
                "finish_reason": "stop"
            }
        ],
        "usage": usage
    }

    return JSONResponse(status_code=200, content=openapi_response)


@app.post("/v1/query/stream")
async def stream_query(req: QueryRequest):
    """
    OpenAPI-compliant streaming endpoint (POST + SSE).
    Streams partial responses as 'data: ...' chunks.
    """
    try:
        await check_models_or_raise()
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": {"message": e.detail}})

    if not req.query:
        raise HTTPException(status_code=400, detail="Missing 'query' in request body.")

    contexts = await retrieve_contexts(req.query, top_k=req.top_k or 5)

    messages = build_chat_messages(req.query, contexts)

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def generate_in_thread():
        """
        Run blocking Llama streaming generation in a background thread
        and push tokens into the asyncio queue.
        """
        try:
            for chunk in app.state.llama.create_chat_completion(
                messages=messages,
                stream=True,
                **settings.LLAMA_DECODE_OPTIONS,
            ):
                queue.put_nowait(chunk)

        except Exception as e:
            logger.exception("Streaming generation failed: %s", e)
            queue.put_nowait(f"[ERROR_GENERATION]{e}")
        finally:
            queue.put_nowait("[GEN_DONE]")

    loop.run_in_executor(None, generate_in_thread)

    async def sse_generator() -> AsyncGenerator[str, None]:
        """
        Yields OpenAPI-compliant SSE messages.
        `llama-cpp-python` > 0.2.20+ already produces
        OpenAI-compliant chunks, so we just proxy them.

        This generator now yields *only the data string*,
        letting EventSourceResponse handle formatting.
        """

        first_chunk = True
        try:
            while True:
                item = await queue.get()

                if item == "[GEN_DONE]":
                    yield "[DONE]"
                    break

                if isinstance(item, str) and item.startswith("[ERROR_GENERATION]"):
                    err_msg = item[len("[ERROR_GENERATION]"):]
                    error_chunk = {"error": {"message": err_msg}}
                    yield json.dumps(error_chunk)
                    yield "[DONE]"

                    break

                chunk = item

                if first_chunk:
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    if "role" not in delta:
                        delta["role"] = "assistant"
                    first_chunk = False

                yield json.dumps(chunk)

        except asyncio.CancelledError:
            logger.warning("SSE stream client disconnected.")
        except Exception as e:
            logger.error(f"SSE stream error: {e}")
        finally:
            logger.info("Streaming completed for query: %s", req.query[:50])

    # Set standard SSE headers
    headers = {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }

    return EventSourceResponse(sse_generator(), headers=headers)


@app.post("/v1/admin/reload-index")
async def reload_index_endpoint():
    """
    Admin endpoint to manually reload the vector index and ID map
    from disk. This allows for updating the index without
    restarting the server.
    """
    logger.info("Received request to reload vector index...")

    # Use the helper to load new data
    new_index, new_id_map = _load_index_data()

    if new_index is None or new_id_map is None:
        logger.error("Index reload failed. Keeping old index.")
        raise HTTPException(
            status_code=500,
            detail="Index reload failed. Check server logs."
        )

    # Acquire lock to safely swap the state
    async with app.state.reload_lock:
        app.state.index = new_index
        app.state.id_map = new_id_map

    logger.info("Index reload successful. New index is live.")
    return JSONResponse(
        status_code=200,
        content={"message": "Index reloaded successfully."}
    )

if __name__ == "__main__":
    """
    Allows running the server directly with `python -m src.server`
    """
    logger.info("Starting Uvicorn server...")
    uvicorn.run(
        "src.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False # Set to True for development, but False is safer for this app
    )

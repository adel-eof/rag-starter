import time
import logging
import json
import uuid
from typing import List, Optional, AsyncGenerator, Dict, Any
import asyncio

from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse  # Correct import for SSE
from llama_cpp import Llama

from .models import QueryRequest
from .embedding_service import EmbeddingService
from .config import settings  # Import the settings instance
from .utils import normalize_whitespace

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("oca.server")

app = FastAPI(title="Offline Code Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# ---------------------------------------------------------------------
# Application State
# ---------------------------------------------------------------------

class AppState:
    """Holds the loaded models and index."""
    def __init__(self):
        self.embedding: Optional[EmbeddingService] = None
        self.index: Optional[Any] = None  # faiss.Index
        self.docs: Optional[List[Dict[str, Any]]] = None
        self.llama: Optional[Llama] = None

app.state = AppState()

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def build_chat_messages(query: str, contexts: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Builds the 'messages' list for the Llama chat completion API.
    """
    context_text = "\n\n----\n".join(
        [f"File: {c['path']}\nSnippet:\n{c['chunk']}" for c in contexts]
    )

    # THE ORIGINAL
    # system_prompt = (
    #     "You are a helpful offline code assistant. "
    #     "Use the file snippets below to answer the user's question. "
    #     "Answer concisely, referencing file paths and function names."
    # )

    system_prompt = (
        "You are CodeMate, an offline code assistant that works strictly with the user's local codebase. "
        "The user query is paired with code snippets retrieved from a FAISS index and their metadata. "
        "Only use the content of these snippets to answer questions. "
        "Do not speculate about files or functions not shown in the snippets. "
        "If the user asks for a specific function, class, or code block, you MUST provide that code block *exactly* as it appears in the context."
        "If a requested symbol or function (e.g., `find_source_files`) is not present in the retrieved snippets, "
        "reply exactly: 'I could not find that code in the provided context.' "
        "When code is present, reproduce it verbatim, citing the file path it came from. "
        "Do not summarize or reformat code — show it exactly as it appears. "
        "Be concise, accurate, and avoid any assumptions about unseen parts of the codebase."
        "If the user asks a general question, provide a summary."
    )

    # THE ORIGINAL
    # user_prompt = (
    #     f"Context:\n{context_text}\n\n"
    #     f"User question: {query}\n\n"
    #     "Answer:"
    # )

    user_prompt = (
        f"The following snippets come from the user's local codebase:\n\n"
        f"{context_text}\n\n"
        f"User question:\n{query}\n\n"
        "If the user asks to see code, show the exact code from the snippets above without rewording. "
        "If the function or code block is not found, respond with "
        "'I could not find that code in the provided context.'"
    )

    # Truncate prompt if it's too large for model’s context window
    if len(user_prompt) > settings.LLAMA_MAX_PROMPT_CHARS:
        logger.warning("User prompt too long (%d chars). Truncating to %d chars.", len(user_prompt), settings.LLAMA_MAX_PROMPT_CHARS)
        user_prompt = user_prompt[:settings.LLAMA_MAX_PROMPT_CHARS] + "\n\n[Truncated for context length]\n"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    return messages


# ---------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """Load embeddings, vector index, and Llama model at startup."""
    logger.info("Starting Offline Code Assistant server...")

    # 1. Load Embedding Service
    try:
        app.state.embedding = EmbeddingService(model_path=str(settings.EMBEDDING_MODEL_PATH))
        app.state.embedding.load()
    except Exception as e:
        logger.exception("FATAL: Failed to load embedding model at startup: %s", e)
        app.state.embedding = None # Ensure it's None on failure

    # 2. Load Vector Index
    if app.state.embedding:
        try:
            index, docs = app.state.embedding.load_index(
                index_path=settings.VECTOR_INDEX_PATH,
                docstore_path=settings.DOCSTORE_PATH
            )
            app.state.index = index
            app.state.docs = docs
            logger.info("Vector index and docs loaded successfully.")
        except FileNotFoundError:
            logger.warning("Vector index not found at startup. Please run indexing before using the API.")
            app.state.index = None
            app.state.docs = None
        except Exception as e:
            logger.exception("Failed to load vector index: %s", e)
            app.state.index = None
            app.state.docs = None

    # 3. Load Llama Model
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

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def check_models_or_raise():
    """Check if all required models are loaded, or raise HTTPException."""
    if app.state.embedding is None:
        raise HTTPException(status_code=500, detail="Embedding model not loaded on server.")
    if app.state.llama is None:
        raise HTTPException(status_code=500, detail="Llama model not loaded on server.")
    if app.state.index is None or app.state.docs is None:
        raise HTTPException(status_code=503, detail="Vector index not available. Run indexing first.")


def retrieve_contexts(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Retrieve top-k code chunks for a given query."""
    if app.state.index is None or app.state.docs is None or app.state.embedding is None:
        logger.error("Retrieval attempted but index or embedding model is not loaded.")
        raise HTTPException(status_code=503, detail="Vector index not available. Run indexing first.")

    qvec = app.state.embedding.embed([normalize_whitespace(query)])[0]
    results = app.state.embedding.search(app.state.index, app.state.docs, qvec.reshape(1, -1), top_k=top_k)
    contexts = [r["doc"] for r in results]
    return contexts

# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------

@app.post("/v1/query")
async def query_endpoint(req: QueryRequest):
    """
    Standard non-streaming query endpoint.
    Returns an OpenAPI-compliant ChatCompletionResponse.
    """
    try:
        check_models_or_raise()
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": {"message": e.detail}})

    if not req.query:
        raise HTTPException(status_code=400, detail="Query text required.")

    # 1. Retrieve Context
    contexts = retrieve_contexts(req.query, top_k=req.top_k or 5)

    # 2. Build Messages
    messages = build_chat_messages(req.query, contexts)

    # 3. Run Llama Completion
    try:
        start_time = time.time()

        # We use create_chat_completion for Instruct models
        resp = app.state.llama.create_chat_completion(
            messages=messages,
            **settings.LLAMA_DECODE_OPTIONS,
        )

        text = resp["choices"][0]["message"]["content"].strip()

        # Extract token usage from llama-cpp response
        usage = resp.get("usage", {
            "prompt_tokens": 0,
            "completion_tokens": len(text.split()),
            "total_tokens": len(text.split())
        })

    except Exception as e:
        logger.exception("Llama generation error: %s", e)
        return JSONResponse(status_code=500, content={"error": {"message": "Model generation failed"}})

    # 4. Build and return OpenAPI-compliant response
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
        check_models_or_raise()
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": {"message": e.detail}})

    if not req.query:
        raise HTTPException(status_code=400, detail="Missing 'query' in request body.")

    # 1. Retrieve Context
    contexts = retrieve_contexts(req.query, top_k=req.top_k or 5)

    # 2. Build Messages
    messages = build_chat_messages(req.query, contexts)

    # 3. Get asyncio loop and create queue
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def generate_in_thread():
        """
        Run blocking Llama streaming generation in a background thread
        and push tokens into the asyncio queue.
        """
        try:
            # create_chat_completion with stream=True returns a generator
            for chunk in app.state.llama.create_chat_completion(
                messages=messages,
                stream=True,
                **settings.LLAMA_DECODE_OPTIONS,
            ):
                # Put the entire compliant chunk from llama-cpp into the queue
                queue.put_nowait(chunk)

        except Exception as e:
            logger.exception("Streaming generation failed: %s", e)
            queue.put_nowait(f"[ERROR_GENERATION]{e}")
        finally:
            # Signal that generation is done
            queue.put_nowait("[GEN_DONE]")

    # Kick off background generation in a thread
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

                # --- THIS IS THE FIX ---
                # Yield only the string "[DONE]"
                # EventSourceResponse will format it to "data: [DONE]\n\n"
                if item == "[GEN_DONE]":
                    yield "[DONE]"
                    break
                # --- END FIX ---

                if isinstance(item, str) and item.startswith("[ERROR_GENERATION]"):
                    err_msg = item[len("[ERROR_GENERATION]"):]
                    error_chunk = {"error": {"message": err_msg}}
                    # --- THIS IS THE FIX ---
                    # Yield the JSON string of the error
                    yield json.dumps(error_chunk)
                    yield "[DONE]"
                    # --- END FIX ---
                    break

                # 'item' is the chunk dictionary from llama-cpp
                chunk = item

                # The first chunk from llama-cpp might not have the 'role'
                # Let's manually inject it if it's the first chunk
                if first_chunk:
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    if "role" not in delta:
                        delta["role"] = "assistant"
                    first_chunk = False

                # --- THIS IS THE FIX ---
                # Yield *only* the JSON string.
                # EventSourceResponse will format it to "data: {...}\n\n"
                yield json.dumps(chunk)
                # --- END FIX ---

        except asyncio.CancelledError:
            # This catches client disconnects
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
        "X-Accel-Buffering": "no",  # disables proxy buffering
    }

    # EventSourceResponse correctly handles formatting strings yielded by sse_generator
    return EventSourceResponse(sse_generator(), headers=headers)

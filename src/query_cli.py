import time
import argparse
import logging
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

from .embedding_service import EmbeddingService
from .utils import normalize_whitespace
from .config import settings

# This check is necessary for llama_cpp.Llama
# It must be imported *after* numpy.
try:
    from llama_cpp import Llama
except ImportError:
    print("Error: llama-cpp-python is not installed.", file=sys.stderr)
    print("Please install it with: pip install llama-cpp-python", file=sys.stderr)
    sys.exit(1)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("oca.query_cli")


def build_chat_messages(query: str, contexts: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Builds the 'messages' list for the Llama chat completion API.
    """
    context_text = "\n\n----\n".join([f"File: {c['path']}\nSnippet:\n{c['chunk']}" for c in contexts])

    system_prompt = (
        "You are an offline code assistant. "
        "Use the following file snippets to answer the user's question."
    )
    user_prompt = (
        f"Context:\n{context_text}\n\n"
        f"User question: {query}\n\n"
        "Answer with file paths and a short explanation."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    return messages


def run_query(query: str, top_k: int = 5):
    """
    Runs a full RAG query from the command line.
    """
    try:
        # 1. Load resources
        emb = EmbeddingService()
        emb.load()
        index, docs = emb.load_index()
        logger.info("Loaded embedding model and index.")
    except FileNotFoundError as e:
        logger.error("Failed to load index. Have you run the indexer? Error: %s", e)
        print("\n[ERROR] Index files not found. Please run `python -m src.indexer` first.", file=sys.stderr)
        return
    except Exception as e:
        logger.error("Failed to load resources: %s", e)
        print(f"\n[ERROR] Failed to load resources: {e}", file=sys.stderr)
        return

    # 2. Embed query and retrieve context
    logger.info("Embedding query...")
    query_vec = emb.embed([normalize_whitespace(query)])[0]
    results = emb.search(index, docs, query_vec.reshape(1, -1), top_k=top_k)
    contexts = [r["doc"] for r in results]
    logger.info("Retrieved %d contexts.", len(contexts))

    if not contexts:
        logger.warning("No contexts found for query.")
        print("\n[WARN] No relevant code snippets found for that query.", file=sys.stderr)
        return

    # 3. Build prompt (messages)
    messages = build_chat_messages(query, contexts)

    # 4. Run Llama locally
    try:
        logger.info("Loading Llama model: %s", settings.LLAMA_MODEL_PATH)
        llama = Llama(
            model_path=str(settings.LLAMA_MODEL_PATH),
            n_ctx=settings.LLAMA_N_CTX,
            n_threads=settings.LLAMA_THREADS,
            n_gpu_layers=settings.LLM_N_GPU_LAYERS,
            verbose=settings.LLM_VERBOSE,
        )
    except Exception as e:
        logger.error("Failed to load Llama model: %s", e)
        print(f"\n[ERROR] Failed to load Llama model at {settings.LLAMA_MODEL_PATH}. {e}", file=sys.stderr)
        return

    logger.info("Running Llama inference...")
    start_time = time.time()

    resp = llama.create_chat_completion(
        messages=messages,
        **settings.LLAMA_DECODE_OPTIONS
    )

    end_time = time.time()
    logger.info("Inference complete in %.2f seconds.", end_time - start_time)

    text = resp.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

    # 5. Format and print output
    out = {
        "id": f"local-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "local-llama-3-8b",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "context": {
            "retrieved_files": [c["path"] for c in contexts]
        }
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the local codebase using RAG.")
    parser.add_argument("query", type=str, help="The query to ask the code assistant.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of context snippets to retrieve.")
    args = parser.parse_args()

    if not Path(settings.LLAMA_MODEL_PATH).exists():
        print(f"[ERROR] Llama model not found at: {settings.LLAMA_MODEL_PATH}", file=sys.stderr)
        print("Please check the LLAMA_MODEL_PATH in your environment or src/config.py", file=sys.stderr)
        sys.exit(1)

    run_query(args.query, args.top_k)

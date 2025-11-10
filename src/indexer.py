import logging
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any

from .config import settings
from .embedding_service import EmbeddingService
from .utils import find_source_files, read_file_text, chunk_text, normalize_whitespace

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("oca.indexer")


def index_codebase(codebase_path: Path, embedding_model_path: str = None) -> bool:
    """
    Scans, chunks, embeds, and indexes the entire codebase.

    Args:
        codebase_path: The root directory of the code to index.
        embedding_model_path: Optional path to an embedding model, overriding config.

    Returns:
        True if indexing was successful, False otherwise.
    """
    logger.info("Starting indexing of codebase: %s", codebase_path)

    # 1. Find all source files (now filtered)
    files = find_source_files(codebase_path, settings.INDEX_EXTENSIONS)
    if not files:
        logger.warning("No source files found with extensions %s in %s", settings.INDEX_EXTENSIONS, codebase_path)
        return False

    docs: List[Dict[str, Any]] = []
    counter = 0

    # 2. Read, normalize, and chunk files (now with better chunker)
    for f in files:
        text = read_file_text(f)
        if not text:
            logger.warning("Skipping empty or unreadable file: %s", f)
            continue

        # Normalize whitespace to keep embedding input stable
        text = normalize_whitespace(text)

        chunks = chunk_text(text, chunk_size=settings.CHUNK_SIZE, overlap=settings.CHUNK_OVERLAP)

        for i, chunk in enumerate(chunks):
            docs.append({
                "id": f"{f.resolve()}::{i}",
                "path": str(f.resolve()),
                "chunk": chunk,
            })
            counter += 1

    if not docs:
        logger.warning("No text chunks were generated from the source files.")
        return False

    logger.info("Prepared %d chunks for embedding", counter)

    # 3. Initialize embedding service and build index
    # Use override model path if provided, otherwise default (from config)
    model_path_to_use = embedding_model_path or str(settings.EMBEDDING_MODEL_PATH)

    emb = EmbeddingService(model_path=model_path_to_use)
    emb.load()

    # build_index now handles all storage (Faiss, SQLite, ID map)
    emb.build_index(docs)

    logger.info("Indexing complete. Indexed %d chunks.", counter)
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index the local codebase into a vector store.")
    parser.add_argument(
        "--path",
        type=str,
        default=str(settings.CODEBASE_PATH),
        help=f"Path to the codebase root. Default: {settings.CODEBASE_PATH}"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Path to the embedding model (overrides config). Default: {settings.EMBEDDING_MODEL_PATH}"
    )
    args = parser.parse_args()

    try:
        ok = index_codebase(Path(args.path), embedding_model_path=args.model)
        if ok:
            print(f"\n[OK] Indexing complete for {args.path}")
        else:
            print(f"\n[WARN] Indexing returned no docs for {args.path}")
    except Exception as e:
        print(f"\n[ERROR] Indexing failed: {e}", file=sys.stderr)
        logger.exception("Indexing script failed")
        sys.exit(1)

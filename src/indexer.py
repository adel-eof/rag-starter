import logging
import sys
import argparse
import multiprocessing
from pathlib import Path
from typing import List, Dict, Any

from .config import settings
from .embedding_service import EmbeddingService
from .utils import find_source_files, read_file_text, chunk_text, normalize_whitespace

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("oca.indexer")


def _process_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Worker function for multiprocessing.
    Reads, normalizes, and chunks a single file.
    Returns a list of doc dicts for that file.
    """
    try:
        text = read_file_text(file_path)
        if not text:
            logger.warning("Skipping empty or unreadable file: %s", file_path)
            return []

        text = normalize_whitespace(text)
        chunks = chunk_text(text, chunk_size=settings.CHUNK_SIZE, overlap=settings.CHUNK_OVERLAP)

        file_docs = []
        for i, chunk in enumerate(chunks):
            file_docs.append({
                "id": f"{file_path.resolve()}::{i}",
                "path": str(file_path.resolve()),
                "chunk": chunk,
            })
        return file_docs
    except Exception as e:
        logger.error(f"Failed to process file {file_path}: {e}")
        return []

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

    # 1. Find all source files
    files = find_source_files(codebase_path, settings.INDEX_EXTENSIONS)
    if not files:
        logger.warning("No source files found with extensions %s in %s", settings.INDEX_EXTENSIONS, codebase_path)
        return False

    # 2. Read, normalize, and chunk files in parallel
    logger.info("Processing %d files in parallel...", len(files))
    num_cores = multiprocessing.cpu_count()
    logger.info("Using %d CPU cores...", num_cores)

    docs: List[Dict[str, Any]] = []
    with multiprocessing.Pool(processes=num_cores) as pool:
        # map applies _process_file to each item in files
        # results is a list of lists (List[List[Dict]])
        results = pool.map(_process_file, files)

        # Flatten the list of lists
        docs = [doc for file_docs in results for doc in file_docs]

    counter = len(docs)

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
        help=f"Path or name of the embedding model (overrides config). Default: {settings.EMBEDDING_MODEL_PATH}"
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

import logging
from pathlib import Path
from typing import List, Set
import re

logger = logging.getLogger("oca.utils")


def find_source_files(root: Path, exts: Set[str]) -> List[Path]:
    """
    Recursively locate files with the given extensions.

    Args:
        root: The root directory to search.
        exts: A set of file extensions to find (e.g., {".m", ".h"}).
    """
    if not root.exists():
        logger.error("Codebase path does not exist: %s", root)
        return []
    if not root.is_dir():
        logger.error("Codebase path is not a directory: %s", root)
        return []

    files = []
    for p in root.rglob("*"):
        if p.suffix.lower() in exts and p.is_file():
            files.append(p)

    logger.info("Found %d source files under %s", len(files), root)
    return files


def read_file_text(path: Path) -> str:
    """
    Read text from a file, ignoring decoding errors.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return text
    except Exception as e:
        logger.exception("Failed to read %s: %s", path, e)
        return ""


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """
    Naive chunker: splits on whitespace windows to generate chunks with overlap.

    Returns:
        A list of text chunks.
    """
    words = text.split()
    if not words:
        return []

    if chunk_size <= overlap:
        logger.warning("Chunk size (%d) is <= overlap (%d). Setting overlap to 0.", chunk_size, overlap)
        overlap = 0

    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))

        if i + chunk_size >= len(words):
            break

        i += chunk_size - overlap

    return chunks


def normalize_whitespace(s: str) -> str:
    """Replace all whitespace sequences with a single space."""
    return re.sub(r"\s+", " ", s).strip()

import logging
from pathlib import Path
from typing import List, Set
import re

# This is a new dependency for the improved chunker
# from langchain_community.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings

logger = logging.getLogger("oca.utils")


def find_source_files(root: Path, exts: Set[str]) -> List[Path]:
    """
    Recursively locate files with the given extensions,
    ignoring specified directories.

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
        # Check if the path is within an excluded directory
        if any(part in settings.INDEX_EXCLUDE_DIRS for part in p.parts):
            continue

        if p.suffix.lower() in exts and p.is_file():
            files.append(p)

    logger.info("Found %d source files under %s (after filtering)", len(files), root)
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
    Splits text using a recursive character splitter.
    Note: chunk_size and overlap are now character-based.

    Returns:
        A list of text chunks.
    """
    if not text:
        return []

    # Initialize the splitter
    # This is much more effective than the naive word splitter
    text_splitter = RecursiveCharacterTextSplitter(
        # TODO: Add code-specific separators if desired
        # separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        is_separator_regex=False,
    )

    return text_splitter.split_text(text)


def normalize_whitespace(s: str) -> str:
    """Replace all whitespace sequences with a single space."""
    return re.sub(r"\s+", " ", s).strip()

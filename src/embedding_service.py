from pathlib import Path
import json
import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from sentence_transformers import SentenceTransformer
import faiss

from .config import settings
from .utils import chunk_text

logger = logging.getLogger("oca.embedding")


class EmbeddingService:
    """
    Handles loading the embedding model, embedding text, and
    building/loading the Faiss vector index.
    """
    def __init__(self, model_path: str = str(settings.EMBEDDING_MODEL_PATH)):
        self.model_path = model_path
        self._model: Optional[SentenceTransformer] = None

    def load(self):
        """Loads the SentenceTransformer model into memory."""
        if self._model:
            return
        logger.info("Loading embedding model from %s", self.model_path)
        try:
            self._model = SentenceTransformer(self.model_path)
        except Exception as e:
            logger.exception("Failed to load embedding model: %s", e)
            raise

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embeds a list of texts into numpy vectors."""
        if self._model is None:
            logger.error("Embedding model not loaded. Call load() first.")
            raise RuntimeError("Embedding model not loaded")

        logger.info("Embedding %d texts...", len(texts))
        embeddings = np.array(self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True))
        return embeddings.astype("float32")

    def build_index(
        self,
        docs: List[Dict[str, Any]],
        index_path: Path = settings.VECTOR_INDEX_PATH,
        docstore_path: Path = settings.DOCSTORE_PATH
    ):
        """
        Builds a Faiss index (flat L2) and saves the index + docstore metadata.

        Args:
            docs: List of dicts, e.g., { 'id': str, 'path': str, 'chunk': str }
        """
        if self._model is None:
            self.load()

        texts = [d["chunk"] for d in docs]
        if not texts:
            logger.warning("No docs provided to build_index.")
            raise ValueError("No docs to index")

        embeddings = self.embed(texts)
        dim = embeddings.shape[1]
        logger.info("Embedding dimension: %d  |  documents: %d", dim, len(texts))

        # Build flat index
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_path))

        # Save docstore as JSON (safer than pickle)
        with open(docstore_path, "w", encoding="utf-8") as f:
            json.dump(docs, f)

        logger.info("Saved index to %s and docstore to %s", index_path, docstore_path)

    def load_index(
        self,
        index_path: Path = settings.VECTOR_INDEX_PATH,
        docstore_path: Path = settings.DOCSTORE_PATH
    ) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
        """Loads the Faiss index and JSON docstore from disk."""
        if not index_path.exists() or not docstore_path.exists():
            logger.error("Index (%s) or docstore (%s) not found.", index_path, docstore_path)
            raise FileNotFoundError(f"Index or docstore not found. Searched paths: {index_path}, {docstore_path}")

        logger.info("Loading index from %s", index_path)
        index = faiss.read_index(str(index_path))

        logger.info("Loading docstore from %s", docstore_path)
        with open(docstore_path, "r", encoding="utf-8") as f:
            docs = json.load(f)

        logger.info("Loaded index and %d docs", len(docs))
        return index, docs

    @staticmethod
    def search(index: faiss.Index, docs: List[Dict[str, Any]], query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs a search on the index and returns the top-k documents.

        Returns:
            List of dicts: [{"score": float, "doc": dict}]
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # D = distances (L2), I = indices
        D, I = index.search(query_embedding.astype("float32"), top_k)

        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(docs):
                # Faiss can return -1 if not enough neighbors are found
                continue
            results.append({"score": float(dist), "doc": docs[int(idx)]})
        return results

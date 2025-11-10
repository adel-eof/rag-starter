from pathlib import Path
import json
import logging
import sqlite3
import os
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
    building/loading the Faiss vector index and SQLite docstore.
    """
    def __init__(self, model_path: str = str(settings.EMBEDDING_MODEL_PATH)):
        self.model_path = model_path
        self._model: Optional[SentenceTransformer] = None
        self.embedding_dim: Optional[int] = None

    def load(self):
        """Loads the SentenceTransformer model into memory."""
        if self._model:
            return
        logger.info("Loading embedding model from %s", self.model_path)
        try:
            self._model = SentenceTransformer(self.model_path)
            # Store the embedding dimension
            self.embedding_dim = self._model.get_sentence_embedding_dimension()
            logger.info("Embedding model loaded. Dimension: %d", self.embedding_dim)
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

    def _cleanup_tmp_files(self):
        """Removes temporary files in case of an indexing failure."""
        logger.warning("Cleaning up temporary files after failed indexing.")
        try:
            if settings.VECTOR_INDEX_PATH_TMP.exists():
                settings.VECTOR_INDEX_PATH_TMP.unlink()
            if settings.DB_PATH_TMP.exists():
                settings.DB_PATH_TMP.unlink()
            if settings.ID_MAP_PATH_TMP.exists():
                settings.ID_MAP_PATH_TMP.unlink()
        except Exception as e:
            logger.exception("Error cleaning up temporary files: %s", e)

    def build_index(
        self,
        docs: List[Dict[str, Any]]
    ):
        """
        Builds a Faiss HNSW index, an ID map, and a SQLite docstore.
        Uses temporary files and atomic renames for safety.

        Args:
            docs: List of dicts, e.g., { 'id': str, 'path': str, 'chunk': str }
        """
        if self._model is None or self.embedding_dim is None:
            self.load()

        texts = [d["chunk"] for d in docs]
        ids = [d["id"] for d in docs]

        if not texts:
            logger.warning("No docs provided to build_index.")
            raise ValueError("No docs to index")

        embeddings = self.embed(texts)
        dim = embeddings.shape[1]
        if dim != self.embedding_dim:
            logger.warning("Model dim %d != embedding dim %d", self.embedding_dim, dim)

        logger.info("Embedding dimension: %d  |  documents: %d", dim, len(texts))

        try:
            # 1. Build and save Faiss HNSW index to a temporary file
            logger.info("Building HNSW index...")
            index = faiss.IndexHNSWFlat(dim, settings.HNSW_M)
            index.hnsw.efConstruction = settings.HNSW_EF_CONSTRUCTION
            index.add(embeddings)

            settings.VECTOR_INDEX_PATH_TMP.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(index, str(settings.VECTOR_INDEX_PATH_TMP))
            logger.info("Saved temp Faiss index to %s", settings.VECTOR_INDEX_PATH_TMP)

            # 2. Build and save SQLite docstore to a temporary file
            conn = None
            try:
                if settings.DB_PATH_TMP.exists():
                    settings.DB_PATH_TMP.unlink()

                conn = sqlite3.connect(settings.DB_PATH_TMP)
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE chunks (
                        id TEXT PRIMARY KEY,
                        path TEXT NOT NULL,
                        chunk TEXT NOT NULL
                    )
                """)

                cursor.executemany(
                    "INSERT INTO chunks (id, path, chunk) VALUES (?, ?, ?)",
                    [(d["id"], d["path"], d["chunk"]) for d in docs]
                )

                conn.commit()
                conn.close()
                logger.info("Saved temp docstore to %s", settings.DB_PATH_TMP)
            except Exception as e:
                logger.exception("Failed to build SQLite docstore: %s", e)
                if conn:
                    conn.close()
                raise

            # 3. Save the ID map to a temporary file
            with open(settings.ID_MAP_PATH_TMP, "w", encoding="utf-8") as f:
                json.dump(ids, f)
            logger.info("Saved temp ID map to %s", settings.ID_MAP_PATH_TMP)

            # 4. Atomic rename: Move temp files to final paths
            logger.info("Atomically moving files to final destination...")
            os.rename(settings.VECTOR_INDEX_PATH_TMP, settings.VECTOR_INDEX_PATH)
            os.rename(settings.DB_PATH_TMP, settings.DB_PATH)
            os.rename(settings.ID_MAP_PATH_TMP, settings.ID_MAP_PATH)
            logger.info("Indexing complete. Files are live.")

        except Exception as e:
            logger.exception("Indexing failed: %s", e)
            self._cleanup_tmp_files()
            raise


    def load_index(
        self
    ) -> Tuple[faiss.Index, List[str]]:
        """
        Loads the Faiss index and the ID map from disk.
        (The SQLite DB is loaded on-demand by the server).
        """
        if not settings.VECTOR_INDEX_PATH.exists() or not settings.ID_MAP_PATH.exists():
            logger.error("Index (%s) or ID map (%s) not found.", settings.VECTOR_INDEX_PATH, settings.ID_MAP_PATH)
            raise FileNotFoundError(f"Index or ID map not found.")

        logger.info("Loading index from %s", settings.VECTOR_INDEX_PATH)
        index = faiss.read_index(str(settings.VECTOR_INDEX_PATH))

        # Set efSearch for HNSW index on load
        if isinstance(index, faiss.IndexHNSW):
            # Parameter for search speed/accuracy trade-off
            index.hnsw.efSearch = 128
            logger.info("Set HNSW efSearch parameter to 128.")

        logger.info("Loading ID map from %s", settings.ID_MAP_PATH)
        with open(settings.ID_MAP_PATH, "r", encoding="utf-8") as f:
            id_map = json.load(f)

        logger.info("Loaded index and %d ID mappings", len(id_map))
        return index, id_map

    @staticmethod
    def search(index: faiss.Index, id_map: List[str], query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs a search on the index and returns the top-k document IDs.

        Returns:
            List of dicts: [{"score": float, "id": str}]
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        D, I = index.search(query_embedding.astype("float32"), top_k)

        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(id_map):
                # Faiss can return -1 if not enough neighbors are found
                continue
            results.append({"score": float(dist), "id": id_map[int(idx)]})
        return results

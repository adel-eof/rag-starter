from pathlib import Path
import json
import logging
import sqlite3
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
        docs: List[Dict[str, Any]]
    ):
        """
        Builds a Faiss index, an ID map, and a SQLite docstore.

        Args:
            docs: List of dicts, e.g., { 'id': str, 'path': str, 'chunk': str }
        """
        if self._model is None:
            self.load()

        texts = [d["chunk"] for d in docs]
        ids = [d["id"] for d in docs]

        if not texts:
            logger.warning("No docs provided to build_index.")
            raise ValueError("No docs to index")

        embeddings = self.embed(texts)
        dim = embeddings.shape[1]
        logger.info("Embedding dimension: %d  |  documents: %d", dim, len(texts))

        # 1. Build and save Faiss index
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        settings.VECTOR_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(settings.VECTOR_INDEX_PATH))
        logger.info("Saved Faiss index to %s", settings.VECTOR_INDEX_PATH)

        # 2. Build and save SQLite docstore
        try:
            # Remove old DB if it exists
            if settings.DB_PATH.exists():
                settings.DB_PATH.unlink()

            conn = sqlite3.connect(settings.DB_PATH)
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
            logger.info("Saved docstore to %s", settings.DB_PATH)
        except Exception as e:
            logger.exception("Failed to build SQLite docstore: %s", e)
            if conn:
                conn.close()
            raise

        # 3. Save the ID map (maps Faiss index-id to chunk-id)
        with open(settings.ID_MAP_PATH, "w", encoding="utf-8") as f:
            json.dump(ids, f)
        logger.info("Saved ID map to %s", settings.ID_MAP_PATH)


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

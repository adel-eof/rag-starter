import pytest
import numpy as np
from pathlib import Path
from src.embedding_service import EmbeddingService
from src import config

# ------------------------------
# Mocks
# ------------------------------

class DummySentenceTransformer:
    """Mock for sentence_transformers.SentenceTransformer"""
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, show_progress_bar=False, convert_to_numpy=True):
        # deterministic dummy vectors (len x dim)
        return np.array([[float(len(t)%10)]*8 for t in texts], dtype="float32")

@pytest.fixture
def patched_embedding_service(monkeypatch, tmp_path):
    """Fixture to create an EmbeddingService with a patched SentenceTransformer."""
    # Patch SentenceTransformer in the module where it's imported
    import src.embedding_service as embmod
    monkeypatch.setattr(embmod, "SentenceTransformer", DummySentenceTransformer)

    # Patch config to use temp paths
    monkeypatch.setattr(config.settings, "VECTOR_INDEX_PATH", tmp_path / "test.faiss")
    monkeypatch.setattr(config.settings, "DOCSTORE_PATH", tmp_path / "test.json")

    svc = EmbeddingService(model_path=str(tmp_path / "fake-model-dir"))
    return svc

# ------------------------------
# Tests
# ------------------------------

def test_embedding_service_load(patched_embedding_service):
    svc = patched_embedding_service
    svc.load()
    assert svc._model is not None
    assert isinstance(svc._model, DummySentenceTransformer)

def test_embedding_service_embed(patched_embedding_service):
    svc = patched_embedding_service
    svc.load()

    vecs = svc.embed(["hello world", "another text"])

    assert vecs.shape == (2, 8) # 2 texts, 8 dims
    assert vecs.dtype == "float32"
    # Check deterministic output
    # len("hello world") % 10 = 11 % 10 = 1
    # len("another text") % 10 = 12 % 10 = 2
    assert np.array_equal(vecs[0], np.array([1.0]*8, dtype="float32"))
    assert np.array_equal(vecs[1], np.array([2.0]*8, dtype="float32"))

def test_build_and_load_index(patched_embedding_service, tmp_path):
    svc = patched_embedding_service

    docs = [
        {"id": "doc1", "path": "/f/a", "chunk": "hello world"},
        {"id": "doc2", "path": "/f/b", "chunk": "another text"},
    ]

    svc.build_index(docs)

    # Check that files were created
    assert (tmp_path / "test.faiss").exists()
    assert (tmp_path / "test.json").exists()

    # Test loading
    index, loaded_docs = svc.load_index()

    assert index.ntotal == 2
    assert index.d == 8
    assert loaded_docs == docs

def test_search(patched_embedding_service):
    svc = patched_embedding_service
    svc.load()

    # Create a dummy index
    import faiss
    dim = 8
    index = faiss.IndexFlatL2(dim)
    # Add the embeddings our mock would create
    vecs = svc.embed(["hello world", "another text"])
    index.add(vecs)

    docs = [
        {"id": "doc1", "path": "/f/a", "chunk": "hello world"},
        {"id": "doc2", "path": "/f/b", "chunk": "another text"},
    ]

    # Query with an embedding close to "hello world" (all 1s)
    query_vec = np.array([[1.1]*8], dtype="float32")

    results = svc.search(index, docs, query_vec, top_k=1)

    assert len(results) == 1
    assert results[0]["doc"]["id"] == "doc1"
    assert "score" in results[0]

def test_build_index_no_docs(patched_embedding_service):
    """Test that building an index with no documents raises an error."""
    svc = patched_embedding_service
    with pytest.raises(ValueError, match="No docs to index"):
        svc.build_index([])

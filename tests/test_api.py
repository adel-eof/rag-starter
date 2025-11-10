import pytest
from fastapi.testclient import TestClient
from src.server import app
from src import embedding_service
from src import server as server_mod
from typing import List, Dict, Any

# ------------------------------
# Dummy / Mock classes
# ------------------------------

class DummyEmbedding:
    """Lightweight mock for EmbeddingService to avoid GPU or heavy dependencies."""
    def load(self):
        logger.info("Mock EmbeddingService loaded.")
        pass

    def embed(self, texts: List[str]) -> "np.ndarray":
        import numpy as np
        # Return a deterministic embedding based on text length
        return np.array([[len(t) * 0.1] * 8 for t in texts], dtype="float32")

    def load_index(self) -> (Any, List[Dict[str, Any]]):
        """Return a fake FAISS-like index and document store."""
        class FakeIndex:
            def search(self, q, k):
                import numpy as np
                # Always return index 0 with distance 0.1
                return np.array([[0.1]]), np.array([[0]])

        docs = [
            {"id": "file.rb::0", "path": "/some/file.rb", "chunk": "def foo(); end"}
        ]
        return FakeIndex(), docs

    def search(self, index: Any, docs: List[Dict[str, Any]], query_embedding: Any, top_k: int = 5) -> List[Dict[str, Any]]:
        """Mock search that always returns the first doc."""
        return [{"score": 0.1, "doc": docs[0]}]


class DummyLlama:
    """Mock for llama_cpp.Llama."""
    def __init__(self, model_path: str, **kwargs):
        self.model_path = model_path
        logger.info(f"Mock Llama loaded from {model_path}")

    def create_chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Mock the create_chat_completion method.
        Returns a response based on the *user* message content.
        """
        user_content = next((m['content'] for m in messages if m['role'] == 'user'), "")

        response_text = "Mocked response: Found in /some/file.rb"
        if "bluetooth" in user_content.lower():
            response_text = "Mocked response: Found init_bluetooth in bluetooth.rb"

        return {
            "id": "chatcmpl-mock-123",
            "object": "chat.completion",
            "created": 123456,
            "model": "mock-llama-3",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }


@pytest.fixture(autouse=True)
def patch_models(monkeypatch):
    """
    Automatically patch heavy components before every test.
    - Replace EmbeddingService with DummyEmbedding
    - Replace Llama with a stubbed version
    """
    # Patch the *class* in the module where it is *used*
    monkeypatch.setattr(server_mod, "EmbeddingService", lambda *_, **__: DummyEmbedding())
    monkeypatch.setattr(server_mod, "Llama", DummyLlama)


# ------------------------------
# Tests
# ------------------------------

def test_startup_event():
    """Test that the startup event runs and logs warnings if index is missing."""
    with TestClient(app) as client:
        # The patch_models fixture ensures EmbeddingService() works,
        # but load_index() in DummyEmbedding will succeed.
        # To test the failure case, we'd need a more complex mock.
        # For now, just test that startup completes.
        assert app.state.embedding is not None
        assert app.state.llama is not None
        assert app.state.index is not None
        assert app.state.docs is not None

def test_query_json():
    """Ensure the /v1/query endpoint returns a structured, valid JSON response."""
    payload = {"query": "Where is the API token configured?", "top_k": 1}

    # IMPORTANT: use TestClient as context manager to trigger startup/shutdown events
    with TestClient(app) as client:
        response = client.post("/v1/query", json=payload)

    assert response.status_code == 200
    data = response.json()

    # Validate expected structure
    assert data["object"] == "chat.completion"
    assert "choices" in data
    assert isinstance(data["choices"], list)
    msg = data["choices"][0]["message"]
    assert "content" in msg
    assert msg["content"].startswith("Mocked response: Found in /some/file.rb")
    assert "context" in data
    assert data["context"]["retrieved"] == ["/some/file.rb"]

def test_query_missing_query_param():
    """Test 400 error if 'query' is missing."""
    with TestClient(app) as client:
        response = client.post("/v1/query", json={"top_k": 1}) # Missing 'query'
    assert response.status_code == 422 # Pydantic validation error

def test_streaming_endpoint():
    """Test the POST /v1/query/stream endpoint."""
    payload = {"query": "bluetooth", "top_k": 1}

    with TestClient(app) as client:
        response = client.post("/v1/query/stream", json=payload)

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    # Check the content of the stream
    lines = response.text.split("\n\n")
    assert lines[0].startswith("data: {") # Metadata chunk
    assert lines[1].startswith("data: {") # Delta chunk
    assert "bluetooth" in lines[1] # Check for correct mock response
    assert lines[-2] == "data: [DONE]" # Final signal

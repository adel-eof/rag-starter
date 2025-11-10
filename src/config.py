import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Set, Dict, Any

def get_home() -> Path:
    """Returns the user's home directory."""
    return Path.home()

class Settings(BaseSettings):
    """
    Configuration for the Offline Code Assistant, loaded from environment
    variables or a .env file.
    """
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    # ---------------------------------------------------------------------
    # Paths
    # ---------------------------------------------------------------------
    CODEBASE_PATH: Path = Field(
        default_factory=lambda: get_home() / "projects/rag-starter/sample_code",
        description="Path to the codebase root to be indexed."
    )
    EMBEDDING_MODEL_PATH: Path = Field(
        default_factory=lambda: get_home() / "projects/llm-models/all-MiniLM-L6-v2",
        description="Path to the SentenceTransformer embedding model directory."
    )
    LLAMA_MODEL_PATH: Path = Field(
        default_factory=lambda: get_home() / "projects/llm-models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        description="Path to the Llama GGUF model file."
    )

    # ---------------------------------------------------------------------
    # Storage paths
    # ---------------------------------------------------------------------
    DATA_DIR: Path = Path.cwd() / "data"
    VECTOR_INDEX_PATH: Path = DATA_DIR / "vector_index.faiss"
    DOCSTORE_PATH: Path = DATA_DIR / "docs_chunks.json"

    # ---------------------------------------------------------------------
    # Indexing and chunking
    # ---------------------------------------------------------------------
    INDEX_EXTENSIONS: Set[str] = {".m", ".h", ".rb", ".xml", ".py", ".ini"}
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    # ---------------------------------------------------------------------
    # Llama-cpp options
    # ---------------------------------------------------------------------
    LLAMA_N_CTX: int = Field(default=4096, alias="LLAMA_N_CTX", description="Context window size for Llama.")
    LLAMA_THREADS: int = Field(default=8, alias="LLAMA_THREADS", description="Number of CPU threads for Llama.")
    LLAMA_MAX_PROMPT_CHARS: int = Field(default=8000, description="Truncate prompts if they exceed this length.")
    LLM_N_GPU_LAYERS: int = Field(default=-1, alias="LLM_N_GPU_LAYERS", description="Number of layers to offload to GPU (-1 for all).")
    LLM_VERBOSE: bool = Field(default=False, alias="LLM_VERBOSE", description="Enable verbose logging from llama_cpp.")

    # ---------------------------------------------------------------------
    # Llama decode options
    # ---------------------------------------------------------------------
    LLAMA_MAX_TOKENS: int = Field(default=512, alias="LLAMA_MAX_TOKENS")
    LLAMA_TEMPERATURE: float = Field(default=0.0, alias="LLAMA_TEMPERATURE")
    LLAMA_TOP_P: float = Field(default=0.95, alias="LLAMA_TOP_P")

    @property
    def LLAMA_DECODE_OPTIONS(self) -> Dict[str, Any]:
        """Returns the dictionary of decoding options for Llama."""
        return {
            "max_tokens": self.LLAMA_MAX_TOKENS,
            "temperature": self.LLAMA_TEMPERATURE,
            "top_p": self.LLAMA_TOP_P,
        }

# Create a single settings instance to be imported by other modules
settings = Settings()

# Ensure data directory exists
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)

from functools import lru_cache
from typing import Dict, List, Optional

from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    log_level: str = "INFO"
    # Base URL of the LLM server (OpenAI-compatible). Default points to local Ollama/llama.cpp style endpoint.
    llm_server: str = "http://host.docker.internal:11434"
    llm_api_key: str = "minirag"
    llm_embedding_model: str = "qwen3-embedding:0.6b"
    llm_generative_model: str = "qwen3:1.7b"
    llm_chat_path: str = "/v1/chat/completions"  # Adjust for OpenAI/Azure endpoints.
    llm_api_version: Optional[str] = None  # Azure OpenAI: e.g. "2024-02-15-preview"
    llm_deployment: Optional[str] = None  # Azure OpenAI deployment name.
    llm_request_timeout: float = 30.0
    llm_max_retries: int = 3
    llm_retry_backoff: float = 2.0  # seconds, exponential backoff base
    index_directory: str = "/index"
    documents_path: str = "documents"
    allowed_origins: str = "*"

    # Prompts per domain.
    prompt: str = "" # Default por si es necesario, hablamos ayer que por defecto podría ser bulletins, lo dejo vacío para determinar cual debe ser
    prompt_news: str = "Respon citant notícies del DOGV."
    prompt_bulletins: str = "Respon citant butlletins del BOUA."
    prompt_parliament: str = "Explica les funcions del parlament."

    # Chunking and retrieval.
    chunk_size: int = 500
    chunk_overlap: int = 100
    top_k: int = 4
    temperature: float = 0.5
    skip_rebuild: bool = False  # If true, do not rebuild index on startup (assumes persisted Chroma)

    languages: List[str] = Field(default_factory=lambda: ["va"])
    domains: List[str] = Field(default_factory=lambda: ["parliament", "news", "bulletins"])
    use_chroma: bool = True  # Persist embeddings to avoid full rebuilds on restart.

    class Config:
        env_prefix = "APP_"
        case_sensitive = False

    def prompt_for_domain(self, domain: str) -> str:
        prompts: Dict[str, str] = {
            "news": self.prompt_news,
            "bulletins": self.prompt_bulletins,
            "parliament": self.prompt_parliament,
        }
        return prompts.get(domain, self.prompt)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[arg-type]

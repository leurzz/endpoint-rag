from functools import lru_cache
from typing import Dict, List, Optional

from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    log_level: str = "INFO"
    llm_server: Optional[HttpUrl] = None
    llm_api_key: str = "minirag"
    llm_embedding_model: str = "qwen3-embedding:0.6b"
    llm_generative_model: str = "qwen3:1.7b"
    index_directory: str = "/index"
    documents_path: str = "documents"
    allowed_origins: str = "*"

    # Prompts per domain.
    prompt: str = "" # Default por si es necesario, hablamos ayer que por defecto podría ser bulletins, lo dejo vacío para determinar cual debe ser
    prompt_news: str = "Responde citando noticias del DOGV."
    prompt_bulletins: str = "Responde citando boletines del BOUA."
    prompt_parliament: str = "Explica las funciones del parlamento."

    # Chunking and retrieval.
    chunk_size: int = 500
    chunk_overlap: int = 100
    top_k: int = 4
    temperature: float = 0.5

    languages: List[str] = Field(default_factory=lambda: ["va"])
    domains: List[str] = Field(default_factory=lambda: ["parliament", "news", "bulletins"])

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

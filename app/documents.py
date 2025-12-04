import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .embeddings import EmbeddingService, cosine_similarity
from .settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    id: str
    text: str
    language: str
    domain: str
    source_path: str
    chunk_index: int
    embedding: np.ndarray
    metadata: Dict[str, str]


class DocumentStore:
    def __init__(self, settings: Settings, embedding_service: EmbeddingService) -> None:
        self.settings = settings
        self.embedding_service = embedding_service
        self.chunks: List[DocumentChunk] = []

    def build(self) -> None:
        documents_path = Path(self.settings.documents_path)
        if not documents_path.exists():
            logger.warning("Documents directory not found: %s", documents_path)
            return

        logger.info("Building index from %s", documents_path)
        for jsonl_file in documents_path.glob("*.jsonl"):
            domain = self._infer_domain(jsonl_file)
            self._process_file(jsonl_file, domain)

        logger.info("Indexed %s chunks", len(self.chunks))

    def _infer_domain(self, path: Path) -> str:
        stem = path.stem.lower()
        for domain in self.settings.domains:
            if domain in stem:
                return domain
        return "parliament"

    def _process_file(self, path: Path, domain: str) -> None:
        logger.info("Processing %s for domain=%s", path.name, domain)
        with path.open("r", encoding="utf-8") as fh:
            for idx, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping invalid JSON at %s:%s", path, idx + 1)
                    continue
                text = payload.get("text")
                language = payload.get("language")
                title = payload.get("title") or path.stem
                if not text or not language:
                    logger.debug("Skipping entry missing text or language at %s:%s", path, idx + 1)
                    continue
                if language not in self.settings.languages:
                    continue
                for chunk_index, chunk_text in enumerate(self._chunk_text(text)):
                    embedding = self.embedding_service.embed(chunk_text)
                    chunk = DocumentChunk(
                        id=f"{path.stem}-{idx}-{chunk_index}",
                        text=chunk_text,
                        language=language,
                        domain=domain,
                        source_path=str(path),
                        chunk_index=chunk_index,
                        embedding=embedding,
                        metadata={"title": title},
                    )
                    self.chunks.append(chunk)

    def _chunk_text(self, text: str) -> Iterable[str]:
        size = max(self.settings.chunk_size, 1)
        overlap = max(min(self.settings.chunk_overlap, size - 1), 0)
        start = 0
        while start < len(text):
            end = min(len(text), start + size)
            yield text[start:end]
            start += size - overlap

    def retrieve(
        self,
        query: str,
        language: str,
        domain: str,
        top_k: Optional[int] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        if not self.chunks:
            logger.warning("No indexed chunks available.")
            return []

        query_embedding = self.embedding_service.embed(query)
        filtered = [
            chunk for chunk in self.chunks if chunk.language == language and chunk.domain == domain
        ]
        scored: List[Tuple[DocumentChunk, float]] = []
        for chunk in filtered:
            score = cosine_similarity(query_embedding, chunk.embedding)
            scored.append((chunk, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        limit = top_k or self.settings.top_k
        return scored[:limit]

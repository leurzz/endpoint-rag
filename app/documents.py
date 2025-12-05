import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import os

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
        self.use_chroma = settings.use_chroma
        self.chroma_client = None
        self.collections: Dict[str, object] = {}

        if self.use_chroma:
            try:
                import chromadb

                self.chroma_client = chromadb.PersistentClient(path=self.settings.index_directory)
            except Exception as exc:
                logger.error("Failed to initialize Chroma, falling back to in-memory store: %s", exc)
                self.use_chroma = False

    def build(self) -> None:
        documents_path = Path(self.settings.documents_path)
        if not documents_path.exists():
            logger.warning("Documents directory not found: %s", documents_path)
            return

        logger.info("Building index from %s", documents_path)
        count = 0
        for jsonl_file in documents_path.glob("*.jsonl"):
            domain = self._infer_domain(jsonl_file)
            count += self._process_file(jsonl_file, domain)

        if self.use_chroma:
            logger.info("Indexed %s chunks into Chroma (persistent).", count)
        else:
            logger.info("Indexed %s chunks in memory.", len(self.chunks))

    def _infer_domain(self, path: Path) -> str:
        stem = path.stem.lower()
        for domain in self.settings.domains:
            if domain in stem:
                return domain
        return "parliament"

    def _process_file(self, path: Path, domain: str) -> int:
        logger.info("Processing %s for domain=%s", path.name, domain)
        added = 0
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
                raw_language = payload.get("language")
                title = payload.get("title") or path.stem
                if not text or not raw_language:
                    logger.debug("Skipping entry missing text or language at %s:%s", path, idx + 1)
                    continue
                language = self.match_language(str(raw_language))
                if not language:
                    continue
                for chunk_index, chunk_text in enumerate(self._chunk_text(text)):
                    embedding = self.embedding_service.embed(chunk_text)
                    metadata = {
                        "title": title,
                        "source_path": str(path),
                        "chunk_index": chunk_index,
                        "language": language,
                        "domain": domain,
                    }
                    chunk_id = f"{path.stem}-{idx}-{chunk_index}"
                    if self.use_chroma and self.chroma_client:
                        collection = self._get_collection(domain, language)
                        collection.upsert(
                            ids=[chunk_id],
                            documents=[chunk_text],
                            metadatas=[metadata],
                            embeddings=[embedding.tolist()],
                        )
                    else:
                        chunk = DocumentChunk(
                            id=chunk_id,
                            text=chunk_text,
                            language=language,
                            domain=domain,
                            source_path=str(path),
                            chunk_index=chunk_index,
                            embedding=embedding,
                            metadata={"title": title},
                        )
                        self.chunks.append(chunk)
                    added += 1
        return added

    def match_language(self, raw_language: str) -> Optional[str]:
        """
        Return a normalized language code if any allowed language appears in the raw string.
        Accepts patterns like "va|es" or "va,es" or "va es".
        """
        tokens = re.split(r"[\\|,\\s]+", raw_language.lower())
        for token in tokens:
            if token in self.settings.languages:
                return token
        if raw_language.lower() in self.settings.languages:
            return raw_language.lower()
        return None

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
        if not self.use_chroma and not self.chunks:
            logger.warning("No indexed chunks available.")
            return []

        limit = top_k or self.settings.top_k
        query_embedding = self.embedding_service.embed(query)

        if self.use_chroma and self.chroma_client:
            collection = self._get_collection(domain, language)
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=limit,
                include=["documents", "metadatas", "distances", "embeddings"],
            )
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            ids = results.get("ids", [[]])[0]
            dists = results.get("distances", [[]])[0] if results.get("distances") else []
            chunks: List[Tuple[DocumentChunk, float]] = []
            for idx, doc in enumerate(docs):
                meta = metas[idx] if idx < len(metas) else {}
                chunk_id = ids[idx] if idx < len(ids) else f"{domain}-{language}-{idx}"
                distance = dists[idx] if idx < len(dists) else 0.0
                score = 1.0 - float(distance)  # cosine distance -> similarity
                chunk = DocumentChunk(
                    id=chunk_id,
                    text=doc,
                    language=meta.get("language", language),
                    domain=meta.get("domain", domain),
                    source_path=meta.get("source_path", ""),
                    chunk_index=int(meta.get("chunk_index", idx)),
                    embedding=np.array([], dtype=np.float32),
                    metadata={"title": meta.get("title", "")},
                )
                chunks.append((chunk, score))
            return chunks

        # In-memory fallback.
        filtered = [
            chunk for chunk in self.chunks if chunk.language == language and chunk.domain == domain
        ]
        scored: List[Tuple[DocumentChunk, float]] = []
        for chunk in filtered:
            score = cosine_similarity(query_embedding, chunk.embedding)
            scored.append((chunk, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    def _get_collection(self, domain: str, language: str):
        if not self.chroma_client:
            raise RuntimeError("Chroma client not initialized")
        name = f"{domain}_{language}"
        if name not in self.collections:
            self.collections[name] = self.chroma_client.get_or_create_collection(
                name=name, metadata={"hnsw:space": "cosine"}
            )
        return self.collections[name]

import logging
from typing import List, Tuple

import httpx

from .documents import DocumentChunk, DocumentStore
from .schemas import ChatMessage, ContextItem
from .settings import Settings

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, store: DocumentStore, settings: Settings) -> None:
        self.store = store
        self.settings = settings

    def retrieve(self, query: str, language: str, domain: str) -> List[ContextItem]:
        results = self.store.retrieve(query=query, language=language, domain=domain)
        contexts: List[ContextItem] = []
        for chunk, score in results:
            contexts.append(
                ContextItem(
                    id=chunk.id,
                    title=chunk.metadata.get("title"),
                    passage=chunk.text,
                    metadata={
                        "score": score,
                        "source_path": chunk.source_path,
                        "chunk_index": chunk.chunk_index,
                    },
                )
            )
        return contexts


class Generator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        if self.settings.llm_server:
            self.client = httpx.AsyncClient(
                base_url=str(self.settings.llm_server),
                headers={"Authorization": f"Bearer {self.settings.llm_api_key}"},
                timeout=30.0,
            )

    async def shutdown(self) -> None:
        if self.client:
            await self.client.aclose()

    async def generate(
        self,
        history: List[ChatMessage],
        prompt: str,
        domain: str,
        language: str,
        contexts: List[ContextItem],
    ) -> str:
        if self.client:
            try:
                return await self._generate_via_llm(history, prompt, domain, language, contexts)
            except Exception:
                logger.exception("Falling back to local generation after LLM call failed.")
        return self._generate_locally(prompt, domain, contexts)

    async def _generate_via_llm(
        self,
        history: List[ChatMessage],
        prompt: str,
        domain: str,
        language: str,
        contexts: List[ContextItem],
    ) -> str:
        assert self.client is not None
        system_prompt = self.settings.prompt_for_domain(domain)
        messages = [{"role": "system", "content": system_prompt}]
        messages += [msg.model_dump() for msg in history]
        context_text = "\n\n".join(f"- {c.passage}" for c in contexts)
        messages.append(
            {
                "role": "user",
                "content": f"Idioma: {language}\n\nContexto:\n{context_text}\n\nPregunta: {prompt}",
            }
        )
        payload = {
            "model": self.settings.llm_generative_model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "stream": False,
        }
        response = await self.client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            return self._generate_locally(prompt, domain, contexts)
        return choices[0]["message"]["content"]

    def _generate_locally(self, prompt: str, domain: str, contexts: List[ContextItem]) -> str:
        system_prompt = self.settings.prompt_for_domain(domain)
        top_contexts = "\n\n".join(c.passage for c in contexts) or "Sin contexto disponible."
        answer = (
            f"{system_prompt}\n\n"
            f"Pregunta: {prompt}\n\n"
            f"Contexto usado:\n{top_contexts}\n\n"
            "Respuesta generada localmente (sustituir por llamada al LLM configurado)."
        )
        return answer

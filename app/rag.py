import asyncio
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
            base_url = str(self.settings.llm_server).rstrip("/")
            self.client = httpx.AsyncClient(
                base_url=base_url,
                headers={"Authorization": f"Bearer {self.settings.llm_api_key}"},
                timeout=self.settings.llm_request_timeout,
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
        path = self._resolve_chat_path()
        response = await self._post_with_retry(path, payload)
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            return self._generate_locally(prompt, domain, contexts)
        return choices[0]["message"]["content"]

    def _resolve_chat_path(self) -> str:
        """
        Build the chat completions path, avoiding double /v1 and supporting Azure-style paths.
        """
        base = str(self.settings.llm_server).rstrip("/")
        path = self.settings.llm_chat_path

        # Azure OpenAI path if deployment/version provided.
        if self.settings.llm_deployment and self.settings.llm_api_version:
            path = (
                f"/openai/deployments/{self.settings.llm_deployment}/chat/completions"
                f"?api-version={self.settings.llm_api_version}"
            )

        # Prevent double /v1 if base already ends with /v1.
        if base.endswith("/v1") and path.startswith("/v1"):
            path = path[len("/v1") :] or "/chat/completions"

        if not path.startswith("/"):
            path = "/" + path
        return path

    async def _post_with_retry(self, path: str, payload: dict) -> httpx.Response:
        assert self.client is not None
        retries = max(self.settings.llm_max_retries, 1)
        backoff = max(self.settings.llm_retry_backoff, 0.5)
        for attempt in range(retries):
            try:
                response = await self.client.post(path, json=payload)
                if response.status_code == 429 and attempt < retries - 1:
                    delay = self._retry_delay(response, backoff, attempt)
                    logger.warning("LLM 429 Too Many Requests, retrying in %.2fs", delay)
                    await asyncio.sleep(delay)
                    continue
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code if exc.response else None
                if status in {500, 502, 503, 504} and attempt < retries - 1:
                    delay = self._retry_delay(exc.response, backoff, attempt)
                    logger.warning("LLM %s error, retrying in %.2fs", status, delay)
                    await asyncio.sleep(delay)
                    continue
                raise
        raise RuntimeError("Exhausted retries calling LLM.")

    def _retry_delay(self, response: httpx.Response, backoff: float, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
        return backoff * (2**attempt)

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

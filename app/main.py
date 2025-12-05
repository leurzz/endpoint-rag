import logging
import asyncio
import json
import re
from typing import List

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from .documents import DocumentStore
from .embeddings import EmbeddingService
from .rag import Generator, Retriever
from .schemas import (
    GetConfigResponse,
    Mode,
    PredictRequest,
    PredictResponse,
)
from .settings import get_settings

settings = get_settings()
numeric_level = logging._nameToLevel.get(
    settings.log_level.upper(), logging.INFO)
logging.basicConfig(level=numeric_level)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MiniRAG Backend",
    version="1.0.0",
    description="FastAPI implementation for MiniRAG endpoints.",
)

if settings.allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip()
                       for origin in settings.allowed_origins.split(",")],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

embedding_service = EmbeddingService()
document_store = DocumentStore(
    settings=settings, embedding_service=embedding_service)
retriever = Retriever(store=document_store, settings=settings)
generator = Generator(settings=settings)


@app.on_event("startup")
async def startup_event() -> None:
    document_store.build()
    await generator.start()
    logger.info("Application ready with %s indexed chunks.",
                len(document_store.chunks))


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await generator.shutdown()


@app.get("/get_config", response_model=GetConfigResponse)
async def get_config() -> GetConfigResponse:
    modes: List[Mode] = []
    for domain in settings.domains:
        for lang in settings.languages:
            modes.append(Mode(language=lang, domain=domain))
    return GetConfigResponse(modes=modes)


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest, http_request: Request) -> Response:
    """
    Streaming endpoint: devuelve la respuesta palabra a palabra via SSE.
    """
    predict_response = await _run_rag(request)
    contexts_payload = [c.model_dump() for c in predict_response.contexts]

    async def event_stream():
        buffer = ""
        for token in _stream_tokens(predict_response.response):
            buffer += token
            payload = {"response": buffer, "contexts": contexts_payload}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.02)  # pequeño delay para simular stream
        final_payload = {"response": buffer, "contexts": contexts_payload}
        yield f"data: {json.dumps(final_payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/query", response_model=PredictResponse)
async def query(request: PredictRequest) -> JSONResponse:
    """
    Endpoint no streaming: responde con JSON completo.
    """
    predict_response = await _run_rag(request)
    return JSONResponse(content=predict_response.model_dump())


async def _run_rag(request: PredictRequest) -> PredictResponse:
    matched_language = document_store.match_language(request.language)
    if not matched_language:
        raise HTTPException(status_code=400, detail="Unsupported language.")
    if request.domain not in settings.domains:
        raise HTTPException(status_code=400, detail="Unsupported domain.")

    query_text = request.prompt  # usar prompt actual como query de retrieval
    contexts = retriever.retrieve(
        query=query_text, language=matched_language, domain=request.domain)
    answer = await generator.generate(
        history=request.history,
        prompt=request.prompt,
        domain=request.domain,
        language=matched_language,
        contexts=contexts,
    )
    return PredictResponse(response=answer, contexts=contexts)


def get_app() -> FastAPI:
    return app


def _stream_tokens(text: str) -> List[str]:
    """
    Split text into small tokens (word + trailing space) to stream incrementally.
    """
    if not text:
        return []
    return re.findall(r"\S+\s*", text)

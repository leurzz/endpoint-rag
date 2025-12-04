import logging
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .documents import DocumentStore
from .embeddings import EmbeddingService
from .rag import Generator, Retriever
from .schemas import GetConfigResponse, Mode, PredictRequest, PredictResponse
from .settings import get_settings

settings = get_settings()
numeric_level = logging._nameToLevel.get(settings.log_level.upper(), logging.INFO)
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
        allow_origins=[origin.strip() for origin in settings.allowed_origins.split(",")],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

embedding_service = EmbeddingService()
document_store = DocumentStore(settings=settings, embedding_service=embedding_service)
retriever = Retriever(store=document_store, settings=settings)
generator = Generator(settings=settings)


@app.on_event("startup")
async def startup_event() -> None:
    document_store.build()
    await generator.start()
    logger.info("Application ready with %s indexed chunks.", len(document_store.chunks))


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
async def predict(request: PredictRequest) -> PredictResponse:
    if request.language not in settings.languages:
        raise HTTPException(status_code=400, detail="Unsupported language.")
    if request.domain not in settings.domains:
        raise HTTPException(status_code=400, detail="Unsupported domain.")

    query_text = "\n".join(msg.content for msg in request.history) + "\n" + request.prompt
    contexts = retriever.retrieve(query=query_text, language=request.language, domain=request.domain)
    answer = await generator.generate(
        history=request.history,
        prompt=request.prompt,
        domain=request.domain,
        language=request.language,
        contexts=contexts,
    )
    return PredictResponse(response=answer, contexts=contexts)


def get_app() -> FastAPI:
    return app

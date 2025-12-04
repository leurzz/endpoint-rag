from typing import List, Optional

from pydantic import BaseModel, Field


class Mode(BaseModel):
    language: str
    domain: str


class GetConfigResponse(BaseModel):
    modes: List[Mode]


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str


class PredictRequest(BaseModel):
    history: List[ChatMessage]
    prompt: str
    domain: str
    language: str


class ContextItem(BaseModel):
    id: str
    title: Optional[str] = None
    passage: str
    timestamp: Optional[str] = None
    url: Optional[str] = None
    metadata: Optional[dict] = None


class PredictResponse(BaseModel):
    response: str
    contexts: List[ContextItem]

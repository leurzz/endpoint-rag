import hashlib
import logging
import re
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


class EmbeddingService:
    """
    Lightweight, dependency-free embedding generator.

    This is intentionally simple to avoid heavyweight model downloads. It creates
    deterministic bag-of-words vectors hashed into a fixed dimension, which is
    sufficient for local retrieval and can be swapped for a real embedding model.
    """

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def embed(self, text: str) -> np.ndarray:
        tokens = _tokenize(text)
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            bucket = int(digest, 16) % self.dimension
            vector[bucket] += 1.0
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        return [self.embed(t) for t in texts]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        logger.debug("Embedding shape mismatch: %s vs %s", a.shape, b.shape)
        return 0.0
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

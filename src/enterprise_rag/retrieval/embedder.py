import asyncio
from threading import Lock

from sentence_transformers import SentenceTransformer

from enterprise_rag.config import settings

_model: SentenceTransformer | None = None
_model_lock = Lock()


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = SentenceTransformer(settings.embedding_model)
    return _model


def _embed(text: str) -> list[float]:
    vector = _get_model().encode(text, normalize_embeddings=True)
    return vector.tolist()


async def embed_async(text: str) -> list[float]:
    """Embed text in a worker thread, including lazy model loading."""
    return await asyncio.to_thread(_embed, text)

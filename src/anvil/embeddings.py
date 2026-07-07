import os
from functools import lru_cache
from typing import Protocol

from anvil import config


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class OpenAIEmbedder:
    """Primary embedder: openai text-embedding-3-small. Requires OPENAI_API_KEY."""

    name = f"openai/{config.OPENAI_EMBEDDING_MODEL}"
    dim = 1536

    def __init__(self) -> None:
        from langchain_openai import OpenAIEmbeddings

        self._client = OpenAIEmbeddings(model=config.OPENAI_EMBEDDING_MODEL)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed_query(text)


class FallbackEmbedder:
    """Keyless CI embedder: all-MiniLM-L6-v2 on CPU. Clearly labeled as the fallback arm."""

    name = config.FALLBACK_EMBEDDING_MODEL
    dim = 384

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(config.FALLBACK_EMBEDDING_MODEL, device="cpu")
        from anvil.rerank import materialize_weights

        materialize_weights(self._model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0].tolist()


def get_embedder(kind: str = "auto") -> Embedder:
    # Resolve "auto" before the cache so "auto" and its concrete kind share one instance.
    if kind == "auto":
        kind = "openai" if os.environ.get("OPENAI_API_KEY") else "fallback"
    return _load_embedder(kind)


@lru_cache(maxsize=2)
def _load_embedder(kind: str) -> Embedder:
    if kind == "openai":
        return OpenAIEmbedder()
    if kind == "fallback":
        return FallbackEmbedder()
    raise ValueError(f"unknown embedder kind: {kind}")

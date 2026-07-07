from functools import lru_cache

from anvil import config


def materialize_weights(module) -> None:
    # On macOS, BLAS matmuls over safetensors mmap-backed weights can return NaN.
    # Cloning forces the weights into ordinary process memory.
    import torch

    with torch.no_grad():
        for p in module.parameters():
            p.data = p.data.clone()


class CrossEncoderReranker:
    """Standard second-stage reranking: ms-marco-MiniLM-L-6-v2 cross-encoder on CPU."""

    def __init__(self) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(config.RERANKER_MODEL, device="cpu")
        materialize_weights(self._model.model)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        return [float(s) for s in self._model.predict([(query, p) for p in passages])]


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()

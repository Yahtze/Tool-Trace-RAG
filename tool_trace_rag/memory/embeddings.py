from __future__ import annotations

from typing import Protocol

from tool_trace_rag.config import EMBEDDING_MODEL


class EmbeddingProvider(Protocol):
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError


class SentenceTransformerEmbeddingProvider:
    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        if hasattr(self._model, "get_embedding_dimension"):
            self.dimensions = int(self._model.get_embedding_dimension())
        else:
            self.dimensions = int(self._model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

from __future__ import annotations

import logging
import math

from compakt.core.interfaces.embeddings import Embeddings
from compakt.core.interfaces.vector_index import VectorIndex
from compakt.core.models import CompaktChunk, CompaktEmbeddingEntry

logger = logging.getLogger(__name__)


class InMemoryVectorIndex(VectorIndex):
    def __init__(self, embeddings: Embeddings) -> None:
        self._embeddings = embeddings
        self._entries: list[CompaktEmbeddingEntry] = []

    def index(self, chunks: list[CompaktChunk]) -> list[CompaktEmbeddingEntry]:
        if not chunks:
            self._entries = []
            return []

        vectors = self._embeddings.embed([chunk.content for chunk in chunks])
        if not isinstance(vectors, list) or (vectors and not isinstance(vectors[0], list)):
            msg = "Embeddings provider must return a list of vectors for batch inputs"
            raise TypeError(msg)

        self._entries = [CompaktEmbeddingEntry(id=str(i), chunk=chunk, embedding=vector) for i, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))]
        logger.debug("Indexed %d entries", len(self._entries))
        return self._entries

    def similarity_search_with_score(self, query: str, k: int = 20) -> list[tuple[CompaktChunk, float]]:
        if not self._entries:
            return []

        query_vector = self._embeddings.embed(query)
        if not isinstance(query_vector, list) or (query_vector and isinstance(query_vector[0], list)):
            msg = "Embeddings provider must return a single vector for string inputs"
            raise TypeError(msg)

        scored = [(entry.chunk, _cosine_similarity(query_vector, entry.embedding)) for entry in self._entries]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: max(1, min(k, len(scored)))]

    async def asimilarity_search_with_score(self, query: str, k: int = 20) -> list[tuple[CompaktChunk, float]]:
        if not self._entries:
            return []

        query_vector = await self._embeddings.aembed(query)
        if not isinstance(query_vector, list) or (query_vector and isinstance(query_vector[0], list)):
            msg = "Embeddings provider must return a single vector for string inputs"
            raise TypeError(msg)

        scored = [(entry.chunk, _cosine_similarity(query_vector, entry.embedding)) for entry in self._entries]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: max(1, min(k, len(scored)))]

    def clear(self) -> None:
        self._entries = []


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0

    dot_product = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)

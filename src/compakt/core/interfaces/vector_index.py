from typing import Protocol

from compakt.core.models import CompaktChunk, CompaktEmbeddingEntry


class VectorIndex(Protocol):
    def index(self, chunks: list[CompaktChunk]) -> list[CompaktEmbeddingEntry]:
        """Embed and index chunks for similarity search."""
        ...

    def similarity_search_with_score(self, query: str, k: int = 20) -> list[tuple[CompaktChunk, float]]:
        """Find the top-k most similar indexed chunks for a query."""
        ...

    async def asimilarity_search_with_score(self, query: str, k: int = 20) -> list[tuple[CompaktChunk, float]]:
        """Asynchronously find the top-k most similar indexed chunks for a query."""
        ...

    def clear(self) -> None:
        """Clear all indexed entries."""
        ...

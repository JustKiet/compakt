from typing import Protocol

from compakt.core.models import (
    CompaktChunk,
    CompaktEmbeddingEntry,
    CompaktRunResult,
    HeaderNode,
)


class SummarizationStrategy(Protocol):
    name: str

    def can_handle(self, markdown: str, markdown_tree: list[HeaderNode]) -> bool:
        """Return True when this strategy can process the parsed markdown tree."""
        ...

    def run(
        self,
        markdown: str,
        markdown_tree: list[HeaderNode],
        chunks: list[CompaktChunk],
        embeddings: list[CompaktEmbeddingEntry],
        level: int,
        retrieval_k: int,
    ) -> CompaktRunResult:
        """Execute summarization and return summary plus generated artifacts."""
        ...

    async def run_async(
        self,
        markdown: str,
        markdown_tree: list[HeaderNode],
        chunks: list[CompaktChunk],
        embeddings: list[CompaktEmbeddingEntry],
        level: int,
        retrieval_k: int,
    ) -> CompaktRunResult:
        """Asynchronously execute summarization and return summary plus artifacts."""
        ...


class EmbbedingBasedSummarizationStrategy(SummarizationStrategy, Protocol):
    """Marker protocol for strategies that rely on embeddings."""

    ...

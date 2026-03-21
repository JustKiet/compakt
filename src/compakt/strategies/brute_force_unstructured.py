import logging

from compakt.core.interfaces.encoder import Encoder
from compakt.core.interfaces.strategy import SummarizationStrategy
from compakt.core.interfaces.summarizer import Summarizer
from compakt.core.models import (
    CompaktChunk,
    CompaktEmbeddingEntry,
    CompaktRunArtifacts,
    CompaktRunResult,
    HeaderNode,
)

logger = logging.getLogger(__name__)


class BruteForceUnstructuredStrategy(SummarizationStrategy):
    name = "brute_force_unstructured"

    def __init__(self, summarizer: Summarizer, encoder: Encoder, token_limit: int) -> None:
        self._summarizer = summarizer
        self._encoder = encoder
        self._token_limit = token_limit

    def can_handle(self, markdown: str, markdown_tree: list[HeaderNode]) -> bool:
        # Estimate total tokens if we include all chunks.
        total_tokens = len(self._encoder.encode(markdown))  # Tokens in the full markdown
        logger.info("Brute-force strategy token estimate: %d tokens", total_tokens)

        return total_tokens <= self._token_limit

    def run(
        self,
        markdown: str,
        markdown_tree: list[HeaderNode],
        chunks: list[CompaktChunk],
        embeddings: list[CompaktEmbeddingEntry],
        level: int,
        retrieval_k: int,
    ) -> CompaktRunResult:
        logger.info("Brute-force unstructured strategy activated (%d chunks)", len(chunks))

        summary = self._summarizer.summarize_unstructured(markdown, level=level)

        return CompaktRunResult(
            summary=summary,
            artifacts=CompaktRunArtifacts(
                strategy=self.name,
                markdown=markdown,
                markdown_tree=markdown_tree,
                chunks=chunks,
                embeddings=embeddings,
                retrieved_chunks={},
                document_structure=None,
            ),
        )

    async def run_async(
        self,
        markdown: str,
        markdown_tree: list[HeaderNode],
        chunks: list[CompaktChunk],
        embeddings: list[CompaktEmbeddingEntry],
        level: int,
        retrieval_k: int,
    ) -> CompaktRunResult:
        logger.info("Brute-force unstructured strategy activated (%d chunks)", len(chunks))

        summary = await self._summarizer.asummarize_unstructured(markdown, level=level)

        return CompaktRunResult(
            summary=summary,
            artifacts=CompaktRunArtifacts(
                strategy=self.name,
                markdown=markdown,
                markdown_tree=markdown_tree,
                chunks=chunks,
                embeddings=embeddings,
                retrieved_chunks={},
                document_structure=None,
            ),
        )

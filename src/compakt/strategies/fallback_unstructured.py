from __future__ import annotations

import asyncio
import logging

from compakt.core.interfaces.strategy import SummarizationStrategy
from compakt.core.interfaces.summarizer import Summarizer
from compakt.core.interfaces.vector_index import VectorIndex
from compakt.core.models import (
    CompaktChunk,
    CompaktEmbeddingEntry,
    CompaktRunArtifacts,
    CompaktRunResult,
    DocumentNode,
    DocumentStructure,
    HeaderNode,
)

logger = logging.getLogger(__name__)


class FallbackUnstructuredStrategy(SummarizationStrategy):
    name = "fallback_unstructured"
    _DEFAULT_QUERY = "Summarize the most important topics in this document."

    def __init__(self, summarizer: Summarizer, vector_index: VectorIndex) -> None:
        self._summarizer = summarizer
        self._vector_index = vector_index

    def can_handle(self, markdown: str, markdown_tree: list[HeaderNode]) -> bool:
        return len(markdown_tree) == 0

    def run(
        self,
        markdown: str,
        markdown_tree: list[HeaderNode],
        chunks: list[CompaktChunk],
        embeddings: list[CompaktEmbeddingEntry],
        level: int,
        retrieval_k: int,
    ) -> CompaktRunResult:
        logger.info("Fallback unstructured strategy activated (%d chunks)", len(chunks))
        retrieved = _select_fallback_docs(
            results=self._vector_index.similarity_search_with_score(
                self._DEFAULT_QUERY,
                k=retrieval_k,
            ),
            chunks=chunks,
        )
        synthetic_structure = _build_synthetic_structure()

        relevant_docs = {"Summary": retrieved}
        summary = self._summarizer.summarize(relevant_docs, synthetic_structure, level=level)

        return CompaktRunResult(
            summary=summary,
            artifacts=CompaktRunArtifacts(
                markdown=markdown,
                markdown_tree=markdown_tree,
                chunks=chunks,
                embeddings=embeddings,
                retrieved_chunks=relevant_docs,
                document_structure=synthetic_structure,
                strategy=self.name,
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
        results = await asyncio.to_thread(
            self._vector_index.similarity_search_with_score,
            self._DEFAULT_QUERY,
            retrieval_k,
        )
        retrieved = _select_fallback_docs(results=results, chunks=chunks)
        synthetic_structure = _build_synthetic_structure()

        relevant_docs = {"Summary": retrieved}
        summary = await asyncio.to_thread(
            self._summarizer.summarize,
            relevant_docs,
            synthetic_structure,
            level,
        )

        return CompaktRunResult(
            summary=summary,
            artifacts=CompaktRunArtifacts(
                markdown=markdown,
                markdown_tree=markdown_tree,
                chunks=chunks,
                embeddings=embeddings,
                retrieved_chunks=relevant_docs,
                document_structure=synthetic_structure,
                strategy=self.name,
            ),
        )


def _select_fallback_docs(
    results: list[tuple[CompaktChunk, float]],
    chunks: list[CompaktChunk],
) -> list[CompaktChunk]:
    retrieved = [chunk for chunk, _ in results]
    if retrieved:
        return retrieved

    return chunks[: max(1, min(5, len(chunks)))]


def _build_synthetic_structure() -> DocumentStructure:
    return DocumentStructure(
        title="Document",
        children=[
            DocumentNode(
                title="Content",
                children=[
                    DocumentNode(
                        title="Overview",
                        children=[DocumentNode(title="Summary")],
                    )
                ],
            )
        ],
    )

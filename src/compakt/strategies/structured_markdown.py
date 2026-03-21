from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from rapidfuzz import fuzz

from compakt.core.exceptions import InvalidRetrievalLevelError
from compakt.core.interfaces.document_structure_resolver import (
    DocumentStructureResolver,
)
from compakt.core.interfaces.strategy import SummarizationStrategy
from compakt.core.interfaces.summarizer import Summarizer
from compakt.core.interfaces.vector_index import VectorIndex
from compakt.core.models import (
    CompaktChunk,
    CompaktEmbeddingEntry,
    CompaktRunArtifacts,
    CompaktRunResult,
    DocumentStructure,
    HeaderNode,
    MarkdownHeader,
)
from compakt.core.utils import elbow_filter, normalize_markdown_title

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScopeRequest:
    original_title: str
    query_title: str
    scope: dict[str, str]


class StructuredMarkdownStrategy(SummarizationStrategy):
    name = "structured_markdown"

    def __init__(
        self,
        document_structure_resolver: DocumentStructureResolver,
        summarizer: Summarizer,
        vector_index: VectorIndex,
        min_relevance_score: float = 0.25,
        min_scope_match_score: float = 85.0,
    ) -> None:
        self._document_structure_resolver = document_structure_resolver
        self._summarizer = summarizer
        self._vector_index = vector_index
        self._min_relevance_score = min_relevance_score
        self._min_scope_match_score = min_scope_match_score

    def can_handle(self, markdown: str, markdown_tree: list[HeaderNode]) -> bool:
        return len(markdown_tree) > 0

    def run(
        self,
        markdown: str,
        markdown_tree: list[HeaderNode],
        chunks: list[CompaktChunk],
        embeddings: list[CompaktEmbeddingEntry],
        level: int,
        retrieval_k: int,
    ) -> CompaktRunResult:
        if level < 1:
            raise InvalidRetrievalLevelError("Level must be >= 1")

        document_structure = self._document_structure_resolver.resolve(markdown_tree)
        requests = _titles_for_level(document_structure, level)
        logger.info(
            "Resolved document structure: %d scope requests at level %d", len(requests), level
        )

        retrieved_docs: dict[str, list[CompaktChunk]] = {}
        for request in requests:
            results = self._vector_index.similarity_search_with_score(
                request.query_title,
                k=retrieval_k,
            )
            retrieved_docs[request.original_title] = _select_docs_for_scope(
                results=results,
                all_chunks=chunks,
                scope=request.scope,
                retrieval_k=retrieval_k,
                min_relevance_score=self._min_relevance_score,
                min_scope_match_score=self._min_scope_match_score,
            )
            logger.debug(
                "Scope '%s': %d docs retrieved",
                request.original_title,
                len(retrieved_docs[request.original_title]),
            )

        summary = self._summarizer.summarize(retrieved_docs, document_structure, level=level)

        return CompaktRunResult(
            summary=summary,
            artifacts=CompaktRunArtifacts(
                markdown=markdown,
                markdown_tree=markdown_tree,
                chunks=chunks,
                embeddings=embeddings,
                retrieved_chunks=retrieved_docs,
                document_structure=document_structure,
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
        if level < 1:
            raise InvalidRetrievalLevelError("Level must be >= 1")

        document_structure = await self._document_structure_resolver.aresolve(markdown_tree)
        requests = _titles_for_level(document_structure, level)

        retrieval_tasks = [
            _retrieve_title_docs(
                vector_index=self._vector_index,
                chunks=chunks,
                request=request,
                retrieval_k=retrieval_k,
                min_relevance_score=self._min_relevance_score,
                min_scope_match_score=self._min_scope_match_score,
            )
            for request in requests
        ]

        retrieved_docs: dict[str, list[CompaktChunk]] = {}
        for original_title, docs in await asyncio.gather(*retrieval_tasks):
            retrieved_docs[original_title] = docs

        summary = await self._summarizer.asummarize(
            retrieved_docs,
            document_structure,
            level,
        )

        return CompaktRunResult(
            summary=summary,
            artifacts=CompaktRunArtifacts(
                markdown=markdown,
                markdown_tree=markdown_tree,
                chunks=chunks,
                embeddings=embeddings,
                retrieved_chunks=retrieved_docs,
                document_structure=document_structure,
                strategy=self.name,
            ),
        )


async def _retrieve_title_docs(
    vector_index: VectorIndex,
    chunks: list[CompaktChunk],
    request: ScopeRequest,
    retrieval_k: int,
    min_relevance_score: float,
    min_scope_match_score: float,
) -> tuple[str, list[CompaktChunk]]:
    """Retrieve docs for one scope in async mode and apply the shared selector."""
    results = await vector_index.asimilarity_search_with_score(
        request.query_title,
        retrieval_k,
    )
    docs = _select_docs_for_scope(
        results=results,
        all_chunks=chunks,
        scope=request.scope,
        retrieval_k=retrieval_k,
        min_relevance_score=min_relevance_score,
        min_scope_match_score=min_scope_match_score,
    )

    return request.original_title, docs


def _select_docs_for_scope(
    results: list[tuple[CompaktChunk, float]],
    all_chunks: list[CompaktChunk],
    scope: dict[str, str],
    retrieval_k: int,
    min_relevance_score: float,
    min_scope_match_score: float,
) -> list[CompaktChunk]:
    """Choose scoped docs with a clear fallback chain.

    Selection order:
    1) Elbow-filtered + relevance-threshold docs within scope.
    2) Elbow-filtered docs within scope (ignores relevance threshold).
    3) All chunks that match the scope from the original chunk list.
    4) Global elbow-filtered docs from similarity results.
    """
    scoped_results = _filter_results_to_scope(
        results,
        scope,
        min_scope_match_score=min_scope_match_score,
    )
    elbow_scoped = elbow_filter(scoped_results)

    thresholded_docs = [chunk for chunk, score in elbow_scoped if score >= min_relevance_score][
        :retrieval_k
    ]
    if thresholded_docs:
        logger.debug(
            "Scope selection: fallback #1 (filtered + threshold), %d docs", len(thresholded_docs)
        )
        return thresholded_docs

    elbow_scoped_docs = [chunk for chunk, _ in elbow_scoped][:retrieval_k]
    if elbow_scoped_docs:
        logger.debug(
            "Scope selection: fallback #2 (filtered, no threshold), %d docs", len(elbow_scoped_docs)
        )
        return elbow_scoped_docs

    all_scoped_docs = _all_chunks_in_scope(
        all_chunks,
        scope,
        min_scope_match_score=min_scope_match_score,
    )
    if all_scoped_docs:
        logger.debug(
            "Scope selection: fallback #3 (all chunks in scope), %d docs",
            len(all_scoped_docs[:retrieval_k]),
        )
        return all_scoped_docs[:retrieval_k]

    global_docs = [chunk for chunk, _ in elbow_filter(results)][:retrieval_k]
    logger.debug("Scope selection: fallback #4 (global filtered), %d docs", len(global_docs))
    return global_docs


def _titles_for_level(document_structure: DocumentStructure, level: int) -> list[ScopeRequest]:
    titles: list[ScopeRequest] = []

    for ancestor_path, node in document_structure.get_nodes_at_depth(level):
        scope = {f"header_{i + 2}": title for i, title in enumerate(ancestor_path)}
        scope[f"header_{level + 1}"] = node.title
        query_title = ": ".join([*ancestor_path, node.title])
        titles.append(ScopeRequest(node.title, query_title, scope))

    return titles


def _filter_results_to_scope(
    results: list[tuple[CompaktChunk, float]],
    scope: dict[str, str],
    min_scope_match_score: float = 85.0,
) -> list[tuple[CompaktChunk, float]]:
    return [
        (chunk, score)
        for chunk, score in results
        if _chunk_matches_scope(
            chunk,
            scope,
            min_scope_match_score=min_scope_match_score,
        )
    ]


def _all_chunks_in_scope(
    chunks: list[CompaktChunk],
    scope: dict[str, str],
    min_scope_match_score: float = 85.0,
) -> list[CompaktChunk]:
    return [
        chunk
        for chunk in chunks
        if _chunk_matches_scope(
            chunk,
            scope,
            min_scope_match_score=min_scope_match_score,
        )
    ]


def _chunk_matches_scope(
    chunk: CompaktChunk,
    scope: dict[str, str],
    min_scope_match_score: float = 85.0,
) -> bool:
    """Return True when a chunk belongs to the requested header scope.

    Matching rules:
    - Every key in ``scope`` (e.g. ``header_2``, ``header_3``) must match.
    - Values are compared with fuzzy WRatio after markdown normalization.
    - For the deepest scoped key, if metadata is missing, use the chunk's
      canonical ``header_type`` + ``header_name`` fields as fallback.
    """
    metadata = chunk.metadata
    deepest_key = _deepest_scope_key(scope)

    for key, expected_value in scope.items():
        actual_value = metadata.get(key)
        if actual_value is None:
            actual_value = _resolve_deepest_scope_value(
                chunk=chunk,
                scope_key=key,
                deepest_scope_key=deepest_key,
            )

        if actual_value is None:
            return False

        if not _is_scope_title_match(
            actual_value,
            expected_value,
            min_scope_match_score=min_scope_match_score,
        ):
            return False
    return True


def _is_scope_title_match(
    actual_value: str,
    expected_value: str,
    min_scope_match_score: float,
) -> bool:
    """Fuzzy-compare two scope titles after markdown-aware normalization."""
    normalized_actual = _normalize_title(actual_value)
    normalized_expected = _normalize_title(expected_value)

    if not normalized_actual or not normalized_expected:
        return normalized_actual == normalized_expected

    return fuzz.WRatio(normalized_actual, normalized_expected) >= min_scope_match_score


def _normalize_title(value: str) -> str:
    return normalize_markdown_title(value)


def _deepest_scope_key(scope: dict[str, str]) -> str | None:
    if not scope:
        return None

    return max(scope.keys(), key=_scope_key_depth)


def _scope_key_depth(scope_key: str) -> int:
    parts = scope_key.split("_", maxsplit=1)
    if len(parts) != 2:
        return -1

    try:
        return int(parts[1])
    except ValueError:
        return -1


def _resolve_deepest_scope_value(
    chunk: CompaktChunk,
    scope_key: str,
    deepest_scope_key: str | None,
) -> str | None:
    if scope_key != deepest_scope_key:
        return None

    expected_header_type = _header_type_for_scope_key(scope_key)
    if expected_header_type is not None and chunk.header_type != expected_header_type:
        return None

    return chunk.header_name or None


def _header_type_for_scope_key(scope_key: str) -> MarkdownHeader | None:
    mapping: dict[str, MarkdownHeader] = {
        "header_1": MarkdownHeader.H1,
        "header_2": MarkdownHeader.H2,
        "header_3": MarkdownHeader.H3,
        "header_4": MarkdownHeader.H4,
        "header_5": MarkdownHeader.H5,
        "header_6": MarkdownHeader.H6,
    }
    return mapping.get(scope_key)

from __future__ import annotations

import unittest

from compakt.core.models import (
    CompaktChunk,
    CompaktEmbeddingEntry,
    HeaderNode,
    MarkdownHeader,
)
from compakt.strategies.fallback_unstructured import (
    FallbackUnstructuredStrategy,
    _select_fallback_docs,
)


def _chunk(content: str = "body") -> CompaktChunk:
    return CompaktChunk(
        header_type=MarkdownHeader.H2,
        header_name="test",
        content=content,
    )


class CanHandleTest(unittest.TestCase):
    def test_empty_tree_returns_true(self) -> None:
        strategy = FallbackUnstructuredStrategy(
            summarizer=None,  # type: ignore[arg-type]
            vector_index=None,  # type: ignore[arg-type]
        )
        self.assertTrue(strategy.can_handle([]))

    def test_non_empty_tree_returns_false(self) -> None:
        strategy = FallbackUnstructuredStrategy(
            summarizer=None,  # type: ignore[arg-type]
            vector_index=None,  # type: ignore[arg-type]
        )
        tree: list[HeaderNode] = [{"title": "Doc", "level": 1}]
        self.assertFalse(strategy.can_handle(tree))


class SelectFallbackDocsTest(unittest.TestCase):
    def test_returns_results_when_available(self) -> None:
        chunks = [_chunk("a"), _chunk("b")]
        results = [(c, 0.9) for c in chunks]
        docs = _select_fallback_docs(results=results, chunks=chunks)
        self.assertEqual(len(docs), 2)

    def test_fallback_to_first_chunks_when_empty(self) -> None:
        chunks = [_chunk(f"c{i}") for i in range(10)]
        docs = _select_fallback_docs(results=[], chunks=chunks)
        self.assertEqual(len(docs), 5)

    def test_fallback_single_chunk(self) -> None:
        chunks = [_chunk("only")]
        docs = _select_fallback_docs(results=[], chunks=chunks)
        self.assertEqual(len(docs), 1)


class FallbackRunTest(unittest.TestCase):
    def test_run_with_fakes(self) -> None:
        chunks = [_chunk("content A"), _chunk("content B")]
        embeddings = [
            CompaktEmbeddingEntry(id="0", chunk=chunks[0], embedding=[0.1]),
            CompaktEmbeddingEntry(id="1", chunk=chunks[1], embedding=[0.2]),
        ]

        class _FakeVectorIndex:
            def similarity_search_with_score(
                self, query: str, k: int = 20
            ) -> list[tuple[CompaktChunk, float]]:
                return [(c, 0.5) for c in chunks]

            def index(self, c: list[CompaktChunk]) -> list[CompaktEmbeddingEntry]:
                return embeddings

            def clear(self) -> None:
                pass

        class _FakeSummarizer:
            def summarize(self, relevant_docs: object, doc_structure: object, level: int = 3) -> str:
                return "Fake summary"

        strategy = FallbackUnstructuredStrategy(
            summarizer=_FakeSummarizer(),
            vector_index=_FakeVectorIndex(),
        )
        result = strategy.run(
            markdown="plain text",
            markdown_tree=[],
            chunks=chunks,
            embeddings=embeddings,
            level=2,
            retrieval_k=10,
        )

        self.assertEqual(result.summary, "Fake summary")
        self.assertEqual(result.artifacts.strategy, "fallback_unstructured")
        self.assertIsNotNone(result.artifacts.document_structure)


if __name__ == "__main__":
    unittest.main()

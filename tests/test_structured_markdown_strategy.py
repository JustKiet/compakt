from __future__ import annotations

import unittest

from compakt.core.models import (
    CompaktChunk,
    CompaktEmbeddingEntry,
    DocumentNode,
    DocumentStructure,
    HeaderNode,
    MarkdownHeader,
)
from compakt.strategies.structured_markdown import (
    StructuredMarkdownStrategy,
    _all_chunks_in_scope,
    _chunk_matches_scope,
    _is_scope_title_match,
    _select_docs_for_scope,
    _titles_for_level,
)


def _chunk(
    header_name: str = "Test",
    header_type: MarkdownHeader = MarkdownHeader.H2,
    content: str = "body",
    metadata: dict[str, str] | None = None,
) -> CompaktChunk:
    return CompaktChunk(
        header_type=header_type,
        header_name=header_name,
        content=content,
        metadata=metadata or {},
    )


def _embedding(chunk: CompaktChunk, idx: int = 0) -> CompaktEmbeddingEntry:
    return CompaktEmbeddingEntry(
        id=str(idx),
        chunk=chunk,
        embedding=[0.1, 0.2],
    )


def _doc_structure() -> DocumentStructure:
    return DocumentStructure(
        title="Doc",
        children=[
            DocumentNode(
                title="Section A",
                children=[
                    DocumentNode(
                        title="Sub A1",
                        children=[DocumentNode(title="Point X")],
                    ),
                ],
            ),
            DocumentNode(
                title="Section B",
                children=[
                    DocumentNode(
                        title="Sub B1",
                        children=[DocumentNode(title="Point Y")],
                    ),
                ],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# _titles_for_level
# ---------------------------------------------------------------------------

class TitlesForLevelTest(unittest.TestCase):
    def test_level_1_returns_section_titles(self) -> None:
        requests = _titles_for_level(_doc_structure(), level=1)
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0].original_title, "Section A")
        self.assertEqual(requests[1].original_title, "Section B")
        self.assertIn("header_2", requests[0].scope)

    def test_level_2_returns_subsection_titles(self) -> None:
        requests = _titles_for_level(_doc_structure(), level=2)
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0].original_title, "Sub A1")
        self.assertIn("header_2", requests[0].scope)
        self.assertIn("header_3", requests[0].scope)

    def test_level_3_returns_h4_titles(self) -> None:
        requests = _titles_for_level(_doc_structure(), level=3)
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0].original_title, "Point X")
        self.assertIn("header_4", requests[0].scope)


# ---------------------------------------------------------------------------
# _chunk_matches_scope
# ---------------------------------------------------------------------------

class ChunkMatchesScopeTest(unittest.TestCase):
    def test_exact_metadata_match(self) -> None:
        chunk = _chunk(metadata={"header_2": "Section A", "header_3": "Sub A1"})
        self.assertTrue(
            _chunk_matches_scope(chunk, {"header_2": "Section A", "header_3": "Sub A1"})
        )

    def test_metadata_mismatch(self) -> None:
        chunk = _chunk(metadata={"header_2": "Introduction to Machine Learning"})
        self.assertFalse(
            _chunk_matches_scope(
                chunk, {"header_2": "Conclusion and Future Work"}
            )
        )

    def test_missing_non_deepest_key_fails(self) -> None:
        chunk = _chunk(
            header_type=MarkdownHeader.H3,
            header_name="Sub A1",
            metadata={"header_3": "Sub A1"},  # missing header_2
        )
        self.assertFalse(
            _chunk_matches_scope(chunk, {"header_2": "Section A", "header_3": "Sub A1"})
        )

    def test_deepest_key_fallback_to_header_name(self) -> None:
        chunk = _chunk(
            header_type=MarkdownHeader.H3,
            header_name="Sub A1",
            metadata={"header_2": "Section A"},  # missing header_3
        )
        self.assertTrue(
            _chunk_matches_scope(chunk, {"header_2": "Section A", "header_3": "Sub A1"})
        )


# ---------------------------------------------------------------------------
# _is_scope_title_match
# ---------------------------------------------------------------------------

class IsScopeTitleMatchTest(unittest.TestCase):
    def test_identical_titles(self) -> None:
        self.assertTrue(_is_scope_title_match("Section A", "Section A", 85.0))

    def test_case_insensitive(self) -> None:
        self.assertTrue(_is_scope_title_match("section a", "Section A", 85.0))

    def test_very_different_titles(self) -> None:
        self.assertFalse(
            _is_scope_title_match("Apples", "Zebras in the wild", 85.0)
        )

    def test_both_empty(self) -> None:
        self.assertTrue(_is_scope_title_match("", "", 85.0))


# ---------------------------------------------------------------------------
# _select_docs_for_scope  (fallback paths)
# ---------------------------------------------------------------------------

class SelectDocsForScopeTest(unittest.TestCase):
    def _make_results(
        self, scores: list[float], scope_meta: dict[str, str] | None = None,
    ) -> tuple[list[tuple[CompaktChunk, float]], list[CompaktChunk]]:
        chunks = [
            _chunk(
                content=f"c{i}",
                metadata=scope_meta or {"header_2": "Section A"},
            )
            for i in range(len(scores))
        ]
        results = list(zip(chunks, scores, strict=True))
        return results, chunks

    def test_fallback_1_threshold_and_filter(self) -> None:
        results, chunks = self._make_results([0.9, 0.85, 0.3, 0.1])
        docs = _select_docs_for_scope(
            results=results,
            all_chunks=chunks,
            scope={"header_2": "Section A"},
            retrieval_k=10,
            min_relevance_score=0.25,
            min_scope_match_score=85.0,
        )
        self.assertGreater(len(docs), 0)

    def test_fallback_4_global_when_no_scope_match(self) -> None:
        # Chunks have metadata that does NOT match the requested scope.
        results, chunks = self._make_results(
            [0.9, 0.5],
            scope_meta={"header_2": "Completely Different"},
        )
        docs = _select_docs_for_scope(
            results=results,
            all_chunks=chunks,
            scope={"header_2": "Section A"},
            retrieval_k=10,
            min_relevance_score=0.25,
            min_scope_match_score=85.0,
        )
        # Should fall through to global fallback.
        self.assertGreater(len(docs), 0)

    def test_retrieval_k_respected(self) -> None:
        results, chunks = self._make_results([0.9] * 20)
        docs = _select_docs_for_scope(
            results=results,
            all_chunks=chunks,
            scope={"header_2": "Section A"},
            retrieval_k=5,
            min_relevance_score=0.25,
            min_scope_match_score=85.0,
        )
        self.assertLessEqual(len(docs), 5)


# ---------------------------------------------------------------------------
# can_handle
# ---------------------------------------------------------------------------

class CanHandleTest(unittest.TestCase):
    def test_non_empty_tree(self) -> None:
        class _FakeResolver:
            def resolve(self, headers: list[HeaderNode]) -> DocumentStructure:
                return _doc_structure()

        class _FakeSummarizer:
            def summarize(self, *a: object, **kw: object) -> str:
                return ""

        class _FakeVectorIndex:
            def index(self, chunks: list[CompaktChunk]) -> list[CompaktEmbeddingEntry]:
                return []
            def similarity_search_with_score(self, query: str, k: int = 20) -> list[tuple[CompaktChunk, float]]:
                return []
            def clear(self) -> None:
                pass

        strategy = StructuredMarkdownStrategy(
            document_structure_resolver=_FakeResolver(),
            summarizer=_FakeSummarizer(),
            vector_index=_FakeVectorIndex(),
        )
        tree: list[HeaderNode] = [{"title": "Doc", "level": 1}]
        self.assertTrue(strategy.can_handle(tree))

    def test_empty_tree(self) -> None:
        strategy = StructuredMarkdownStrategy(
            document_structure_resolver=None,  # type: ignore[arg-type]
            summarizer=None,  # type: ignore[arg-type]
            vector_index=None,  # type: ignore[arg-type]
        )
        self.assertFalse(strategy.can_handle([]))


if __name__ == "__main__":
    unittest.main()

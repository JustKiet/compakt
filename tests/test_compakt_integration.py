from __future__ import annotations

import asyncio
import unittest
from typing import Any, cast

from dependency_injector import providers

from compakt import AsyncCompakt, Container
from compakt.core.interfaces.strategy import SummarizationStrategy
from compakt.core.models import (
    CompaktChunk,
    DocumentNode,
    DocumentStructure,
    HeaderNode,
    MarkdownHeader,
)


class _FakeFileReader:
    def read(self, file_path: str) -> str:
        return "# Demo\n## Section A\n### Subsection A1\n#### Point A\nBody text"


class _FakeResolver:
    def resolve(self, headers: list[HeaderNode]) -> DocumentStructure:
        return DocumentStructure(
            title="Demo",
            children=[
                DocumentNode(
                    title="Section A",
                    children=[
                        DocumentNode(
                            title="Subsection A1",
                            children=[DocumentNode(title="Point A")],
                        )
                    ],
                )
            ],
        )


class _FakeSummarizer:
    def summarize(
        self,
        relevant_docs: dict[str, list[CompaktChunk]],
        doc_structure: DocumentStructure,
        level: int = 3,
    ) -> str:
        return (
            f"Summary for {doc_structure.title} at level {level} "
            f"with {len(relevant_docs)} retrieval groups"
        )


class _FakeEmbeddings:
    def embed(self, payload: str | list[str]) -> list[float] | list[list[float]]:
        def _vector_for(text: str) -> list[float]:
            length = float(len(text))
            checksum = float(sum(ord(ch) for ch in text) % 997)
            return [length, checksum]

        if isinstance(payload, str):
            return _vector_for(payload)

        return [_vector_for(item) for item in payload]


class _FakeEncoder:
    def encode(self, text: str) -> list[int]:
        return [ord(ch) for ch in text]


class CompaktIntegrationTest(unittest.TestCase):
    def test_container_compakt_runs_and_returns_artifacts(self) -> None:
        container = Container()
        container_any: Any = container
        container_any.file_reader.override(providers.Object(_FakeFileReader()))
        container_any.document_structure_resolver.override(
            providers.Object(_FakeResolver())
        )
        container_any.summarizer.override(providers.Object(_FakeSummarizer()))
        container_any.embeddings.override(providers.Object(_FakeEmbeddings()))
        container_any.encoder.override(providers.Object(_FakeEncoder()))

        compakt = container.compakt()
        result = compakt.summarize("ignored.pdf", level=3, retrieval_k=5)

        self.assertIn("Summary for Demo", result.summary)
        self.assertEqual(result.artifacts.strategy, "structured_markdown")
        self.assertGreater(len(result.artifacts.chunks), 0)
        self.assertEqual(
            len(result.artifacts.chunks),
            len(result.artifacts.embeddings),
        )
        self.assertGreater(len(result.artifacts.retrieved_chunks), 0)
        document_structure = result.artifacts.document_structure
        if document_structure is None:
            self.fail("Expected document_structure in structured strategy result")
        self.assertEqual(document_structure.title, "Demo")
        self.assertEqual(result.artifacts.chunks[0].header_type, MarkdownHeader.H4)

    def test_async_compakt_runs_and_returns_artifacts(self) -> None:
        container = Container()
        container_any: Any = container
        container_any.file_reader.override(providers.Object(_FakeFileReader()))
        container_any.document_structure_resolver.override(
            providers.Object(_FakeResolver())
        )
        container_any.summarizer.override(providers.Object(_FakeSummarizer()))
        container_any.embeddings.override(providers.Object(_FakeEmbeddings()))
        container_any.encoder.override(providers.Object(_FakeEncoder()))

        async_compakt = AsyncCompakt(
            file_reader=container.file_reader(),
            markdown_tree_parser=container.markdown_tree_parser(),
            text_splitter=container.text_splitter(),
            vector_index=container.vector_index(),
            strategies=cast(list[SummarizationStrategy], container.strategies()),
            encoder=container.encoder(),
        )

        result = asyncio.run(
            async_compakt.summarize("ignored.pdf", level=3, retrieval_k=5)
        )

        self.assertIn("Summary for Demo", result.summary)
        self.assertEqual(result.artifacts.strategy, "structured_markdown")
        self.assertGreater(len(result.artifacts.chunks), 0)
        self.assertEqual(
            len(result.artifacts.chunks),
            len(result.artifacts.embeddings),
        )
        self.assertGreater(len(result.artifacts.retrieved_chunks), 0)
        document_structure = result.artifacts.document_structure
        if document_structure is None:
            self.fail("Expected document_structure in structured strategy result")
        self.assertEqual(document_structure.title, "Demo")


if __name__ == "__main__":
    unittest.main()

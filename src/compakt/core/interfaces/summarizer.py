from typing import Protocol

from compakt.core.models import CompaktChunk, DocumentStructure


class Summarizer(Protocol):
    def summarize(
        self,
        relevant_docs: dict[str, list[CompaktChunk]],
        doc_structure: DocumentStructure | None,
        level: int = 3,
    ) -> str:
        """
        Summarizes relevant chunks based on a document structure.

        Args:
            relevant_docs: Chunks selected per section/subsection title.
            doc_structure: Structured document hierarchy, or None for unstructured summarization.
            level: Requested summary granularity.
        Returns:
            str: A concise structured summary.
        """
        ...

    async def asummarize(
        self,
        relevant_docs: dict[str, list[CompaktChunk]],
        doc_structure: DocumentStructure | None,
        level: int = 3,
    ) -> str:
        """
        Asynchronously summarizes relevant chunks based on a document structure.

        Args:
            relevant_docs: Chunks selected per section/subsection title.
            doc_structure: Structured document hierarchy, or None for unstructured summarization.
            level: Requested summary granularity.
        Returns:
            str: A concise structured summary.
        """
        ...

    def summarize_unstructured(
        self,
        markdown: str,
        level: int = 3,
    ) -> str:
        """
        Summarizes an unstructured markdown document.

        Args:
            markdown: The raw markdown content to summarize.
            level: Requested summary granularity.
        Returns:
            str: A concise summary of the markdown content.
        """
        ...

    async def asummarize_unstructured(
        self,
        markdown: str,
        level: int = 3,
    ) -> str:
        """
        Asynchronously summarizes an unstructured markdown document.

        Args:
            markdown: The raw markdown content to summarize.
            level: Requested summary granularity.
        Returns:
            str: A concise summary of the markdown content.
        """
        ...

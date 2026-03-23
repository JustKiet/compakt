from __future__ import annotations

from enum import Enum
from typing import TypedDict

from pydantic import BaseModel, Field
from typing_extensions import NotRequired


class MarkdownHeader(Enum):
    H1 = "#"
    H2 = "##"
    H3 = "###"
    H4 = "####"
    H5 = "#####"
    H6 = "######"


class CompaktChunk(BaseModel):
    header_type: MarkdownHeader
    header_name: str
    content: str
    metadata: dict[str, str] = Field(default_factory=dict)


class CompaktEmbeddingEntry(BaseModel):
    id: str
    chunk: CompaktChunk
    embedding: list[float]


class CompaktRunArtifacts(BaseModel):
    markdown: str
    markdown_tree: list[HeaderNode]
    chunks: list[CompaktChunk]
    embeddings: list[CompaktEmbeddingEntry]
    retrieved_chunks: dict[str, list[CompaktChunk]]
    document_structure: DocumentStructure | None
    strategy: str


class CompaktRunResult(BaseModel):
    summary: str
    artifacts: CompaktRunArtifacts


class HeaderNode(TypedDict):
    title: str
    level: int
    children: NotRequired[list[HeaderNode]]


class MarkdownNode(TypedDict):
    title: str
    level: int
    children: list[MarkdownNode]


class DocumentNode(BaseModel):
    """A node in the document hierarchy. Can represent a section, subsection, or any header level."""

    title: str
    """The header title text."""
    children: list[DocumentNode] = Field(default_factory=list)
    """Child nodes. Empty for leaf nodes."""


# Backward-compatible aliases
Section = DocumentNode
Subsection = DocumentNode
H4Header = DocumentNode


class DocumentStructure(BaseModel):
    """Hierarchical structure of a document, resolved from parsed headers."""

    title: str
    """The document title (typically the first H1 header)."""
    children: list[DocumentNode] = Field(default_factory=list)
    """Top-level sections of the document."""

    def get_nodes_at_depth(self, depth: int) -> list[tuple[list[str], DocumentNode]]:
        """Collect nodes at a given depth with their ancestor title path.

        Args:
            depth: Target depth (1 = direct children, 2 = grandchildren, etc.)
        Returns:
            List of (ancestor_titles, node) tuples.
        """
        results: list[tuple[list[str], DocumentNode]] = []

        def _walk(
            node_children: list[DocumentNode],
            ancestors: list[str],
            current_depth: int,
        ) -> None:
            for child in node_children:
                if current_depth == depth:
                    results.append((list(ancestors), child))
                else:
                    _walk(child.children, [*ancestors, child.title], current_depth + 1)

        _walk(self.children, [], 1)
        return results

    def get_document_tree(self, level: int = 3) -> str:
        """Render the document structure as a folder-style tree string.

        Args:
            level: Maximum depth to render (1 = title only, 2 = sections, etc.)
        """
        if level < 1:
            raise ValueError("Level must be >= 1")

        def _build(
            node_title: str,
            node_children: list[DocumentNode],
            prefix: str,
            is_last: bool,
            current_level: int,
        ) -> str:
            connector = "└── " if is_last else "├── "
            tree_str = prefix + connector + node_title + "\n"

            if current_level >= level or not node_children:
                return tree_str

            new_prefix = prefix + ("    " if is_last else "│   ")
            for i, child in enumerate(node_children):
                last_child = i == len(node_children) - 1
                tree_str += _build(
                    child.title, child.children, new_prefix, last_child, current_level + 1
                )
            return tree_str

        tree = self.title + "\n"
        if level > 1:
            for i, child in enumerate(self.children):
                tree += _build(child.title, child.children, "", i == len(self.children) - 1, 2)
        return tree

    # --- Backward-compatible convenience methods ---

    @property
    def sections(self) -> list[DocumentNode]:
        """Alias for children (backward compatibility)."""
        return self.children

    def get_section_titles(self) -> list[str]:
        return [f"{self.title}: {child.title}" for child in self.children]

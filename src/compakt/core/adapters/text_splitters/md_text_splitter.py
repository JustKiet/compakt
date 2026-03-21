from typing import Any, cast

from langchain_text_splitters import MarkdownHeaderTextSplitter

from compakt.core.interfaces.text_splitters import TextSplitter
from compakt.core.models import CompaktChunk, MarkdownHeader
from compakt.core.utils import normalize_markdown_title


class LangchainMarkdownTextSplitter(TextSplitter):
    def __init__(self, headers_to_split_on: list[tuple[MarkdownHeader, str]]) -> None:
        super().__init__()
        self._headers_to_split_on = headers_to_split_on
        self._lc_headers_to_split_on: list[tuple[str, str]] = []

        for header_enum, metadata_key in headers_to_split_on:
            self._lc_headers_to_split_on.append((header_enum.value, metadata_key))

        self._splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self._lc_headers_to_split_on
        )

    def split(self, text: str) -> list[CompaktChunk]:
        """
        Splits the input markdown text into chunks based on the specified headers.

        Args:
            text (str): The input markdown text to be split.
        Returns:
            list[CompaktChunk]: A list of CompaktChunk objects, where each chunk contains a header and its corresponding content.
        """
        lc_chunks = self._splitter.split_text(text)

        compakt_chunks: list[CompaktChunk] = []
        for document in lc_chunks:
            raw_metadata: object = cast(Any, document).metadata
            metadata = cast(dict[str, str], raw_metadata)
            header_enum = self._resolve_header_for_document(metadata)
            header_name = self._resolve_header_name(metadata, header_enum)

            compakt_chunks.append(
                CompaktChunk(
                    header_type=header_enum,
                    header_name=header_name,
                    content=document.page_content,
                    metadata=metadata,
                )
            )
        return compakt_chunks

    def _resolve_header_for_document(self, metadata: dict[str, str]) -> MarkdownHeader:
        # Prefer the deepest configured header that appears in metadata.
        for header_enum, metadata_key in reversed(self._headers_to_split_on):
            if metadata_key in metadata:
                return header_enum

        msg = f"Could not resolve markdown header from metadata keys: {list(metadata.keys())}"
        raise ValueError(msg)

    def _resolve_header_name(
        self,
        metadata: dict[str, str],
        header_enum: MarkdownHeader,
    ) -> str:
        for configured_header, metadata_key in self._headers_to_split_on:
            if configured_header == header_enum:
                raw_name = metadata.get(metadata_key, "")
                return normalize_markdown_title(raw_name)

        return ""

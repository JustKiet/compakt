from __future__ import annotations

import json
import logging
import textwrap
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from compakt.core.interfaces.document_structure_resolver import (
    DocumentStructureResolver,
)
from compakt.core.interfaces.encoder import Encoder
from compakt.core.models import DocumentStructure, HeaderNode

logger = logging.getLogger(__name__)


class OpenAIDocumentStructureResolver(DocumentStructureResolver):
    PROMPT = textwrap.dedent("""
    Given a list of headers with their levels and texts, determine the hierarchical structure of the document.
    The first header should be considered the document title.
    Organize the remaining headers into a recursive tree of nodes, where each node has a title and an optional list of children.
    Higher-level headers (e.g., H2) become top-level children of the document, and lower-level headers (e.g., H3, H4) nest inside them.
    The tree can be arbitrarily deep — do not limit it to a fixed number of levels.
    Note that the input headers may be incorrect or inconsistent; resolve the correct structure based on the header contents rather than relying solely on header levels.
    """)

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        encoder: Encoder | None = None,
        max_input_tokens: int = 120_000,
        timeout: float | None = 120.0,
        **kwargs: Any,
    ):
        self._model = model
        self._kwargs = kwargs
        self._client: BaseChatModel = ChatOpenAI(model=model, timeout=timeout, **kwargs)
        self._llm = self._client.with_structured_output(DocumentStructure)  # type: ignore[assignment]
        self._encoder = encoder
        self._max_input_tokens = max_input_tokens

    def resolve(self, headers: list[HeaderNode]) -> DocumentStructure:
        """
        Resolve the document structure based on the provided headers.

        Args:
            headers: A list of dictionaries, where each dictionary contains metadata about a header (e.g., header level and text).

        Returns:
            A DocumentStructure object representing the hierarchical structure of the document.
        """
        headers_text = json.dumps(headers, ensure_ascii=False)
        headers_text = self._truncate_if_needed(headers_text, headers)

        response = self._llm.invoke(  # type: ignore[partial-unknown]
            [
                {"role": "system", "content": self.PROMPT},
                {"role": "user", "content": headers_text},
            ]
        )
        if not isinstance(response, DocumentStructure):
            raise ValueError("Expected response to be of type DocumentStructure")

        return response

    async def aresolve(self, headers: list[HeaderNode]) -> DocumentStructure:
        """
        Asynchronously resolve the document structure based on the provided headers.

        Args:
            headers: A list of dictionaries, where each dictionary contains metadata about a header (e.g., header level and text).

        Returns:
            A DocumentStructure object representing the hierarchical structure of the document.
        """
        headers_text = json.dumps(headers, ensure_ascii=False)
        headers_text = self._truncate_if_needed(headers_text, headers)

        response = await self._llm.ainvoke(  # type: ignore[partial-unknown]
            [
                {"role": "system", "content": self.PROMPT},
                {"role": "user", "content": headers_text},
            ]
        )
        if not isinstance(response, DocumentStructure):
            raise ValueError("Expected response to be of type DocumentStructure")

        return response

    def _truncate_if_needed(self, headers_text: str, headers: list[HeaderNode]) -> str:
        """Count tokens and truncate headers if over the limit."""
        if self._encoder is None:
            return headers_text

        full_text = self.PROMPT + headers_text
        token_count = len(self._encoder.encode(full_text))
        logger.info("Resolver prompt token count: %d", token_count)

        if token_count <= self._max_input_tokens:
            return headers_text

        logger.warning(
            "Resolver prompt (%d tokens) exceeds limit (%d). Truncating headers.",
            token_count,
            self._max_input_tokens,
        )

        # Remove headers from the end until we fit.
        truncated = list(headers)
        while token_count > self._max_input_tokens and len(truncated) > 1:
            truncated.pop()
            headers_text = json.dumps(truncated, ensure_ascii=False)
            full_text = self.PROMPT + headers_text
            token_count = len(self._encoder.encode(full_text))

        logger.info(
            "Truncated resolver headers from %d to %d (%d tokens)",
            len(headers),
            len(truncated),
            token_count,
        )
        return headers_text

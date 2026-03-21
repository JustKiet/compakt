from __future__ import annotations

import json
import logging
import textwrap
from typing import Any, Mapping, cast

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from compakt.core.interfaces.encoder import Encoder
from compakt.core.interfaces.summarizer import Summarizer
from compakt.core.models import CompaktChunk, DocumentStructure

logger = logging.getLogger(__name__)

PROMPT = textwrap.dedent(
    """
        You are a document summarization agent.

        Your task is to summarize a set of retrieved documents that share a hierarchical structure
        (e.g., title → sections → subsections). The goal is to produce a clear,
        concise summary that preserves the logical organization of the source material.

        Instructions:
        1. Read all provided documents carefully.
        2. Identify the overall document title, purpose, and main sections.
        3. Summarize the content while preserving the document hierarchy.
        4. Capture the key ideas, important facts, and relationships between sections.
        5. Remove redundancy and avoid unnecessary details.
        6. Do NOT invent information that is not present in the documents.
        7. If multiple documents contribute to the same section, synthesize them into a single coherent summary.
        8. Keep summaries concise and factual.

        Output Format (must have these sections in the output):
        # <Document Title>
        ## Overview: A short 4-6 sentence explanation of the document's purpose, scope, and main topic.
        ### <Section Title> A bullet-point list of the main sections and subsections, preserving the hierarchy.
        The length of each section summary should vary based on their importance (that you determine)
        but should generally be around 6-8 sentences for main sections and 4-6 sentences for subsections.

        Guidelines:
        - Maintain the original section hierarchy when possible.
        - Use bullet points for sections and subsections.
        - Each section summary should be **6-8 sentences maximum**.
        - Each subsection summary should be **4-6 sentences**.
        - Focus on meaning, not wording from the source.
        - Avoid quotes unless absolutely necessary.
        """
)


class OpenAISummarizer(Summarizer):
    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        encoder: Encoder | None = None,
        max_input_tokens: int = 120_000,
        timeout: float | None = 120.0,
        **kwargs: Any,
    ) -> None:
        self._client: BaseChatModel = ChatOpenAI(model=model, timeout=timeout, **kwargs)
        self._encoder = encoder
        self._max_input_tokens = max_input_tokens

    def _count_tokens(self, text: str) -> int | None:
        if self._encoder is None:
            return None
        return len(self._encoder.encode(text))

    def _prepare_user_content(
        self,
        relevant_docs: dict[str, list[CompaktChunk]],
        doc_structure: DocumentStructure | None,
        level: int,
    ) -> str:
        level_to_prompt = {
            1: "High-level",
            2: "Mid-level",
            3: "Low-level",
        }
        condensed_docs = {
            title: [chunk.content for chunk in chunks] for title, chunks in relevant_docs.items()
        }
        # Check if there are no content within the condensed docs and raise error if true
        if all(
            not any(content.strip() for content in chunks) for chunks in condensed_docs.values()
        ):
            raise ValueError("No content to summarize in the relevant documents")

        doc_tree = doc_structure.get_document_tree(level) if doc_structure else None
        level_label = level_to_prompt.get(level, "Mid-level")

        condensed_docs = self._truncate_if_needed(condensed_docs, doc_tree, level_label)

        user_content = textwrap.dedent(
            f"""Relevant documents: {json.dumps(condensed_docs, ensure_ascii=False)}
            ---
            Document structure to be summarized: {doc_tree}
            ---
            The summary should be a {level_label} summary based on the document structure provided.
            """
        )

        return user_content

    def _validate_response(self, content: Any) -> str:
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            content_items = cast(list[object], content)
            normalized_parts: list[str] = []
            for item in content_items:
                if isinstance(item, str):
                    normalized_parts.append(item)
                elif isinstance(item, Mapping):
                    mapping_item = cast(Mapping[str, object], item)
                    text_value = mapping_item.get("text")
                    if isinstance(text_value, str):
                        normalized_parts.append(text_value)

            if normalized_parts:
                return "\n".join(normalized_parts)

        msg = "Unable to normalize summary response content"
        raise TypeError(msg)

    def summarize(
        self,
        relevant_docs: dict[str, list[CompaktChunk]],
        doc_structure: DocumentStructure | None,
        level: int = 3,
    ) -> str:
        user_content = self._prepare_user_content(relevant_docs, doc_structure, level)

        response = self._client.invoke(
            [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": user_content},
            ]
        )

        content = cast(Any, response).content
        return self._validate_response(content)

    async def asummarize(
        self,
        relevant_docs: dict[str, list[CompaktChunk]],
        doc_structure: DocumentStructure | None,
        level: int = 3,
    ) -> str:
        user_content = self._prepare_user_content(relevant_docs, doc_structure, level)

        response = await self._client.ainvoke(
            [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": user_content},
            ]
        )

        content = cast(Any, response).content
        return self._validate_response(content)

    def summarize_unstructured(
        self,
        markdown: str,
        level: int = 3,
    ) -> str:
        user_content = textwrap.dedent(
            f"""Summarize the following unstructured markdown document into a concise summary.
            The summary should be a {level}-level summary based on the content of the markdown.

            Markdown content:
            {markdown}
            """
        )

        response = self._client.invoke(
            [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": user_content},
            ]
        )

        content = cast(Any, response).content
        return self._validate_response(content)

    async def asummarize_unstructured(
        self,
        markdown: str,
        level: int = 3,
    ) -> str:
        user_content = textwrap.dedent(
            f"""Summarize the following unstructured markdown document into a concise summary.
            The summary should be a {level}-level summary based on the content of the markdown.

            Markdown content:
            {markdown}
            """
        )

        response = await self._client.ainvoke(
            [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": user_content},
            ]
        )

        content = cast(Any, response).content
        return self._validate_response(content)

    def _truncate_if_needed(
        self,
        condensed_docs: dict[str, list[str]],
        doc_tree: str | None,
        level_label: str,
    ) -> dict[str, list[str]]:
        """Count prompt tokens and truncate condensed_docs if over the limit."""
        if self._encoder is None:
            return condensed_docs

        # Build the full prompt text for token counting.
        def _build_user_msg(docs: dict[str, list[str]]) -> str:
            return f"Relevant documents: {json.dumps(docs, ensure_ascii=False)}\n---\nDocument structure to be summarized: {doc_tree}\n---\nThe summary should be a {level_label} summary based on the document structure provided."

        full_text = PROMPT + _build_user_msg(condensed_docs)
        token_count = len(self._encoder.encode(full_text))
        logger.info("Summarizer prompt token count: %d", token_count)

        if token_count <= self._max_input_tokens:
            return condensed_docs

        logger.warning(
            "Summarizer prompt (%d tokens) exceeds limit (%d). Truncating.",
            token_count,
            self._max_input_tokens,
        )

        # Truncate by removing content entries from the longest lists first.
        docs = {k: list(v) for k, v in condensed_docs.items()}
        while token_count > self._max_input_tokens:
            longest_key = max(docs, key=lambda k: sum(len(c) for c in docs[k]))
            if not docs[longest_key]:
                break
            docs[longest_key].pop()
            full_text = PROMPT + _build_user_msg(docs)
            token_count = len(self._encoder.encode(full_text))

        logger.info("Truncated summarizer prompt to %d tokens", token_count)
        return docs

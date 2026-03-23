"""Document processing functions used by the worker.

Uses docling for unified multi-format document extraction. This module is only
imported by the worker process (which runs as __main__), never by the Azure
Function (where docling's torch dependency would crash).
"""

import logging
import os
import tempfile
from pathlib import Path

LOGGER = logging.getLogger("backend.processing")


def fallback_summary(blob_name: str, content: bytes) -> str:
    preview = content[:400].decode("utf-8", errors="replace")
    return (
        f"# Summary for {blob_name}\n\n"
        f"Compakt summarization was unavailable, so this is a fallback summary.\n\n"
        f"- Blob name: `{blob_name}`\n"
        f"- Blob size: `{len(content)}` bytes\n"
        f"- Preview:\n\n"
        f"```text\n{preview}\n```\n"
    )


def limit_markdown_tokens(markdown: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""

    import tiktoken

    encoding_name = os.getenv("WORKER_PREVIEW_ENCODING", "cl100k_base")
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(markdown)
    if len(tokens) <= max_tokens:
        return markdown

    trimmed = encoding.decode(tokens[:max_tokens])
    return (
        trimmed
        + "\n\n---\n\n"
        + f"[Preview truncated to {max_tokens} tokens for temporary display.]"
    )


def extract_markdown(blob_name: str, content: bytes) -> str:
    suffix = Path(blob_name).suffix.lower()

    if suffix == ".pdf":
        temp_path = ""
        try:
            import pymupdf4llm

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                temp_path = tmp_file.name
                tmp_file.write(content)
            extracted = str(pymupdf4llm.to_markdown(temp_path))
            return extracted
        except Exception:
            LOGGER.exception("Failed to extract PDF markdown for blob=%s", blob_name)
            return fallback_summary(blob_name, content)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    if suffix in {".md", ".markdown", ".txt"}:
        return content.decode("utf-8", errors="replace")

    return fallback_summary(blob_name, content)


def summarize_with_compakt(blob_name: str, markdown_content: str) -> tuple[str, str]:
    suffix = Path(blob_name).suffix.lower() or ".md"
    if suffix not in {".pdf", ".md", ".markdown", ".txt"}:
        LOGGER.debug(
            "Unsupported source extension for Compakt; using fallback summary",
        )
        return markdown_content, "fallback"

    if not os.getenv("OPENAI_API_KEY"):
        LOGGER.debug("OPENAI_API_KEY missing; using fallback summary")
        return markdown_content, "fallback"

    try:
        from compakt import Compakt

        LOGGER.debug("Running Compakt summarization for blob=%s", blob_name)
        result = Compakt(skip_file_reader=True).summarize_text(
            markdown_content, level=2, retrieval_k=20
        )
        return result.summary, "compakt"
    except Exception:
        LOGGER.exception("Compakt summarization failed for blob=%s; using fallback", blob_name)
        return markdown_content, "fallback"

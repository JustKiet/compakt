"""Extraction worker — standalone HTTP server for document extraction using docling.

Only handles the heavy docling extraction step. Returns extracted markdown text.
The Azure Function handles the rest (compakt summarization, status updates).

Run with: uv run uvicorn backend.worker:app --port 8001
"""

import logging
import os
import tempfile

from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from backend.pipeline_state import download_source_blob
from backend.processing import fallback_summary

load_dotenv()

converter = DocumentConverter()

LOGGER = logging.getLogger("backend.worker")
logging.basicConfig(
    level=getattr(logging, os.getenv("WORKER_LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def extract_markdown_docling(blob_name: str, content: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(content)
        tmp_file.flush()
    try:
        extracted = converter.convert(tmp_file.name)
        md = extracted.document.export_to_markdown(page_break_placeholder="<--PAGE_BREAK-->")
        return md
    except Exception:
        LOGGER.exception("Docling extraction failed for blob=%s", blob_name)
        return fallback_summary(blob_name, content)
    finally:
        if tmp_file and os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)


app = FastAPI()


class ExtractPayload(BaseModel):
    blob_name: str


class ExtractResponse(BaseModel):
    blob_name: str
    markdown: str


@app.post("/extract", response_model=ExtractResponse)
async def extract(payload: ExtractPayload):
    """Download blob and extract markdown using docling."""
    LOGGER.info("Extracting blob=%s", payload.blob_name)
    content = download_source_blob(payload.blob_name)
    LOGGER.debug("Downloaded blob=%s bytes=%d", payload.blob_name, len(content))
    markdown = extract_markdown_docling(blob_name=payload.blob_name, content=content)
    LOGGER.info("Extraction complete blob=%s chars=%d", payload.blob_name, len(markdown))
    return ExtractResponse(blob_name=payload.blob_name, markdown=markdown)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)

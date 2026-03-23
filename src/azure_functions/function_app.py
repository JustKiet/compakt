"""Azure Function: queue-triggered blob summarization.

Receives messages from the "uploaded-blobs" queue, calls the extraction worker
for docling-based document extraction, then runs compakt summarization and
pushes status updates to the backend via webhook.
"""

import json
import logging
import os
import site
import sys
from pathlib import Path
from typing import Any

import azure.functions as func
import httpx

from backend.pipeline_state import (
    summary_blob_name_for,
    upload_summary_blob,
    write_summary_status,
)
from backend.processing import limit_markdown_tokens, summarize_with_compakt

# Add the project venv so dependencies are importable in the Azure Functions runtime.
_venv_sp = str(
    Path(__file__).resolve().parents[2] / ".venv" / "lib" / "python3.13" / "site-packages"
)
if os.path.isdir(_venv_sp) and _venv_sp not in sys.path:
    site.addsitedir(_venv_sp)

# Add src/ so backend.* is importable.
_src_dir = str(Path(__file__).resolve().parents[1])
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

LOGGER = logging.getLogger("azure_functions.summarize")

WORKER_URL = os.getenv("WORKER_URL", "http://localhost:8001/extract")
BACKEND_WEBHOOK_URL = os.getenv("BACKEND_WEBHOOK_URL", "http://localhost:8000/webhook/status")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "dev-secret")

app = func.FunctionApp()


def _notify_backend(
    blob_name: str,
    job_id: str,
    status: str,
    summary_blob_name: str | None = None,
    error: str | None = None,
    processor: str | None = None,
) -> None:
    """Call the FastAPI webhook to push a status update to WebSocket clients."""
    payload: dict[str, Any] = {
        "blob_name": blob_name,
        "job_id": job_id,
        "status": status,
        "summary_blob_name": summary_blob_name,
        "error": error,
        "processor": processor,
    }
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.post(
                BACKEND_WEBHOOK_URL,
                json=payload,
                headers={"X-Webhook-Secret": WEBHOOK_SECRET},
            )
            resp.raise_for_status()
    except Exception:
        LOGGER.warning(
            "Failed to notify backend webhook for blob=%s status=%s",
            blob_name,
            status,
        )


@app.queue_trigger(
    arg_name="msg",
    queue_name="uploaded-blobs",
    connection="AzureWebJobsStorage",
)
def summarize_blob(msg: func.QueueMessage) -> None:
    """Queue-triggered function: extract via worker, then summarize with compakt."""
    raw = msg.get_body().decode("utf-8")
    payload = json.loads(raw)
    blob_name = payload["blob_name"]
    job_id = payload["job_id"]

    LOGGER.info("Processing blob=%s job_id=%s", blob_name, job_id)

    try:
        # 1. Call worker for docling extraction
        with httpx.Client(timeout=300) as client:
            resp = client.post(WORKER_URL, json={"blob_name": blob_name})
            resp.raise_for_status()
            extracted_markdown = resp.json()["markdown"]
        LOGGER.info("Extraction complete blob=%s chars=%d", blob_name, len(extracted_markdown))

        # 2. Upload preview and mark processing
        summary_blob = summary_blob_name_for(blob_name)
        preview_max_tokens = int(os.getenv("WORKER_PREVIEW_MAX_TOKENS", "20000"))
        preview = limit_markdown_tokens(extracted_markdown, preview_max_tokens)
        upload_summary_blob(summary_blob, preview)
        write_summary_status(
            blob_name=blob_name,
            job_id=job_id,
            status="processing",
            summary_blob_name=summary_blob,
            processor="preview_markdown",
        )
        _notify_backend(
            blob_name,
            job_id,
            "processing",
            summary_blob_name=summary_blob,
            processor="preview_markdown",
        )

        # 3. Run compakt summarization (no docling needed — uses skip_file_reader)
        summary_markdown, processor = summarize_with_compakt(blob_name, extracted_markdown)
        upload_summary_blob(summary_blob, summary_markdown)

        # 4. Mark completed
        write_summary_status(
            blob_name=blob_name,
            job_id=job_id,
            status="completed",
            summary_blob_name=summary_blob,
            processor=processor,
        )
        _notify_backend(
            blob_name,
            job_id,
            "completed",
            summary_blob_name=summary_blob,
            processor=processor,
        )
        LOGGER.info(
            "Completed job blob=%s job_id=%s processor=%s",
            blob_name,
            job_id,
            processor,
        )

    except Exception as exc:
        LOGGER.exception("Job failed blob=%s job_id=%s", blob_name, job_id)
        write_summary_status(
            blob_name=blob_name,
            job_id=job_id,
            status="failed",
            error=str(exc),
        )
        _notify_backend(blob_name, job_id, "failed", error=str(exc))

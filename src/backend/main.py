import json
import os

from azure.storage.blob import BlobSasPermissions
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from backend.pipeline_state import (
    QUEUE_NAME,
    blob_exists,
    ensure_container_exists,
    generate_blob_sas_url,
    generate_job_id,
    get_queue_client,
    list_summary_status_history,
    read_summary_blob_text,
    read_summary_status,
    summary_blob_exists,
    write_summary_status,
)
from backend.validation import validate_blob_name, validate_job_id
from backend.ws_manager import manager

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "dev-secret")

app = FastAPI()


@app.exception_handler(ValueError)
async def value_error_handler(request: object, exc: ValueError) -> object:
    raise HTTPException(status_code=400, detail=str(exc))


class UploadCallbackPayload(BaseModel):
    blob_name: str
    job_id: str | None = None


class SummaryStatusResponse(BaseModel):
    blob_name: str
    job_id: str | None = None
    status: str
    summary_blob_name: str | None = None
    summary_blob_url_with_sas: str | None = None
    source_blob_exists: bool
    error: str | None = None
    processor: str | None = None
    timestamps: dict[str, str] | None = None
    updated_at: str | None = None
    history: list[dict[str, str | None]] | None = None


class SummaryContentResponse(BaseModel):
    blob_name: str
    job_id: str | None = None
    status: str
    summary_blob_name: str | None = None
    summary_markdown: str | None = None
    error: str | None = None


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/get-sas-token")
async def get_sas_token(blob_name: str):
    validate_blob_name(blob_name)
    ensure_container_exists()
    sas_token, blob_url_with_sas = generate_blob_sas_url(
        blob_name=blob_name,
        permission=BlobSasPermissions(read=False, write=True, list=False),
    )
    return {"sas_token": sas_token, "blob_url_with_sas": blob_url_with_sas}


@app.post("/get-read-sas-token")
async def get_read_sas_token(blob_name: str):
    validate_blob_name(blob_name)
    ensure_container_exists()
    sas_token, blob_url_with_sas = generate_blob_sas_url(
        blob_name=blob_name,
        permission=BlobSasPermissions(read=True, write=False, list=False),
    )
    return {"sas_token": sas_token, "blob_url_with_sas": blob_url_with_sas}


@app.post("/upload-callback")
async def upload_callback(payload: UploadCallbackPayload):
    validate_blob_name(payload.blob_name)
    if payload.job_id:
        validate_job_id(payload.job_id)
    job_id = payload.job_id or generate_job_id()
    write_summary_status(payload.blob_name, job_id=job_id, status="queued")
    queue_client = get_queue_client()
    queue_client.send_message(json.dumps({"blob_name": payload.blob_name, "job_id": job_id}))
    await manager.broadcast_to_blob(
        payload.blob_name,
        {
            "blob_name": payload.blob_name,
            "job_id": job_id,
            "status": "queued",
        },
    )
    return {
        "status": "queued",
        "queue_name": QUEUE_NAME,
        "blob_name": payload.blob_name,
        "job_id": job_id,
    }


@app.get("/get-summary", response_model=SummaryStatusResponse)
async def get_summary(blob_name: str, job_id: str | None = None, include_history: bool = False):
    validate_blob_name(blob_name)
    if job_id:
        validate_job_id(job_id)
    status_record = read_summary_status(blob_name, job_id=job_id)
    source_exists = blob_exists(blob_name)

    if status_record is None:
        return SummaryStatusResponse(
            blob_name=blob_name,
            job_id=job_id,
            status="not_requested",
            source_blob_exists=source_exists,
        )

    summary_blob_name = status_record.get("summary_blob_name")
    summary_blob_url_with_sas: str | None = None
    if isinstance(summary_blob_name, str) and summary_blob_exists(summary_blob_name):
        _, summary_blob_url_with_sas = generate_blob_sas_url(
            blob_name=summary_blob_name,
            permission=BlobSasPermissions(read=True, write=False, list=False),
        )

    history: list[dict[str, str | None]] | None = None
    if include_history:
        history_records = list_summary_status_history(blob_name)
        history = [
            {
                "job_id": str(record.get("job_id")) if record.get("job_id") else None,
                "status": str(record.get("status")) if record.get("status") else None,
                "updated_at": str(record.get("updated_at")) if record.get("updated_at") else None,
            }
            for record in history_records
        ]

    return SummaryStatusResponse(
        blob_name=blob_name,
        job_id=status_record.get("job_id"),
        status=status_record.get("status", "unknown"),
        summary_blob_name=summary_blob_name,
        summary_blob_url_with_sas=summary_blob_url_with_sas,
        source_blob_exists=source_exists,
        error=status_record.get("error"),
        processor=status_record.get("processor"),
        timestamps=status_record.get("timestamps"),
        updated_at=status_record.get("updated_at"),
        history=history,
    )


@app.get("/get-summary-content", response_model=SummaryContentResponse)
async def get_summary_content(blob_name: str, job_id: str | None = None):
    validate_blob_name(blob_name)
    if job_id:
        validate_job_id(job_id)
    status_record = read_summary_status(blob_name, job_id=job_id)
    if status_record is None:
        return SummaryContentResponse(
            blob_name=blob_name,
            job_id=job_id,
            status="not_requested",
        )

    summary_blob_name = status_record.get("summary_blob_name")
    summary_markdown: str | None = None
    if isinstance(summary_blob_name, str) and summary_blob_exists(summary_blob_name):
        summary_markdown = read_summary_blob_text(summary_blob_name)

    return SummaryContentResponse(
        blob_name=blob_name,
        job_id=status_record.get("job_id"),
        status=status_record.get("status", "unknown"),
        summary_blob_name=summary_blob_name,
        summary_markdown=summary_markdown,
        error=status_record.get("error"),
    )


class WebhookStatusPayload(BaseModel):
    blob_name: str
    job_id: str
    status: str
    summary_blob_name: str | None = None
    error: str | None = None
    processor: str | None = None


@app.post("/webhook/status")
async def webhook_status(payload: WebhookStatusPayload, request: Request):
    """Receive status updates from Azure Function and broadcast to WebSocket clients."""
    if request.headers.get("X-Webhook-Secret") != WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")
    message: dict[str, str | None] = {
        "job_id": payload.job_id,
        "status": payload.status,
        "summary_blob_name": payload.summary_blob_name,
        "error": payload.error,
        "processor": payload.processor,
    }
    await manager.broadcast_to_blob(payload.blob_name, message)
    return {"ok": True}


@app.websocket("/ws/summary/{blob_name}")
async def ws_summary(websocket: WebSocket, blob_name: str):
    """WebSocket endpoint for live summary status updates."""
    validate_blob_name(blob_name)
    await manager.connect(blob_name, websocket)
    try:
        status_record = read_summary_status(blob_name)
        if status_record:
            await websocket.send_json(
                {
                    "blob_name": blob_name,
                    "job_id": status_record.get("job_id"),
                    "status": status_record.get("status", "unknown"),
                    "summary_blob_name": status_record.get("summary_blob_name"),
                    "error": status_record.get("error"),
                    "processor": status_record.get("processor"),
                }
            )
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(blob_name, websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

import datetime
import json
import os
import uuid
from typing import Any, cast

from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.storage.blob import (
    BlobClient,
    BlobSasPermissions,
    BlobServiceClient,
    generate_blob_sas,
)
from azure.storage.queue import QueueServiceClient

ACCOUNT_NAME = "devstoreaccount1"
ACCOUNT_KEY = (
    "Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw=="
)
CONTAINER_NAME = "container"
QUEUE_NAME = os.getenv("QUEUE_NAME", "uploaded-blobs")
BLOB_ENDPOINT = os.getenv("BLOB_ENDPOINT", "http://127.0.0.1:10000")
QUEUE_ENDPOINT = os.getenv("QUEUE_ENDPOINT", "http://127.0.0.1:10001")
STORAGE_API_VERSION = os.getenv("AZURE_STORAGE_API_VERSION", "2021-12-02")
SUMMARY_PREFIX = "summaries"
SUMMARY_STATUS_PREFIX = "summary-status"

ALLOWED_TRANSITIONS: dict[str | None, set[str]] = {
    None: {"queued", "processing", "failed"},
    "queued": {"processing", "failed"},
    "processing": {"completed", "failed"},
    "completed": set(),
    "failed": set(),
}


def generate_job_id() -> str:
    return str(uuid.uuid4())


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def get_blob_service_client() -> BlobServiceClient:
    account_url = f"{BLOB_ENDPOINT}/{ACCOUNT_NAME}"
    return BlobServiceClient(
        account_url=account_url,
        credential=ACCOUNT_KEY,
        api_version=STORAGE_API_VERSION,
    )


def ensure_container_exists() -> None:
    container_client = get_blob_service_client().get_container_client(CONTAINER_NAME)
    try:
        container_client.create_container()
    except ResourceExistsError:
        pass


def get_queue_client():
    account_url = f"{QUEUE_ENDPOINT}/{ACCOUNT_NAME}"
    queue_service_client = QueueServiceClient(
        account_url=account_url,
        credential=ACCOUNT_KEY,
        api_version=STORAGE_API_VERSION,
    )
    queue_client = queue_service_client.get_queue_client(QUEUE_NAME)
    try:
        queue_client.create_queue()
    except ResourceExistsError:
        pass
    return queue_client


def source_blob_client(blob_name: str) -> BlobClient:
    return get_blob_service_client().get_blob_client(
        container=CONTAINER_NAME,
        blob=blob_name,
    )


def blob_exists(blob_name: str) -> bool:
    return source_blob_client(blob_name).exists()


def download_source_blob(blob_name: str) -> bytes:
    return source_blob_client(blob_name).download_blob().readall()


def source_blob_url(blob_name: str, sas_token: str | None = None) -> str:
    base = f"{BLOB_ENDPOINT}/{ACCOUNT_NAME}/{CONTAINER_NAME}/{blob_name}"
    if sas_token:
        return f"{base}?{sas_token}"
    return base


def generate_blob_sas_url(
    blob_name: str,
    permission: BlobSasPermissions,
) -> tuple[str, str]:
    start_time = datetime.datetime.now(datetime.timezone.utc)
    expiry_time = start_time + datetime.timedelta(days=1)
    sas_token = generate_blob_sas(
        account_name=ACCOUNT_NAME,
        container_name=CONTAINER_NAME,
        blob_name=blob_name,
        account_key=ACCOUNT_KEY,
        version=STORAGE_API_VERSION,
        permission=permission,
        start=start_time,
        expiry=expiry_time,
    )
    return sas_token, source_blob_url(blob_name, sas_token=sas_token)


def summary_blob_name_for(blob_name: str) -> str:
    return f"{SUMMARY_PREFIX}/{blob_name}.summary.md"


def status_blob_name_for(blob_name: str, job_id: str) -> str:
    return f"{SUMMARY_STATUS_PREFIX}/{blob_name}/{job_id}.json"


def latest_status_blob_name_for(blob_name: str) -> str:
    return f"{SUMMARY_STATUS_PREFIX}/{blob_name}/latest.json"


def upload_summary_blob(summary_blob_name: str, summary_markdown: str) -> None:
    ensure_container_exists()
    blob_client = source_blob_client(summary_blob_name)
    blob_client.upload_blob(
        summary_markdown.encode("utf-8"),
        overwrite=True,
        content_type="text/markdown",
    )


def read_summary_blob_text(summary_blob_name: str) -> str:
    raw = source_blob_client(summary_blob_name).download_blob().readall()
    return raw.decode("utf-8", errors="replace")


def summary_blob_exists(summary_blob_name: str) -> bool:
    return source_blob_client(summary_blob_name).exists()


def write_summary_status(
    blob_name: str,
    job_id: str,
    status: str,
    summary_blob_name: str | None = None,
    error: str | None = None,
    processor: str | None = None,
) -> dict[str, Any]:
    ensure_container_exists()
    existing = read_summary_status(blob_name=blob_name, job_id=job_id)
    previous_status = existing.get("status") if isinstance(existing, dict) else None
    _assert_valid_transition(previous_status, status)

    timestamps = (
        dict(existing.get("timestamps", {}))
        if isinstance(existing, dict) and isinstance(existing.get("timestamps"), dict)
        else {}
    )
    timestamps[f"{status}_at"] = now_iso()

    payload: dict[str, Any] = {
        "blob_name": blob_name,
        "job_id": job_id,
        "status": status,
        "summary_blob_name": summary_blob_name,
        "error": error,
        "processor": processor,
        "updated_at": now_iso(),
        "timestamps": timestamps,
    }
    status_blob_name = status_blob_name_for(blob_name, job_id)
    blob_client = source_blob_client(status_blob_name)
    blob_client.upload_blob(
        json.dumps(payload).encode("utf-8"),
        overwrite=True,
        content_type="application/json",
    )

    latest_blob_client = source_blob_client(latest_status_blob_name_for(blob_name))
    latest_blob_client.upload_blob(
        json.dumps({"job_id": job_id}).encode("utf-8"),
        overwrite=True,
        content_type="application/json",
    )
    return payload


def read_summary_status(blob_name: str, job_id: str | None = None) -> dict[str, Any] | None:
    resolved_job_id = job_id or _read_latest_job_id(blob_name)
    if not resolved_job_id:
        return None

    status_blob_name = status_blob_name_for(blob_name, resolved_job_id)
    blob_client = source_blob_client(status_blob_name)
    try:
        raw = blob_client.download_blob().readall()
    except ResourceNotFoundError:
        return None
    return json.loads(raw.decode("utf-8"))


def list_summary_status_history(blob_name: str, limit: int = 20) -> list[dict[str, Any]]:
    container_client = get_blob_service_client().get_container_client(CONTAINER_NAME)
    prefix = f"{SUMMARY_STATUS_PREFIX}/{blob_name}/"
    records: list[dict[str, Any]] = []

    for item in container_client.list_blobs(name_starts_with=prefix):
        if item.name.endswith("/latest.json"):
            continue
        try:
            raw = source_blob_client(item.name).download_blob().readall()
            record = json.loads(raw.decode("utf-8"))
            if isinstance(record, dict):
                normalized = cast(dict[str, Any], record)
                records.append(normalized)
        except Exception:
            continue

    records.sort(key=lambda r: str(r.get("updated_at", "")), reverse=True)
    return records[:limit]


def _read_latest_job_id(blob_name: str) -> str | None:
    blob_client = source_blob_client(latest_status_blob_name_for(blob_name))
    try:
        raw = blob_client.download_blob().readall()
    except ResourceNotFoundError:
        return None

    payload = json.loads(raw.decode("utf-8"))
    job_id = payload.get("job_id")
    if isinstance(job_id, str) and job_id:
        return job_id
    return None


def _assert_valid_transition(previous_status: str | None, next_status: str) -> None:
    allowed = ALLOWED_TRANSITIONS.get(previous_status)
    if allowed is None or next_status not in allowed:
        raise ValueError(f"Invalid status transition from {previous_status!r} to {next_status!r}")

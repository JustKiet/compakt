from __future__ import annotations

import re

_BLOB_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9._/ -]+$")
_MAX_BLOB_NAME_LENGTH = 200
_MAX_JOB_ID_LENGTH = 64
_JOB_ID_PATTERN = re.compile(r"^[a-zA-Z0-9._-]+$")


def validate_blob_name(blob_name: str) -> str:
    """Validate blob_name for safety. Raises ValueError on invalid input."""
    if not blob_name:
        raise ValueError("blob_name must not be empty")
    if len(blob_name) > _MAX_BLOB_NAME_LENGTH:
        raise ValueError(f"blob_name exceeds maximum length of {_MAX_BLOB_NAME_LENGTH}")
    if ".." in blob_name:
        raise ValueError("blob_name must not contain path traversal sequences")
    if blob_name.startswith("/"):
        raise ValueError("blob_name must not start with /")
    if "\x00" in blob_name:
        raise ValueError("blob_name must not contain null bytes")
    if not _BLOB_NAME_PATTERN.match(blob_name):
        raise ValueError(
            "blob_name contains invalid characters; "
            "allowed: alphanumeric, dots, underscores, hyphens, spaces, slashes"
        )
    return blob_name


def validate_job_id(job_id: str) -> str:
    """Validate job_id for safety. Raises ValueError on invalid input."""
    if not job_id:
        raise ValueError("job_id must not be empty")
    if len(job_id) > _MAX_JOB_ID_LENGTH:
        raise ValueError(f"job_id exceeds maximum length of {_MAX_JOB_ID_LENGTH}")
    if ".." in job_id or "/" in job_id:
        raise ValueError("job_id must not contain path traversal sequences")
    if "\x00" in job_id:
        raise ValueError("job_id must not contain null bytes")
    if not _JOB_ID_PATTERN.match(job_id):
        raise ValueError(
            "job_id contains invalid characters; allowed: alphanumeric, dots, underscores, hyphens"
        )
    return job_id

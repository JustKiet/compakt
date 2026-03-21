import logging
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import httpx
import pymupdf4llm

from compakt.core.interfaces.file_reader import FileReaderAsMarkdown

logger = logging.getLogger(__name__)


class PyMuPDFMarkdownFileReader(FileReaderAsMarkdown):
    def read(self, file_path: str) -> str:
        if _is_url(file_path):
            return self._read_from_url(file_path)

        path = Path(file_path)
        if not path.exists():
            msg = f"File does not exist: {file_path}"
            raise FileNotFoundError(msg)

        if path.suffix.lower() in {".md", ".markdown"}:
            text = path.read_text(encoding="utf-8")
            logger.info("Read markdown file: %s (%d chars)", file_path, len(text))
            return text

        result = str(pymupdf4llm.to_markdown(str(path)))  # type: ignore[unknown-args]
        logger.info("Read PDF file: %s (%d chars markdown)", file_path, len(result))
        return result

    def _read_from_url(self, file_url: str) -> str:
        _validate_allowed_url(file_url)
        response = httpx.get(file_url, timeout=60.0)
        response.raise_for_status()

        parsed = urlparse(file_url)
        suffix = Path(parsed.path).suffix.lower()

        if suffix in {".md", ".markdown", ".txt"}:
            return response.content.decode("utf-8", errors="replace")

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".pdf") as f:
            temp_path = f.name
            f.write(response.content)

        try:
            return str(pymupdf4llm.to_markdown(temp_path))  # type: ignore[unknown-args]
        finally:
            Path(temp_path).unlink(missing_ok=True)


def _is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _validate_allowed_url(file_url: str) -> None:
    parsed = urlparse(file_url)
    hostname = (parsed.hostname or "").lower()
    scheme = parsed.scheme.lower()

    if not hostname:
        raise ValueError(f"URL must include a valid hostname: {file_url}")

    # Default allowlist supports local Azurite and Azure Blob endpoint.
    allowed_hosts_raw = os.getenv(
        "COMPAKT_ALLOWED_URL_HOSTS",
        "localhost,127.0.0.1,devstoreaccount1.blob.core.windows.net",
    )
    allowed_hosts = [item.strip().lower() for item in allowed_hosts_raw.split(",") if item.strip()]

    if not _is_allowed_host(hostname, allowed_hosts):
        raise ValueError(f"URL host '{hostname}' is not in COMPAKT_ALLOWED_URL_HOSTS allowlist")

    # Keep local HTTP for Azurite; everything else should be HTTPS unless explicitly allowed.
    allow_non_https = os.getenv("COMPAKT_ALLOW_NON_HTTPS_URLS", "false").lower() == "true"
    is_local_host = hostname in {"localhost", "127.0.0.1"}
    if scheme != "https" and not is_local_host and not allow_non_https:
        raise ValueError("Non-HTTPS URL blocked. Set COMPAKT_ALLOW_NON_HTTPS_URLS=true to override.")


def _is_allowed_host(hostname: str, allowed_hosts: list[str]) -> bool:
    for allowed in allowed_hosts:
        if allowed.startswith("*."):
            suffix = allowed[1:]  # keep leading dot for suffix match
            if hostname.endswith(suffix):
                return True
        elif hostname == allowed:
            return True
    return False

import json
import os
import textwrap

import httpx
import websockets

STORAGE_API_VERSION = os.getenv("AZURE_STORAGE_API_VERSION", "2021-12-02")


async def get_sas_token(blob_name: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/get-sas-token",
            params={"blob_name": blob_name},
        )
        response.raise_for_status()
        data = response.json()
        return data["sas_token"], data["blob_url_with_sas"]


async def get_read_sas_token(blob_name: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/get-read-sas-token",
            params={"blob_name": blob_name},
        )
        response.raise_for_status()
        data = response.json()
        return data["sas_token"], data["blob_url_with_sas"]


async def publish_upload_callback(blob_name: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/upload-callback",
            json={"blob_name": blob_name},
        )
        response.raise_for_status()
        return response.json()


async def get_summary_status(blob_name: str, job_id: str | None = None):
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "http://localhost:8000/get-summary",
            params={"blob_name": blob_name, "job_id": job_id}
            if job_id
            else {"blob_name": blob_name},
        )
        response.raise_for_status()
        return response.json()


async def get_summary_content(blob_name: str, job_id: str | None = None):
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "http://localhost:8000/get-summary-content",
            params={"blob_name": blob_name, "job_id": job_id}
            if job_id
            else {"blob_name": blob_name},
        )
        response.raise_for_status()
        return response.json()


async def watch_summary_ws(blob_name: str):
    """Connect to WebSocket and stream status updates until completed/failed."""
    uri = f"ws://localhost:8000/ws/summary/{blob_name}"
    async with websockets.connect(uri) as ws:
        async for raw in ws:
            update = json.loads(raw)
            status = update.get("status")
            print(f"[WS] Status update: {update}")
            if status in ("completed", "failed"):
                return update


async def upload_blob(blob_url_with_sas: str, data: bytes):
    async with httpx.AsyncClient() as client:
        response = await client.put(
            blob_url_with_sas,
            content=data,
            headers={
                # Required by Azure Blob Storage for Put Blob operations.
                "x-ms-blob-type": "BlockBlob",
                "x-ms-version": STORAGE_API_VERSION,
                "Content-Type": "application/octet-stream",
            },
        )
        response.raise_for_status()
        return response.status_code


async def download_blob(blob_url_with_sas: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(
            blob_url_with_sas,
            headers={"x-ms-version": STORAGE_API_VERSION},
        )
        response.raise_for_status()
        return response.content


async def main():
    # Write some data to upload
    DATA_PATH = "data/Group_Project.pdf"
    with open(DATA_PATH, "rb") as f:
        data = f.read()

    blob_name = os.path.basename(DATA_PATH)

    # Get SAS token and blob URL
    sas_token, blob_url_with_sas = await get_sas_token(blob_name)
    print(f"SAS Token: {sas_token}")
    print(f"Blob URL with SAS: {blob_url_with_sas}")

    # Upload the blob using the SAS URL
    status_code = await upload_blob(blob_url_with_sas, data)
    print(f"Upload status code: {status_code}")

    # Notify backend that upload completed so the Azure Function can process it.
    callback_response = await publish_upload_callback(blob_name)
    print(f"Callback response: {callback_response}")
    job_id = callback_response.get("job_id")

    # Watch for live status updates via WebSocket (replaces polling).
    print("Connecting to WebSocket for live updates...")
    result = await watch_summary_ws(blob_name)

    if result and result.get("status") == "completed":
        # Download the summary blob via SAS URL.
        summary_status = await get_summary_status(blob_name, job_id=job_id)
        sas_url = summary_status.get("summary_blob_url_with_sas")
        if sas_url:
            import openai
            import tiktoken
            from dotenv import load_dotenv

            load_dotenv()

            client = openai.AsyncOpenAI()
            encoding = tiktoken.get_encoding("cl100k_base")

            summary_bytes = await download_blob(sas_url)
            summary_text = summary_bytes.decode("utf-8")
            print(f"Summary content (from blob): {summary_text}")

            # Limit to first 100000 tokens.
            tokens = encoding.encode(summary_text)
            tokens = tokens[:100000]
            summary_text = encoding.decode(tokens)

            response = await client.chat.completions.create(
                model="gpt-5.4-nano",
                messages=[
                    {
                        "role": "system",
                        "content": textwrap.dedent("""
                            You are a helpful assistant that summarizes documents.
                        """).strip(),
                    },
                    {
                        "role": "user",
                        "content": textwrap.dedent(f"""
                            Here's is the document content:
                            {summary_text}
                            ---
                            Provide a concise summary of the above document.
                        """).strip(),
                    },
                ],
            )
            print(f"Summary content (from OpenAI): {response.choices[0].message.content}")
        else:
            summary_content = await get_summary_content(blob_name, job_id=job_id)
            print(f"Summary content: {summary_content}")
    elif result:
        print(f"Summary generation failed with error: {result.get('error')}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

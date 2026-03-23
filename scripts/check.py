import argparse
import asyncio
import json

import httpx


def check_rest(blob_name: str) -> None:
    """One-shot REST status check (original behavior)."""
    latest = httpx.get(
        "http://localhost:8000/get-summary",
        params={"blob_name": blob_name, "include_history": "true"},
    )
    print("latest:", latest.status_code, latest.json())
    content = httpx.get(
        "http://localhost:8000/get-summary-content", params={"blob_name": blob_name}
    )
    payload = content.json()
    print(
        "content:",
        content.status_code,
        {
            k: payload.get(k)
            for k in ["blob_name", "job_id", "status", "summary_blob_name", "error"]
        },
    )
    print("summary_len:", len(payload.get("summary_markdown") or ""))
    print("summary_head:", (payload.get("summary_markdown") or "")[:100].replace("\n", " "))


async def watch_ws(blob_name: str) -> None:
    """Live WebSocket status watcher."""
    import websockets

    uri = f"ws://localhost:8000/ws/summary/{blob_name}"
    print(f"Connecting to {uri} ...")
    async with websockets.connect(uri) as ws:
        async for raw in ws:
            update = json.loads(raw)
            print(f"[WS] {update}")
            if update.get("status") in ("completed", "failed"):
                print("Terminal status reached, exiting.")
                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check summarization status.")
    parser.add_argument("--blob", default="Group_Project.pdf", help="Blob name to check.")
    parser.add_argument("--ws", action="store_true", help="Watch live updates via WebSocket.")
    args = parser.parse_args()

    if args.ws:
        asyncio.run(watch_ws(args.blob))
    else:
        check_rest(args.blob)

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from compakt import AsyncCompakt

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown"}


async def summarize_file(
    compakt: AsyncCompakt,
    file_path: Path,
    output_dir: Path,
    level: int,
    retrieval_k: int,
    include_artifacts: bool,
) -> tuple[Path, bool, str | None]:
    try:
        result = await compakt.summarize(
            file_path=str(file_path),
            level=level,
            retrieval_k=retrieval_k,
        )

        payload: dict[str, Any]
        if include_artifacts:
            payload = result.model_dump(mode="json")
        else:
            payload = {
                "summary": result.summary,
                "strategy": result.artifacts.strategy,
                "source_file": str(file_path),
            }

        output_path = output_dir / f"{file_path.stem}.json"
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return file_path, True, None
    except Exception as exc:
        return file_path, False, str(exc)


async def worker(
    semaphore: asyncio.Semaphore,
    compakt: AsyncCompakt,
    file_path: Path,
    output_dir: Path,
    level: int,
    retrieval_k: int,
    include_artifacts: bool,
) -> tuple[Path, bool, str | None]:
    async with semaphore:
        return await summarize_file(
            compakt=compakt,
            file_path=file_path,
            output_dir=output_dir,
            level=level,
            retrieval_k=retrieval_k,
            include_artifacts=include_artifacts,
        )


async def run(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        [
            path
            for path in data_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    )

    if not files:
        print(f"No supported files found in {data_dir}")
        return

    compakt = AsyncCompakt()
    semaphore = asyncio.Semaphore(max(1, args.concurrency))

    tasks = [
        worker(
            semaphore=semaphore,
            compakt=compakt,
            file_path=file_path,
            output_dir=output_dir,
            level=args.level,
            retrieval_k=args.retrieval_k,
            include_artifacts=args.include_artifacts,
        )
        for file_path in files
    ]

    results = await asyncio.gather(*tasks)

    success_count = 0
    for file_path, ok, error in results:
        if ok:
            success_count += 1
            print(f"[OK] {file_path.name} -> {file_path.stem}.json")
        else:
            print(f"[ERROR] {file_path.name}: {error}")

    print(
        f"Completed: {success_count}/{len(files)} files summarized to {output_dir.resolve()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize all supported files in data/ and save JSON outputs"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing source documents",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for generated JSON outputs",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="Summary retrieval level",
    )
    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=20,
        help="Top-k retrieval before elbow filtering",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Max number of files processed concurrently",
    )
    parser.add_argument(
        "--include-artifacts",
        action="store_true",
        help="Include full artifacts in output JSON (can be large)",
    )

    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

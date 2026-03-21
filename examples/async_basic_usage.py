from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from compakt import AsyncCompakt


async def run(args: argparse.Namespace) -> None:
    input_path = Path(args.file_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    compakt = AsyncCompakt()
    result = await compakt.summarize(
        file_path=str(input_path),
        level=args.level,
        retrieval_k=args.retrieval_k,
    )

    print("=== Summary ===")
    print(result.summary)
    print("\n=== Artifacts ===")
    print(f"Strategy: {result.artifacts.strategy}")
    print(f"Chunks: {len(result.artifacts.chunks)}")
    print(f"Embeddings: {len(result.artifacts.embeddings)}")
    print(f"Retrieved groups: {len(result.artifacts.retrieved_chunks)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AsyncCompakt summarization")
    parser.add_argument("file_path", help="Path to input file (PDF/Markdown)")
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

    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

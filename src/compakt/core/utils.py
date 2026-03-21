import logging
import re

import numpy as np

from compakt.core.models import CompaktChunk

logger = logging.getLogger(__name__)


def elbow_filter(
    data: list[tuple[CompaktChunk, float]],
    percentile: float = 80.0,
) -> list[tuple[CompaktChunk, float]]:
    """Filter similarity search results by keeping scores above a percentile cutoff.

    Args:
        data: Tuples of (CompaktChunk, relevance_score), assumed sorted by
              descending score from the vector index.
        percentile: The percentile of the score distribution to use as the
                    cutoff.  Items with scores >= this threshold are kept.
                    Lower values keep more results; higher values are stricter.
    Returns:
        Filtered list preserving the original order.  Always returns at least
        one item when *data* is non-empty.
    """
    if not data:
        return []

    if len(data) <= 2:
        return data

    scores = np.array([score for _, score in data])
    cutoff = float(np.percentile(scores, percentile))

    filtered = [(chunk, score) for chunk, score in data if score >= cutoff]

    if not filtered:
        # Guarantee at least the highest-scoring item is returned.
        filtered = [data[0]]

    logger.debug(
        "elbow_filter: %d -> %d items (percentile=%.1f, cutoff=%.4f)",
        len(data),
        len(filtered),
        percentile,
        cutoff,
    )
    return filtered


def normalize_markdown_title(value: str) -> str:
    """Normalize markdown-flavored heading text for reliable comparisons."""
    normalized = value

    # Images/links first to preserve their visible labels.
    normalized = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", normalized)
    normalized = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", normalized)

    # Remove code fences and inline code markers.
    normalized = normalized.replace("```", " ").replace("`", " ")

    # Strip common markdown emphasis markers.
    normalized = re.sub(r"(\*\*\*|___|\*\*|__|\*|_|~~)", "", normalized)

    # Remove HTML tags that may appear in headings.
    normalized = re.sub(r"<[^>]+>", " ", normalized)

    # Collapse whitespace and normalize case.
    return " ".join(normalized.casefold().split())

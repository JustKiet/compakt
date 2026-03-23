from compakt.client import AsyncCompakt, Compakt
from compakt.core.models import (
    CompaktRunArtifacts,
    CompaktRunResult,
    DocumentNode,
    DocumentStructure,
)


def __getattr__(name: str):
    if name == "Container":
        from compakt.containers import Container

        return Container
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Compakt",
    "AsyncCompakt",
    "Container",
    "CompaktRunArtifacts",
    "CompaktRunResult",
    "DocumentNode",
    "DocumentStructure",
]

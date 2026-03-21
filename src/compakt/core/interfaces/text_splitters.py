from typing import Protocol

from compakt.core.models import CompaktChunk


class TextSplitter(Protocol):
    def split(self, text: str) -> list[CompaktChunk]:
        """
        Splits the given text into smaller chunks.

        Args:
            text (str): The input text to be split.
        Returns:
            list[CompaktChunk]: A list of CompaktChunk objects representing the split chunks of the input text.
        """
        ...

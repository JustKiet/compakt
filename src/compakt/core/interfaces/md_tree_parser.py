from typing import Protocol

from compakt.core.models import HeaderNode


class MarkdownTreeParser(Protocol):
    def parse(self, markdown_text: str) -> list[HeaderNode]:
        """
        Parses the markdown text into a tree structure.

        Args:
            markdown_text (str): The markdown text to be parsed.
        Returns:
            list[MarkdownNode]: The parsed markdown tree structure.
        """
        ...

from typing import Protocol

from compakt.core.models import DocumentStructure, HeaderNode


class DocumentStructureResolver(Protocol):
    def resolve(self, headers: list[HeaderNode]) -> DocumentStructure:
        """
        Resolves the document structure from a list of header nodes.
        This method is required as the given headers nodes may be incomplete/incorrect and does not necessarily represent the actual document structure.

        Args:
            headers (list[HeaderNode]): The list of header nodes to be resolved.
        Returns:
            DocumentStructure: The resolved document structure.
        """
        ...

    async def aresolve(self, headers: list[HeaderNode]) -> DocumentStructure:
        """
        Asynchronously resolves the document structure from a list of header nodes.
        This method is required as the given headers nodes may be incomplete/incorrect and does not necessarily represent the actual document structure.

        Args:
            headers (list[HeaderNode]): The list of header nodes to be resolved.
        Returns:
            DocumentStructure: The resolved document structure.
        """
        ...

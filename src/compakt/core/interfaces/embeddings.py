from typing import Protocol, overload


class Embeddings(Protocol):
    @overload
    def embed(self, payload: str) -> list[float]:
        """
        Embeds the given text into a vector representation.

        Args:
            text (str): The text to be embedded.
        Returns:
            list[float]: The vector representation of the embedded text.
        """
        ...

    @overload
    def embed(self, payload: list[str]) -> list[list[float]]:
        """
        Embeds a list of texts into their vector representations.

        Args:
            texts (list[str]): The list of texts to be embedded.
        Returns:
            list[list[float]]: A list of vector representations for each embedded text.
        """
        ...

    def embed(self, payload: str | list[str]) -> list[float] | list[list[float]]:
        """
        Embeds the given text or list of texts into their vector representations.

        Args:
            payload (str | list[str]): The text or list of texts to be embedded.
        Returns:
            list[float] | list[list[float]]: The vector representation(s) of the embedded text(s).
        """
        ...

    @overload
    async def aembed(self, payload: str) -> list[float]:
        """
        Asynchronously embeds the given text into a vector representation.

        Args:
            text (str): The text to be embedded.
        Returns:
            list[float]: The vector representation of the embedded text.
        """
        ...

    @overload
    async def aembed(self, payload: list[str]) -> list[list[float]]:
        """
        Asynchronously embeds a list of texts into their vector representations.

        Args:
            texts (list[str]): The list of texts to be embedded.
        Returns:
            list[list[float]]: A list of vector representations for each embedded text.
        """
        ...

    async def aembed(self, payload: str | list[str]) -> list[float] | list[list[float]]:
        """
        Asynchronously embeds the given text or list of texts into their vector representations.

        Args:
            payload (str | list[str]): The text or list of texts to be embedded.
        Returns:
            list[float] | list[list[float]]: The vector representation(s) of the embedded text(s).
        """
        ...

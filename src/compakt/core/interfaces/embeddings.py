from enum import Enum
from typing import Protocol, Sequence, overload


class PayloadType(str, Enum):
    DOCUMENT = "document"
    QUERY = "query"


VectorLike = list[float] | list[int]


class Embeddings(Protocol):
    @overload
    def embed(self, payload: str, payload_type: PayloadType = PayloadType.DOCUMENT) -> VectorLike:
        """
        Embeds the given text into a vector representation.

        Args:
            text (str): The text to be embedded.
            payload_type (PayloadType): The type of the payload.
        Returns:
            VectorLike: The vector representation of the embedded text.
        """
        ...

    @overload
    def embed(
        self, payload: list[str], payload_type: PayloadType = PayloadType.DOCUMENT
    ) -> Sequence[VectorLike]:
        """
        Embeds a list of texts into their vector representations.

        Args:
            texts (list[str]): The list of texts to be embedded.
            payload_type (PayloadType): The type of the payload.
        Returns:
            Sequence[VectorLike]: A sequence of vector representations for each embedded text.
        """
        ...

    def embed(
        self, payload: str | list[str], payload_type: PayloadType = PayloadType.DOCUMENT
    ) -> VectorLike | Sequence[VectorLike]:
        """
        Embeds the given text or list of texts into their vector representations.

        Args:
            payload (str | list[str]): The text or list of texts to be embedded.
            payload_type (PayloadType): The type of the payload.
        Returns:
            VectorLike | Sequence[VectorLike]: The vector representation(s) of the embedded text(s).
        """
        ...

    @overload
    async def aembed(
        self, payload: str, payload_type: PayloadType = PayloadType.DOCUMENT
    ) -> VectorLike:
        """
        Asynchronously embeds the given text into a vector representation.

        Args:
            text (str): The text to be embedded.
            payload_type (PayloadType): The type of the payload.
        Returns:
            VectorLike: The vector representation of the embedded text.
        """
        ...

    @overload
    async def aembed(
        self, payload: list[str], payload_type: PayloadType = PayloadType.DOCUMENT
    ) -> Sequence[VectorLike]:
        """
        Asynchronously embeds a list of texts into their vector representations.

        Args:
            texts (list[str]): The list of texts to be embedded.
            payload_type (PayloadType): The type of the payload.
        Returns:
            Sequence[VectorLike]: A sequence of vector representations for each embedded text.
        """
        ...

    async def aembed(
        self, payload: str | list[str], payload_type: PayloadType = PayloadType.DOCUMENT
    ) -> VectorLike | Sequence[VectorLike]:
        """
        Asynchronously embeds the given text or list of texts into their vector representations.

        Args:
            payload (str | list[str]): The text or list of texts to be embedded.
            payload_type (PayloadType): The type of the payload.
        Returns:
            VectorLike | Sequence[VectorLike]: The vector representation(s) of the embedded text(s).
        """
        ...

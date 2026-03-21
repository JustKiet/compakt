from typing import Protocol


class Encoder(Protocol):
    def encode(self, text: str) -> list[int]:
        """
        Encodes a given text into a list of token IDs.

        Args:
            text (str): The input text to be encoded.
        Returns:
            list[int]: A list of token IDs representing the encoded text.
        """
        ...

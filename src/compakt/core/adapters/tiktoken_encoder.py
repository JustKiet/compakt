import tiktoken

from compakt.core.interfaces.encoder import Encoder


class TiktokenEncoder(Encoder):
    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self._encoding = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode(text)

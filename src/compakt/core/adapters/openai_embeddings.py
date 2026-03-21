from typing import Any, overload

from langchain_openai import OpenAIEmbeddings as LangchainOpenAIEmbeddings

from compakt.core.interfaces.embeddings import Embeddings


class OpenAIEmbeddings(Embeddings):
    def __init__(self, model: str = "text-embedding-3-small", **kwargs: Any):
        self._model = model
        self._kwargs = kwargs
        self._client = LangchainOpenAIEmbeddings(
            model=model,
            **kwargs,
        )

    @overload
    def embed(self, payload: str) -> list[float]: ...

    @overload
    def embed(self, payload: list[str]) -> list[list[float]]: ...

    def embed(self, payload: str | list[str]) -> list[float] | list[list[float]]:
        was_single_payload = isinstance(payload, str)
        request_payload = [payload] if was_single_payload else payload

        response = self._client.embed_documents(request_payload)

        if was_single_payload:
            return response[0]

        return response

    @overload
    async def aembed(self, payload: str) -> list[float]: ...

    @overload
    async def aembed(self, payload: list[str]) -> list[list[float]]: ...

    async def aembed(self, payload: str | list[str]) -> list[float] | list[list[float]]:
        was_single_payload = isinstance(payload, str)
        request_payload = [payload] if was_single_payload else payload

        response = await self._client.aembed_documents(request_payload)

        if was_single_payload:
            return response[0]

        return response

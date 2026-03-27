from typing import Any, Sequence, overload

from langchain_openai import OpenAIEmbeddings as LangchainOpenAIEmbeddings

from compakt.core.interfaces.embeddings import Embeddings, PayloadType, VectorLike


class OpenAIEmbeddings(Embeddings):
    def __init__(self, model: str = "text-embedding-3-small", **kwargs: Any):
        self._model = model
        self._kwargs = kwargs
        self._client = LangchainOpenAIEmbeddings(
            model=model,
            **kwargs,
        )

    @overload
    def embed(
        self, payload: str, payload_type: PayloadType = PayloadType.DOCUMENT
    ) -> VectorLike: ...

    @overload
    def embed(
        self, payload: list[str], payload_type: PayloadType = PayloadType.DOCUMENT
    ) -> Sequence[VectorLike]: ...

    def embed(
        self, payload: str | list[str], payload_type: PayloadType = PayloadType.DOCUMENT
    ) -> VectorLike | Sequence[VectorLike]:
        was_single_payload = isinstance(payload, str)
        request_payload = [payload] if was_single_payload else payload

        response = self._client.embed_documents(request_payload)

        if was_single_payload:
            return response[0]

        return response

    @overload
    async def aembed(
        self, payload: str, payload_type: PayloadType = PayloadType.DOCUMENT
    ) -> VectorLike: ...

    @overload
    async def aembed(
        self, payload: list[str], payload_type: PayloadType = PayloadType.DOCUMENT
    ) -> Sequence[VectorLike]: ...

    async def aembed(
        self, payload: str | list[str], payload_type: PayloadType = PayloadType.DOCUMENT
    ) -> VectorLike | Sequence[VectorLike]:
        was_single_payload = isinstance(payload, str)
        request_payload = [payload] if was_single_payload else payload

        response = await self._client.aembed_documents(request_payload)

        if was_single_payload:
            return response[0]

        return response

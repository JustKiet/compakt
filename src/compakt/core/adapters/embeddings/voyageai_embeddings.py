from typing import Sequence, overload

from voyageai.client import Client
from voyageai.client_async import AsyncClient

from compakt.core.interfaces.embeddings import Embeddings, PayloadType, VectorLike


class VoyageAIEmbeddings(Embeddings):
    def __init__(self, model: str):
        self.client = Client()
        self.async_client = AsyncClient()
        self.model = model

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

        response = self.client.embed(
            texts=request_payload,
            model=self.model,
            input_type=payload_type.value,
        )

        if was_single_payload:
            return response.embeddings[0]

        return response.embeddings

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

        response = await self.async_client.embed(
            texts=request_payload,
            model=self.model,
            input_type=payload_type.value,
        )

        if was_single_payload:
            return response.embeddings[0]

        return response.embeddings

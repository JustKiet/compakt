from __future__ import annotations

import asyncio
import logging

from markdown_it import MarkdownIt

from compakt.core.adapters.embeddings.openai_embeddings import OpenAIEmbeddings
from compakt.core.adapters.in_memory_vector_index import InMemoryVectorIndex
from compakt.core.adapters.md_it_tree_parser import MarkdownItTreeParser
from compakt.core.adapters.openai_document_structure_resolver import (
    OpenAIDocumentStructureResolver,
)
from compakt.core.adapters.openai_summarizer import OpenAISummarizer
from compakt.core.adapters.text_splitters.md_text_splitter import (
    LangchainMarkdownTextSplitter,
)
from compakt.core.adapters.tiktoken_encoder import TiktokenEncoder
from compakt.core.exceptions import (
    EmptyDocumentError,
    UnsupportedDocumentStrategyError,
)
from compakt.core.interfaces.encoder import Encoder
from compakt.core.interfaces.file_reader import FileReaderAsMarkdown
from compakt.core.interfaces.md_tree_parser import MarkdownTreeParser
from compakt.core.interfaces.strategy import SummarizationStrategy
from compakt.core.interfaces.text_splitters import TextSplitter
from compakt.core.interfaces.vector_index import VectorIndex
from compakt.core.models import CompaktRunResult, HeaderNode, MarkdownHeader
from compakt.strategies.brute_force_unstructured import BruteForceUnstructuredStrategy
from compakt.strategies.fallback_unstructured import FallbackUnstructuredStrategy
from compakt.strategies.structured_markdown import StructuredMarkdownStrategy

logger = logging.getLogger(__name__)


class Compakt:
    def __init__(
        self,
        brute_force_token_limit: int = 50_000,
        file_reader: FileReaderAsMarkdown | None = None,
        markdown_tree_parser: MarkdownTreeParser | None = None,
        text_splitter: TextSplitter | None = None,
        vector_index: VectorIndex | None = None,
        strategies: list[SummarizationStrategy] | None = None,
        brute_force_strategy: SummarizationStrategy | None = None,
        encoder: Encoder | None = None,
        chat_model: str = "gpt-4.1-mini",
        embedding_model: str = "text-embedding-3-small",
        encoding_name: str = "cl100k_base",
        skip_file_reader: bool = False,
    ) -> None:
        needs_defaults = (
            markdown_tree_parser is None
            or text_splitter is None
            or vector_index is None
            or strategies is None
            or encoder is None
            or brute_force_strategy is None
        )

        if needs_defaults and skip_file_reader:
            defaults = self.build_defaults_without_reader(
                chat_model=chat_model,
                embedding_model=embedding_model,
                encoding_name=encoding_name,
                token_limit=brute_force_token_limit,
            )
            markdown_tree_parser = defaults[0]
            text_splitter = defaults[1]
            vector_index = defaults[2]
            strategies = defaults[3]
            encoder = defaults[4]
            brute_force_strategy = defaults[5]
        elif needs_defaults or (file_reader is None and not skip_file_reader):
            (
                default_file_reader,
                markdown_tree_parser,
                text_splitter,
                vector_index,
                strategies,
                encoder,
                brute_force_strategy,
            ) = self.build_defaults(
                chat_model=chat_model,
                embedding_model=embedding_model,
                encoding_name=encoding_name,
                token_limit=brute_force_token_limit,
            )
            if file_reader is None:
                file_reader = default_file_reader
        else:
            (
                file_reader,
                markdown_tree_parser,
                text_splitter,
                vector_index,
                strategies,
                encoder,
                brute_force_strategy,
            ) = self.build_defaults(
                chat_model=chat_model,
                embedding_model=embedding_model,
                encoding_name=encoding_name,
                token_limit=brute_force_token_limit,
            )

        self._brute_force_token_limit = brute_force_token_limit

        self._file_reader = file_reader
        self._markdown_tree_parser = markdown_tree_parser
        self._text_splitter = text_splitter
        self._vector_index = vector_index
        self._strategies = strategies
        self._encoder = encoder
        self._brute_force_strategy = brute_force_strategy

    def create_tree(self, markdown: str) -> list[HeaderNode]:
        markdown_tree = self._markdown_tree_parser.parse(markdown)
        return markdown_tree

    def summarize_text(
        self, markdown: str, level: int = 2, retrieval_k: int = 20
    ) -> CompaktRunResult:
        """Summarize pre-extracted markdown text (skips file reading)."""
        logger.info("Text summarization started (level=%d, k=%d)", level, retrieval_k)
        if not markdown.strip():
            raise EmptyDocumentError("The provided text is empty")
        return self._summarize_markdown(markdown, level=level, retrieval_k=retrieval_k)

    def summarize(self, file_path: str, level: int = 2, retrieval_k: int = 20) -> CompaktRunResult:
        logger.info("Summarization started: %s (level=%d, k=%d)", file_path, level, retrieval_k)
        if not self._file_reader:
            raise ValueError("File reader is required for summarize() but was not provided")
        markdown = self._file_reader.read(file_path)
        if not markdown.strip():
            raise EmptyDocumentError("The document is empty after markdown conversion")
        return self._summarize_markdown(markdown, level=level, retrieval_k=retrieval_k)

    def _summarize_markdown(self, markdown: str, level: int, retrieval_k: int) -> CompaktRunResult:
        try:
            if self.count_tokens(markdown) <= self._brute_force_token_limit:
                logger.info(
                    "Document is within brute-force token limit, using brute-force strategy"
                )

                result = self._brute_force_strategy.run(
                    markdown=markdown,
                    markdown_tree=[],
                    chunks=[],
                    embeddings=[],
                    level=level,
                    retrieval_k=retrieval_k,
                )
                logger.info("Summarization completed with brute-force strategy")
                return result

            markdown_tree = self._markdown_tree_parser.parse(markdown)
            chunks = self._text_splitter.split(markdown)
            if not chunks:
                raise EmptyDocumentError("No chunks were produced from the document")
            logger.debug("Produced %d chunks", len(chunks))

            embeddings = self._vector_index.index(chunks)
            logger.debug("Indexed %d embeddings", len(embeddings))

            for strategy in self._strategies:
                if strategy.can_handle(markdown, markdown_tree):
                    logger.info("Selected strategy: %s", strategy.name)
                    result = strategy.run(
                        markdown=markdown,
                        markdown_tree=markdown_tree,
                        chunks=chunks,
                        embeddings=embeddings,
                        level=level,
                        retrieval_k=retrieval_k,
                    )
                    logger.info("Summarization completed")
                    return result

            raise UnsupportedDocumentStrategyError(
                "No summarization strategy can handle this document"
            )
        finally:
            self._vector_index.clear()

    def count_tokens(self, text: str) -> int:
        return len(self._encoder.encode(text))

    @staticmethod
    def build_defaults(
        chat_model: str,
        embedding_model: str,
        encoding_name: str,
        token_limit: int = 50_000,
    ) -> tuple[
        FileReaderAsMarkdown,
        MarkdownTreeParser,
        TextSplitter,
        VectorIndex,
        list[SummarizationStrategy],
        Encoder,
        BruteForceUnstructuredStrategy,
    ]:
        from docling.document_converter import DocumentConverter

        from compakt.core.adapters.readers.docling_reader import DoclingFileReader

        markdown_it = MarkdownIt()
        document_converter = DocumentConverter()
        file_reader = DoclingFileReader(document_converter=document_converter)
        markdown_tree_parser = MarkdownItTreeParser(markdown_it)
        text_splitter = LangchainMarkdownTextSplitter(
            headers_to_split_on=[
                (MarkdownHeader.H1, "header_1"),
                (MarkdownHeader.H2, "header_2"),
                (MarkdownHeader.H3, "header_3"),
                (MarkdownHeader.H4, "header_4"),
            ]
        )

        embeddings = OpenAIEmbeddings(model=embedding_model)
        vector_index = InMemoryVectorIndex(embeddings)
        encoder = TiktokenEncoder(encoding_name=encoding_name)

        document_structure_resolver = OpenAIDocumentStructureResolver(
            model=chat_model, encoder=encoder
        )
        summarizer = OpenAISummarizer(model=chat_model, encoder=encoder)

        strategies: list[SummarizationStrategy] = [
            StructuredMarkdownStrategy(
                document_structure_resolver=document_structure_resolver,
                summarizer=summarizer,
                vector_index=vector_index,
            ),
            FallbackUnstructuredStrategy(
                summarizer=summarizer,
                vector_index=vector_index,
            ),
        ]

        brute_force_strategy = BruteForceUnstructuredStrategy(
            summarizer=summarizer,
            encoder=encoder,
            token_limit=token_limit,
        )

        return (
            file_reader,
            markdown_tree_parser,
            text_splitter,
            vector_index,
            strategies,
            encoder,
            brute_force_strategy,
        )

    @staticmethod
    def build_defaults_without_reader(
        chat_model: str = "gpt-4.1-mini",
        embedding_model: str = "text-embedding-3-small",
        encoding_name: str = "cl100k_base",
        token_limit: int = 50_000,
    ) -> tuple[
        MarkdownTreeParser,
        TextSplitter,
        VectorIndex,
        list[SummarizationStrategy],
        Encoder,
        BruteForceUnstructuredStrategy,
    ]:
        """Build default components without the file reader (no docling dependency)."""
        markdown_it = MarkdownIt()
        markdown_tree_parser = MarkdownItTreeParser(markdown_it)
        text_splitter = LangchainMarkdownTextSplitter(
            headers_to_split_on=[
                (MarkdownHeader.H1, "header_1"),
                (MarkdownHeader.H2, "header_2"),
                (MarkdownHeader.H3, "header_3"),
                (MarkdownHeader.H4, "header_4"),
            ]
        )

        embeddings = OpenAIEmbeddings(model=embedding_model)
        vector_index = InMemoryVectorIndex(embeddings)
        encoder = TiktokenEncoder(encoding_name=encoding_name)

        document_structure_resolver = OpenAIDocumentStructureResolver(
            model=chat_model, encoder=encoder
        )
        summarizer = OpenAISummarizer(model=chat_model, encoder=encoder)

        strategies: list[SummarizationStrategy] = [
            StructuredMarkdownStrategy(
                document_structure_resolver=document_structure_resolver,
                summarizer=summarizer,
                vector_index=vector_index,
            ),
            FallbackUnstructuredStrategy(
                summarizer=summarizer,
                vector_index=vector_index,
            ),
        ]

        brute_force_strategy = BruteForceUnstructuredStrategy(
            summarizer=summarizer,
            encoder=encoder,
            token_limit=token_limit,
        )

        return (
            markdown_tree_parser,
            text_splitter,
            vector_index,
            strategies,
            encoder,
            brute_force_strategy,
        )


CompaktClient = Compakt


class AsyncCompakt:
    def __init__(
        self,
        brute_force_token_limit: int = 50_000,
        file_reader: FileReaderAsMarkdown | None = None,
        markdown_tree_parser: MarkdownTreeParser | None = None,
        text_splitter: TextSplitter | None = None,
        vector_index: VectorIndex | None = None,
        strategies: list[SummarizationStrategy] | None = None,
        brute_force_strategy: SummarizationStrategy | None = None,
        encoder: Encoder | None = None,
        chat_model: str = "gpt-4.1-mini",
        embedding_model: str = "text-embedding-3-small",
        encoding_name: str = "cl100k_base",
        skip_file_reader: bool = False,
    ) -> None:
        needs_defaults = (
            markdown_tree_parser is None
            or text_splitter is None
            or vector_index is None
            or strategies is None
            or encoder is None
            or brute_force_strategy is None
        )

        if needs_defaults and skip_file_reader:
            defaults = Compakt.build_defaults_without_reader(
                chat_model=chat_model,
                embedding_model=embedding_model,
                encoding_name=encoding_name,
                token_limit=brute_force_token_limit,
            )
            markdown_tree_parser = defaults[0]
            text_splitter = defaults[1]
            vector_index = defaults[2]
            strategies = defaults[3]
            encoder = defaults[4]
            brute_force_strategy = defaults[5]
        elif needs_defaults or (file_reader is None and not skip_file_reader):
            (
                default_file_reader,
                markdown_tree_parser,
                text_splitter,
                vector_index,
                strategies,
                encoder,
                brute_force_strategy,
            ) = Compakt.build_defaults(
                chat_model=chat_model,
                embedding_model=embedding_model,
                encoding_name=encoding_name,
                token_limit=brute_force_token_limit,
            )
            if file_reader is None:
                file_reader = default_file_reader
        else:
            (
                file_reader,
                markdown_tree_parser,
                text_splitter,
                vector_index,
                strategies,
                encoder,
                brute_force_strategy,
            ) = Compakt.build_defaults(
                chat_model=chat_model,
                embedding_model=embedding_model,
                encoding_name=encoding_name,
                token_limit=brute_force_token_limit,
            )

        self._file_reader = file_reader
        self._markdown_tree_parser = markdown_tree_parser
        self._text_splitter = text_splitter
        self._vector_index = vector_index
        self._strategies = strategies
        self._encoder = encoder
        self._brute_force_token_limit = brute_force_token_limit
        self._brute_force_strategy = brute_force_strategy

    def create_tree(self, markdown: str) -> list[HeaderNode]:
        markdown_tree = self._markdown_tree_parser.parse(markdown)
        return markdown_tree

    async def summarize_text(
        self, markdown: str, level: int = 2, retrieval_k: int = 20
    ) -> CompaktRunResult:
        """Summarize pre-extracted markdown text (skips file reading)."""
        logger.info("Async text summarization started (level=%d, k=%d)", level, retrieval_k)
        if not markdown.strip():
            raise EmptyDocumentError("The provided text is empty")
        return await self._summarize_markdown(markdown, level=level, retrieval_k=retrieval_k)

    async def summarize(
        self, file_path: str, level: int = 2, retrieval_k: int = 20
    ) -> CompaktRunResult:
        logger.info(
            "Async summarization started: %s (level=%d, k=%d)", file_path, level, retrieval_k
        )
        if not self._file_reader:
            raise ValueError("File reader is required for summarize() but was not provided")

        markdown = await asyncio.to_thread(self._file_reader.read, file_path)
        if not markdown.strip():
            raise EmptyDocumentError("The document is empty after markdown conversion")
        return await self._summarize_markdown(markdown, level=level, retrieval_k=retrieval_k)

    async def _summarize_markdown(
        self, markdown: str, level: int, retrieval_k: int
    ) -> CompaktRunResult:
        try:
            if self.count_tokens(markdown) <= self._brute_force_token_limit:
                logger.info(
                    "Document is within brute-force token limit, using brute-force strategy"
                )

                result = await self._brute_force_strategy.run_async(
                    markdown=markdown,
                    markdown_tree=[],
                    chunks=[],
                    embeddings=[],
                    level=level,
                    retrieval_k=retrieval_k,
                )
                logger.info("Async summarization completed with brute-force strategy")
                return result

            markdown_tree_task = asyncio.to_thread(self._markdown_tree_parser.parse, markdown)
            chunks_task = asyncio.to_thread(self._text_splitter.split, markdown)
            markdown_tree, chunks = await asyncio.gather(markdown_tree_task, chunks_task)

            if not chunks:
                raise EmptyDocumentError("No chunks were produced from the document")
            logger.debug("Produced %d chunks", len(chunks))

            embeddings = await asyncio.to_thread(self._vector_index.index, chunks)
            logger.debug("Indexed %d embeddings", len(embeddings))

            for strategy in self._strategies:
                if strategy.can_handle(markdown, markdown_tree):
                    logger.info("Selected strategy: %s", strategy.name)
                    result = await strategy.run_async(
                        markdown=markdown,
                        markdown_tree=markdown_tree,
                        chunks=chunks,
                        embeddings=embeddings,
                        level=level,
                        retrieval_k=retrieval_k,
                    )
                    logger.info("Async summarization completed")
                    return result

            raise UnsupportedDocumentStrategyError(
                "No summarization strategy can handle this document"
            )
        finally:
            await asyncio.to_thread(self._vector_index.clear)

    def count_tokens(self, text: str) -> int:
        return len(self._encoder.encode(text))


AsyncCompaktClient = AsyncCompakt

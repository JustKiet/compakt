from dependency_injector import containers, providers
from docling.document_converter import DocumentConverter
from markdown_it import MarkdownIt

from compakt.client import Compakt
from compakt.core.adapters.in_memory_vector_index import InMemoryVectorIndex
from compakt.core.adapters.md_it_tree_parser import MarkdownItTreeParser
from compakt.core.adapters.openai_document_structure_resolver import (
    OpenAIDocumentStructureResolver,
)
from compakt.core.adapters.openai_embeddings import OpenAIEmbeddings
from compakt.core.adapters.openai_summarizer import OpenAISummarizer
from compakt.core.adapters.readers.docling_reader import DoclingFileReader
from compakt.core.adapters.readers.pymupdf_reader import PyMuPDFMarkdownFileReader
from compakt.core.adapters.text_splitters.md_text_splitter import (
    LangchainMarkdownTextSplitter,
)
from compakt.core.adapters.tiktoken_encoder import TiktokenEncoder
from compakt.core.models import MarkdownHeader
from compakt.strategies.fallback_unstructured import FallbackUnstructuredStrategy
from compakt.strategies.structured_markdown import StructuredMarkdownStrategy


class Container(containers.DeclarativeContainer):
    config = providers.Configuration()

    markdown_it = providers.Singleton(MarkdownIt)

    file_reader = providers.Singleton(PyMuPDFMarkdownFileReader)
    docling_document_converter = DocumentConverter()
    docling_file_reader = providers.Singleton(
        DoclingFileReader, document_converter=docling_document_converter
    )
    markdown_tree_parser = providers.Singleton(MarkdownItTreeParser, markdown_it=markdown_it)

    text_splitter = providers.Singleton(
        LangchainMarkdownTextSplitter,
        headers_to_split_on=providers.Object(
            [
                (MarkdownHeader.H1, "header_1"),
                (MarkdownHeader.H2, "header_2"),
                (MarkdownHeader.H3, "header_3"),
                (MarkdownHeader.H4, "header_4"),
            ]
        ),
    )

    embeddings = providers.Singleton(
        OpenAIEmbeddings,
        model=config.embedding_model.from_value("text-embedding-3-small"),
    )
    vector_index = providers.Singleton(InMemoryVectorIndex, embeddings=embeddings)

    encoder = providers.Singleton(
        TiktokenEncoder,
        encoding_name=config.encoding_name.from_value("cl100k_base"),
    )
    document_structure_resolver = providers.Singleton(
        OpenAIDocumentStructureResolver,
        model=config.chat_model.from_value("gpt-4.1-mini"),
        encoder=encoder,
    )
    summarizer = providers.Singleton(
        OpenAISummarizer,
        model=config.chat_model.from_value("gpt-4.1-mini"),
        encoder=encoder,
    )

    structured_markdown_strategy = providers.Factory(
        StructuredMarkdownStrategy,
        document_structure_resolver=document_structure_resolver,
        summarizer=summarizer,
        vector_index=vector_index,
    )
    fallback_unstructured_strategy = providers.Factory(
        FallbackUnstructuredStrategy,
        summarizer=summarizer,
        vector_index=vector_index,
    )

    strategies = providers.List(
        structured_markdown_strategy,
        fallback_unstructured_strategy,
    )

    compakt = providers.Factory(
        Compakt,
        file_reader=docling_file_reader,
        markdown_tree_parser=markdown_tree_parser,
        text_splitter=text_splitter,
        vector_index=vector_index,
        strategies=strategies,
        encoder=encoder,
    )

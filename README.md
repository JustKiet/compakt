# Compakt

Document summarization library that reads PDF and Markdown files, preserves their hierarchical structure, and produces LLM-powered summaries using OpenAI models.

Compakt parses documents into structured chunks, embeds them for similarity search, then applies the best-fit summarization strategy — whether the document has rich headers, no structure at all, or is small enough to process whole.

## Features

- **Hierarchical structure preservation** — Extracts and maintains document headers, sections, and subsections throughout the summarization pipeline
- **Three-strategy system** — Automatically selects the best approach per document:
  - *Brute Force* for small documents (< 50k tokens) — sends full text to the LLM
  - *Structured Markdown* for documents with headers — scope-filtered retrieval per section using fuzzy matching and elbow-filtered similarity search
  - *Fallback Unstructured* for headerless documents — global similarity search with synthetic structure
- **Configurable granularity** — `level` parameter (1–3) controls summary depth: sections only, subsections, or down to H4 headers
- **Sync and async clients** — `Compakt` for synchronous use, `AsyncCompakt` for async workflows and batch processing
- **Pluggable architecture** — Protocol-based interfaces let you swap any component (file reader, embeddings, vector index, summarizer) without changing the pipeline
- **PDF and Markdown input** — Reads local files and HTTP/HTTPS URLs

## Installation

Requires Python 3.13+.

```bash
pip install compakt
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add compakt
```

### Environment Setup

Compakt uses OpenAI models by default. Set your API key:

```bash
export OPENAI_API_KEY="your-api-key"
```

Or create a `.env` file in your project root:

```
OPENAI_API_KEY=your-api-key
```

## Quick Start

### Basic Usage

```python
from compakt import Compakt

compakt = Compakt()
result = compakt.summarize("path/to/document.pdf", level=2)

print(result.summary)
print(f"Strategy used: {result.artifacts.strategy}")
print(f"Chunks processed: {len(result.artifacts.chunks)}")
```

### Async Usage

```python
import asyncio
from compakt import AsyncCompakt

async def main():
    compakt = AsyncCompakt()
    result = await compakt.summarize("path/to/document.pdf", level=2)
    print(result.summary)

asyncio.run(main())
```

### Batch Processing

```python
import asyncio
from compakt import AsyncCompakt

async def main():
    compakt = AsyncCompakt()
    files = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
    semaphore = asyncio.Semaphore(4)

    async def process(path):
        async with semaphore:
            return await compakt.summarize(path, level=2)

    results = await asyncio.gather(*[process(f) for f in files])
    for r in results:
        print(r.summary)

asyncio.run(main())
```

## API Reference

### `Compakt` / `AsyncCompakt`

```python
Compakt(
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
)
```

Pass `None` for any component to use the built-in default. Override specific components while keeping defaults for everything else.

#### Methods

| Method | Description |
|--------|-------------|
| `summarize(file_path, level=2, retrieval_k=20)` | Summarize a document. Returns `CompaktRunResult`. |
| `create_tree(markdown)` | Parse markdown string into a header tree (`list[HeaderNode]`). |
| `count_tokens(text)` | Count tokens using the configured encoder. |

**Parameters:**
- `file_path` — Path to a PDF or Markdown file (or HTTP/HTTPS URL)
- `level` — Summary granularity: `1` = sections, `2` = subsections, `3` = H4 headers
- `retrieval_k` — Number of top-k chunks to retrieve before elbow filtering (default: 20)

### `CompaktRunResult`

```python
result.summary       # str — The generated summary
result.artifacts     # CompaktRunArtifacts
```

### `CompaktRunArtifacts`

```python
artifacts.markdown            # str — Raw markdown from file reader
artifacts.markdown_tree       # list[HeaderNode] — Parsed header tree
artifacts.chunks              # list[CompaktChunk] — Text chunks
artifacts.embeddings          # list[CompaktEmbeddingEntry] — Chunk embeddings
artifacts.retrieved_chunks    # dict[str, list[CompaktChunk]] — Chunks retrieved per section
artifacts.document_structure  # DocumentStructure | None — Resolved structure
artifacts.strategy            # str — Name of the strategy used
```

## Architecture

Compakt follows the **Ports & Adapters** pattern. All core abstractions are Python `Protocol` classes in `src/compakt/core/interfaces/`, with concrete implementations in `src/compakt/core/adapters/`.

### Pipeline Flow

```
File (PDF/MD) → FileReader → Raw Markdown
                                  ↓
                          MarkdownTreeParser → Header Tree
                                  ↓
                            TextSplitter → Chunks
                                  ↓
                            VectorIndex → Embedded & Indexed Chunks
                                  ↓
                    SummarizationStrategy (auto-selected)
                                  ↓
                          CompaktRunResult
```

### Strategy Selection

Strategies are evaluated in order. The first whose `can_handle()` returns `True` is used:

1. **BruteForceUnstructuredStrategy** — If total tokens ≤ `brute_force_token_limit`
2. **StructuredMarkdownStrategy** — If the document has headers
3. **FallbackUnstructuredStrategy** — If the document has no headers

### Default Components

| Component | Default Implementation |
|-----------|----------------------|
| File Reader | `PyMuPDFMarkdownFileReader` (pymupdf4llm) |
| Tree Parser | `MarkdownItTreeParser` (markdown-it-py) |
| Text Splitter | `LangchainMarkdownTextSplitter` |
| Encoder | `TiktokenEncoder` (cl100k_base) |
| Embeddings | `OpenAIEmbeddings` (text-embedding-3-small) |
| Vector Index | `InMemoryVectorIndex` (cosine similarity) |
| Structure Resolver | `OpenAIDocumentStructureResolver` (gpt-4.1-mini) |
| Summarizer | `OpenAISummarizer` (gpt-4.1-mini) |

### Custom Components

Implement any `Protocol` from `compakt.core.interfaces` and pass it to the client:

```python
from compakt import Compakt
from compakt.core.interfaces import Embeddings

class MyEmbeddings:
    def embed(self, payload):
        # your implementation
        ...

    async def aembed(self, payload):
        ...

compakt = Compakt(
    # swap just the embeddings, keep everything else default
    vector_index=InMemoryVectorIndex(MyEmbeddings()),
)
```

## Development

```bash
# Clone and install
git clone https://github.com/justkiet/compakt.git
cd compakt
uv sync

# Run tests
uv run python -m pytest tests/

# Run a single test
uv run python -m pytest tests/test_compakt_integration.py::CompaktIntegrationTest::test_method_name

# Run an example
uv run python examples/basic_usage.py path/to/file.pdf --level 2
```

## License

MIT

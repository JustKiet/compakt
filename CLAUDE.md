# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Compakt is a Python library for document summarization. It reads PDF/Markdown files, parses their structure, chunks and embeds the content, then uses LLM-powered strategies to produce structured summaries. It uses OpenAI models (GPT-4.1-mini for chat, text-embedding-3-small for embeddings) by default.

## Commands

- **Install**: `uv sync` (uses uv with Python 3.13)
- **Run tests**: `uv run python -m pytest tests/`
- **Run single test**: `uv run python -m pytest tests/test_compakt_integration.py::CompaktIntegrationTest::test_method_name`
- **Run backend**: `uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000` (from `src/`)
- **Example usage**: `uv run python examples/basic_usage.py <file_path> --level 2`

## Architecture

### Core Pipeline

The `Compakt` and `AsyncCompakt` clients (`src/compakt/client.py`) orchestrate the summarization pipeline:

1. **FileReader** reads PDF/Markdown into raw markdown text
2. **MarkdownTreeParser** parses markdown into a header tree (`list[HeaderNode]`)
3. **TextSplitter** chunks the markdown by headers into `CompaktChunk` objects
4. **VectorIndex** embeds and indexes chunks for similarity search
5. **SummarizationStrategy** (selected via `can_handle`) retrieves relevant chunks and generates the summary

### Interfaces and Adapters (Ports & Adapters pattern)

All core abstractions are Protocol classes in `src/compakt/core/interfaces/`:
- `FileReaderAsMarkdown`, `MarkdownTreeParser`, `TextSplitter`, `Encoder`
- `Embeddings`, `VectorIndex`, `DocumentStructureResolver`, `Summarizer`
- `SummarizationStrategy` — the top-level strategy protocol

Concrete implementations live in `src/compakt/core/adapters/` (e.g., `PyMuPDFMarkdownFileReader`, `TiktokenEncoder`, `OpenAIEmbeddings`, `InMemoryVectorIndex`).

### Strategies

Strategies in `src/compakt/strategies/` are tried in order; the first whose `can_handle()` returns True is used:
1. **StructuredMarkdownStrategy** — for documents with headers. Uses `DocumentStructureResolver` to extract a `DocumentStructure` model, then retrieves scoped chunks per section using fuzzy title matching (rapidfuzz) and elbow-filtered similarity search.
2. **FallbackUnstructuredStrategy** — for headerless documents. Performs a single global similarity search.

Both strategies support sync `run()` and `run_async()` methods.

### Key Models (`src/compakt/core/models.py`)

- `CompaktChunk` — a text chunk with header metadata
- `CompaktEmbeddingEntry` — chunk + embedding vector
- `DocumentStructure` → `Section` → `Subsection` → `H4Header` — hierarchical document outline (Pydantic models)
- `CompaktRunResult` — final output containing `summary` string and `CompaktRunArtifacts`

### Dependency Injection

`src/compakt/containers.py` provides a `dependency-injector` `Container` for wiring all components. Tests override providers with fakes (see `tests/test_compakt_integration.py`). The `Compakt` client also has a `build_defaults()` static method for standalone use without the container.

### Backend (separate app)

`src/backend/` is a FastAPI service for async summarization via Azure Blob/Queue Storage. It is a separate application from the core library, with its own `pipeline_state.py` for Azure resource management.

## Conventions

- All domain exceptions inherit from `CompaktError` (`src/compakt/core/exceptions.py`)
- Tests use `unittest.TestCase` with fake implementations (not mocks)
- The `level` parameter (1-3) controls summary granularity: 1=sections, 2=subsections, 3=H4 headers
- Async variants use `asyncio.to_thread` to wrap sync adapter calls

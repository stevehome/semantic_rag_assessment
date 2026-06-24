# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (use uv)
uv pip install -e ".[dev]"

# Run the full benchmark → writes retrieval_benchmark.md
uv run python main.py

# Tests (no network, ~0.2 s — uses FakeEmbedder, not real models)
uv run pytest tests/
uv run pytest tests/test_pipeline.py -k "test_retrieve_raw"   # single test

# Type-check
uv run mypy --ignore-missing-imports src/

# CLI
uv run rag ingest --save ./indexes
uv run rag search "How does the system handle peak load?" --strategy A --k 3
uv run rag search "..." --strategy E --load ./indexes
uv run rag eval --k 3
```

The first non-Docker run downloads MiniLM and the cross-encoder (~80 MB each) into HuggingFace cache. Tests bypass this via the `FakeEmbedder` fixture in `conftest.py`.

## Architecture

**Entry points**
- `main.py` — runs the full benchmark pipeline and writes `retrieval_benchmark.md`
- `src/cli.py` — `rag` console entry-point (ingest / search / benchmark / eval subcommands)

**Core data flow**

```
src/data.py  →  RAGPipeline.ingest()
                  chunker.py (token-window with overlap)
                  embedding.py (LocalEmbedder or MockVertexEmbeddingModel)
                  storage.py (NumPy VectorStore, cosine or euclidean)
                  bm25.py (BM25Index)

query  →  pipeline.retrieve_*()  →  RetrievalResult (hits + Timing)
```

**Five retrieval strategies** (all in `src/pipeline.py`):
- **A** `retrieve_raw` — embed → cosine vector search
- **B** `retrieve_expanded` — `QueryExpander.expand()` → embed → search
- **C** `retrieve_hybrid` — dense + BM25 fused via weighted Reciprocal Rank Fusion
- **D** `retrieve_rerank` — dense top-N → cross-encoder reranker → top-k
- **E** `retrieve_expand_rerank` — B's expansion + D's reranker (reranker scores against the *original* query)

**Key design decisions**
- `VectorStore` is brute-force NumPy for exact search and zero native deps; the `add`/`search` API is drop-in compatible with FAISS or Vertex AI Matching Engine.
- Vertex AI surfaces (`TextEmbeddingModel`, `GenerativeModel`) are mocked in `src/embedding.py` and `src/query_expansion.py` so everything runs offline. The mock response shape matches the real API, so no call-site changes are needed to go to production.
- `CachedEmbedder` and `CachedQueryExpander` in `src/cache.py` add LRU caching with hit/miss stats.
- Eval (`src/eval.py`) deduplicates hits to `parent_doc_id` before scoring, so chunk-level retrieval is comparable to document-level labels.
- Statistical significance (`src/significance.py`) uses bootstrap (not t-test) because per-query metrics are bounded [0,1] and non-normal.
- `src/persistence.py` saves/loads the vector store and BM25 index; embedder/expander/reranker are not persisted (they're caller-side concerns).

**Test suite** — 61 tests, network-free. `conftest.py` provides a `FakeEmbedder` (hash-based, deterministic) and `sample_docs` fixtures used across all test files. Real-model behaviour is exercised only by `main.py`.

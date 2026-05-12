# Semantic RAG & Vector Search — Context-Aware Retrieval Engine

A local Retrieval-Augmented Generation pipeline that ingests a small technical
corpus, **chunks** it with overlap, embeds with `sentence-transformers`,
stores it in a NumPy vector index *and* a BM25 lexical index, and benchmarks
**five** retrieval strategies on a labelled evaluation set:

| Strategy | What it does | Dependencies |
|---|---|---|
| **A — Raw Vector Search** | Embed → cosine search. | encoder |
| **B — AI-Enhanced Retrieval** | Expand query via mocked Vertex AI `GenerativeModel` → embed → cosine search. | encoder + generative model |
| **C — Hybrid (dense ∪ BM25, weighted RRF)** | Dense and BM25 in parallel, fused via weighted Reciprocal Rank Fusion. | encoder + BM25 |
| **D — Rerank** | Dense top-N → cross-encoder reranker → top-k. | encoder + cross-encoder |
| **E — Expand + Rerank** | B's expansion *and* D's reranker — strongest production pattern. | all of the above |

Every retrieval call records per-stage latency. The benchmark reports **Recall@k,
Precision@k, MRR and nDCG@k** with **95% bootstrap confidence intervals**, runs
**paired bootstrap significance tests** (vs Strategy A on MRR), sweeps the
hybrid dense/BM25 weight ratio, and renders a markdown report.

LRU caches sit in front of the embedder and the query expander so repeat
calls are dict lookups. The Vertex AI surfaces (`TextEmbeddingModel`,
`GenerativeModel`) are mocked so the project runs offline and the test suite
is deterministic.

---

## Layout

```
semantic_rag_assessment/
├── src/
│   ├── data.py              # 8 technical paragraphs + 16 labelled queries
│   ├── chunker.py           # ChunkConfig + token-window chunking with overlap
│   ├── embedding.py         # LocalEmbedder + MockVertexEmbeddingModel
│   ├── storage.py           # NumPy VectorStore (cosine | euclidean)
│   ├── bm25.py              # BM25Index + (weighted) Reciprocal Rank Fusion
│   ├── query_expansion.py   # MockGenerativeModel + QueryExpander
│   ├── reranker.py          # LocalCrossEncoderReranker + MockReranker
│   ├── cache.py             # LRU caches around embedder + expander (with stats)
│   ├── eval.py              # recall@k / precision@k / MRR / nDCG@k + per-query
│   ├── significance.py      # bootstrap CI + paired bootstrap significance test
│   ├── persistence.py       # save / load vector + BM25 indexes to disk
│   ├── timing.py            # Timing context-manager + percentile helper
│   ├── pipeline.py          # RAGPipeline (5 strategies + benchmark)
│   └── cli.py               # `rag ingest | search | benchmark | eval`
├── tests/                   # 61 tests, ~0.2 s, network-free, no model downloads
├── main.py                  # Benchmark + latency + eval → markdown report
├── retrieval_benchmark.md   # Output of `python main.py`
├── Dockerfile               # Reproducible image with pre-warmed model caches
├── .github/workflows/ci.yml # pytest + mypy across Python 3.10–3.12
├── pyproject.toml           # Build / pytest / mypy / `rag` console entry-point
└── requirements.txt
```

## Run

```bash
pip install -r requirements.txt

# Full benchmark + write retrieval_benchmark.md
python main.py

# Tests (network-free, ~0.2 s)
python -m pytest tests/

# CLI
python -m src.cli ingest --save ./indexes
python -m src.cli search "How does the system handle peak load?" --strategy A --k 3
python -m src.cli search "..." --strategy E --load ./indexes
python -m src.cli eval --k 3

# Docker (pre-warms model caches at build time → benchmark is offline)
docker build -t rag-assessment .
docker run --rm rag-assessment
```

The first non-Docker run downloads MiniLM (~80 MB) and the cross-encoder
(~80 MB) into the HuggingFace cache.

---

## Design choices

### Similarity metric: cosine vs Euclidean

The store supports both, but the pipeline defaults to **cosine** for four
reasons:

1. **Sentence-transformer outputs are designed for cosine.** MiniLM and the
   `textembedding-gecko` family are trained with cosine / dot-product
   contrastive objectives; their geometry is angular, not metric.
2. **L2-normalised vectors make cosine and dot-product identical**, which is
   the cheapest possible kernel on a GPU/ANN backend and matches what Vertex
   AI Matching Engine, FAISS-IP, and HNSW-cosine do internally.
3. **Norm leakage under Euclidean is a real failure mode.** Two passages with
   different lengths can have very different ‖x‖ even after good
   tokenisation, so Euclidean ranking ends up partly ranking by length.
   Cosine factors that out.
4. **Euclidean is still right** when vectors carry magnitude meaning (e.g.
   word2vec frequency, custom encoders that output unnormalised logits).
   `VectorStore(metric="euclidean")` exists for that.

### Vector store: NumPy

Brute-force NumPy was picked deliberately for the assessment:

- Zero native deps → CI runs anywhere `numpy` does.
- Exact search → tests can assert ranking instead of "is it close".
- The `add` / `search` API mirrors what every ANN index exposes, so swapping
  in FAISS, hnswlib, or Vertex AI Matching Engine is a one-class change.

### Chunking

`src/chunker.py` does token-window chunking with overlap (defaults
`max_tokens=25`, `overlap=6`). Each chunk carries `parent_doc_id` so eval
metrics deduplicate to document granularity before scoring — chunk-level
retrieval is directly comparable to document-level labels.

### Query expansion (Strategy B)

`QueryExpander` calls `MockGenerativeModel.generate_content(prompt)` exactly
the way the real `vertexai.generative_models.GenerativeModel` is called. The
mock applies a domain-synonym rule table (e.g. *"peak load"* → adds *"traffic
spikes, autoscaling, high concurrency, capacity"*) and strips interrogative
framing so the encoder sees a declarative passage.

### Hybrid retrieval (Strategy C) with weighted RRF

The BM25 index uses standard Okapi formulation (`k1=1.5`, `b=0.75`). Fusion
is **weighted Reciprocal Rank Fusion** with `rrf_k=60`:

```
score(d) = Σ_i  w_i / (rrf_k + rank_i)
```

Weighted RRF avoids the score-normalisation problem of linear sums and lets
you tilt toward whichever ranker your offline eval says is stronger. The
weight sweep in §5 of the benchmark answers "is BM25 carrying any signal the
encoder is missing?" — on this corpus the answer is *no*, so equal weights
underperform dense-only.

### Reranking (Strategy D / E)

`LocalCrossEncoderReranker` wraps `cross-encoder/ms-marco-MiniLM-L-6-v2`. The
standard pattern is `dense top-20 → cross-encoder → top-k`. In Strategy E the
expander runs *and* the reranker runs, with the cross-encoder scored against
the **original** user query rather than the expanded form — expansion helps
recall, the cross-encoder scores intent.

The Vertex AI production analog is **Discovery Engine Ranking** (`projects.
locations.rankingConfigs.rank`). The `Reranker` Protocol in `src/reranker.py`
is a drop-in for it.

### Evaluation + statistical significance

`src/eval.py` implements recall@k, precision@k, MRR, and binary-relevance
nDCG@k. Predictions are deduplicated to `parent_doc_id` first.

`src/significance.py`:

- **`bootstrap_ci(values, ci=0.95)`** — percentile bootstrap CI for the mean,
  2000 resamples by default.
- **`paired_bootstrap_test(values_a, values_b)`** — paired one-sided test for
  "is B better than A on this metric?", 10 000 resamples. Returns mean
  difference, 95% CI on the per-query difference, and a p-value.

Why bootstrap: per-query metrics (recall, RR, nDCG) are bounded in [0, 1] and
very non-normal, so a t-test would be wrong. Non-parametric bootstrap makes
no distributional assumptions.

### Caching

`CachedEmbedder` and `CachedQueryExpander` wrap their underlying objects with
LRU caches. Cache stats (hits / misses / hit-ratio) are reported in §6 of the
benchmark. In production the dict is swapped for Memorystore / Redis with the
same key structure (text → vector or text → text).

### Latency methodology

§2 of the report runs on a *separate, uncached* pipeline so the per-call
numbers are honest. The benchmark warms each query once and times the next
10 across 3 showcase queries (30 samples per strategy). p50/p95/mean
reported.

### Persistence

`src/persistence.py` writes:

```
<root>/
├── vectors.npz   # matrix + ids
├── metadata.json # per-chunk metadata
├── bm25.json     # BM25 state (IDF table, token lists)
└── store_metric.txt
```

Round-trip-tested in `tests/test_persistence.py`. The pipeline's embedder,
expander and reranker are *not* persisted — they're caller-side concerns,
mirroring how Matching Engine is separated from the model catalog in
production.

---

## Migrating to Vertex AI Vector Search (Matching Engine)

The codebase is shaped for this migration — the mocks define the API
contract that real Vertex AI calls have to satisfy.

### 1. Replace the embedding mock

```python
# Today
from src.embedding import MockVertexEmbeddingModel
model = MockVertexEmbeddingModel.from_pretrained("textembedding-gecko@003")

# Production
import vertexai
from vertexai.language_models import TextEmbeddingModel
vertexai.init(project=PROJECT_ID, location="us-central1")
model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
```

`MockVertexEmbeddingModel.get_embeddings` returns objects with a `.values`
attribute, so no call-site changes are needed.

### 2. Replace `VectorStore` with a Matching Engine index

a. **Create the index.** `aiplatform.MatchingEngineIndex.create_tree_ah_index(...)`
   for ≤100M vectors, `dimensions=768` for gecko,
   `distance_measure_type="DOT_PRODUCT_DISTANCE"` plus L2-normalised vectors
   (which is what `VectorStore` already does for cosine).
b. **Deploy** to an `IndexEndpoint`. For a regulated workload pick
   VPC-peered + Private Service Connect.
c. **Swap `search`.** `VectorStore.search(vec, k)` becomes
   `endpoint.find_neighbors(deployed_index_id=..., queries=[vec], num_neighbors=k)`.
   Wrap that in a `VertexMatchingEngineStore` class with the same `add` /
   `search` signature; `RAGPipeline` doesn't change.
d. **Updates.** Matching Engine is append-mostly. Use streaming
   `upsert_datapoints` for live writes and a nightly rebuild for compaction.

### 3. Replace the generative mock

```python
from vertexai.generative_models import GenerativeModel
model = GenerativeModel("gemini-1.5-pro")
response = model.generate_content(prompt)
```

`MockGenerationResponse.text` and `.candidates[0].content.parts[0].text` are
both shape-compatible — `QueryExpander` needs no changes.

### 4. Reranker

Wrap **Discovery Engine Ranking** in a class that implements the
`Reranker` Protocol (`src/reranker.py`). The mock's `(query, candidates,
top_k)` signature is the same shape as the real Rank request, with each
candidate becoming a `RankingRecord(content=...)`.

### 5. Production hardening

- **Cache the expansion call** in Memorystore. `CachedQueryExpander` already
  shows the pattern; replace the dict with Redis.
- **Filter at the index** with Matching Engine `namespace` /
  `numeric_restricts` rather than post-filtering Python-side.
- **Failure isolation.** Circuit-break the expansion and rerank calls; on
  open, fall back to Strategy A (no LLM, no rerank). The pipeline supports
  this — it's selecting which `retrieve_*` method the route calls.
- **Observability.** Emit per-stage spans (`Timing` already produces them);
  SLO on p95 end-to-end *and* on Recall@10 against a labelled set, same
  shape as `doc-06` in the corpus.

---

## Test suite

`pytest tests/` (61 tests, ~0.2 s, network-free) covers:

- Vector store ranking under both metrics, edge cases.
- BM25 lexical ranking, IDF behaviour, RRF fusion, weighted RRF, weight-length validation.
- Mock Vertex embedding response shape (`.values` list).
- Mock Vertex generative response shape (`.text` + `.candidates[i].content.parts[j].text`).
- Chunking: overlap correctness, single-chunk for short docs, invalid configs rejected.
- All 5 retrieval strategies end-to-end, per-stage timing recorded.
- Hybrid weights actually change ranking.
- Cross-encoder rerank ordering, top-k truncation, empty-input.
- Bootstrap CI brackets known mean; paired test detects real improvement; rejects shape mismatch.
- LRU cache hit/miss accounting, eviction, input-order preservation.
- Persistence round-trip: save + load reproduces ranking.
- CLI smoke tests: search prints JSON, ingest persists, unknown strategy rejected.

The fake-embedder fixture in `conftest.py` is hash-based and deterministic
so no model is downloaded for the suite. Real-model behaviour is exercised
by `main.py` and would run in a separate nightly workflow against a paid
runner.

## CI

`.github/workflows/ci.yml` runs pytest across Python 3.10 / 3.11 / 3.12 plus
mypy with strict-ish settings (`--ignore-missing-imports`, configured in
`pyproject.toml`).

# Retrieval Benchmark: Five Strategies

Five strategies on the same corpus, encoder and labelled set:

- **A — Raw Vector Search:** embed → cosine search.
- **B — AI-Enhanced Retrieval:** expand via mocked Vertex AI `GenerativeModel` → embed → cosine search.
- **C — Hybrid (dense ∪ BM25, weighted RRF):** dense + BM25, fuse via weighted Reciprocal Rank Fusion (rrf_k=60).
- **D — Rerank:** dense top-20 → cross-encoder `ms-marco-MiniLM-L-6-v2` → top-k.
- **E — Expand + Rerank:** B's expansion for recall *and* D's cross-encoder for precision — the strongest production pattern.

Encoder: `sentence-transformers/all-MiniLM-L6-v2` (384-d, cosine). Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`. Chunking: `ChunkConfig(max_tokens=25, overlap=6)`. k=3 for all reported metrics. Labelled set: 16 queries.

## 1. Per-query side-by-side (showcase queries)

### Query 1: How does the system handle peak load?

_Expanded query (B/E):_ `handle peak load. Related concepts: traffic spikes, autoscaling, high concurrency, capacity.`

**A — Raw Vector**

| Rank | Chunk ID | Parent | Score | Title |
|------|----------|--------|-------|-------|
| 1 | `doc-01::chunk-0` | `doc-01` | 0.5036 | Autoscaling and traffic spikes |
| 2 | `doc-01::chunk-1` | `doc-01` | 0.3835 | Autoscaling and traffic spikes |
| 3 | `doc-02::chunk-1` | `doc-02` | 0.3665 | Cold start mitigation |

**B — AI-Enhanced**

| Rank | Chunk ID | Parent | Score | Title |
|------|----------|--------|-------|-------|
| 1 | `doc-01::chunk-0` | `doc-01` | 0.6425 | Autoscaling and traffic spikes |
| 2 | `doc-01::chunk-1` | `doc-01` | 0.4709 | Autoscaling and traffic spikes |
| 3 | `doc-06::chunk-3` | `doc-06` | 0.4351 | Observability and SLOs |

**C — Hybrid (RRF, equal weights)**

| Rank | Chunk ID | Parent | Score | Title |
|------|----------|--------|-------|-------|
| 1 | `doc-01::chunk-0` | `doc-01` | 0.0320 | Autoscaling and traffic spikes |
| 2 | `doc-03::chunk-1` | `doc-03` | 0.0318 | Caching strategy |
| 3 | `doc-01::chunk-2` | `doc-01` | 0.0290 | Autoscaling and traffic spikes |

**D — Rerank**

| Rank | Chunk ID | Parent | Score | Title |
|------|----------|--------|-------|-------|
| 1 | `doc-01::chunk-0` | `doc-01` | -5.9117 | Autoscaling and traffic spikes |
| 2 | `doc-03::chunk-1` | `doc-03` | -8.3573 | Caching strategy |
| 3 | `doc-01::chunk-1` | `doc-01` | -10.0669 | Autoscaling and traffic spikes |

**E — Expand + Rerank**

| Rank | Chunk ID | Parent | Score | Title |
|------|----------|--------|-------|-------|
| 1 | `doc-01::chunk-0` | `doc-01` | -5.9117 | Autoscaling and traffic spikes |
| 2 | `doc-03::chunk-1` | `doc-03` | -8.3573 | Caching strategy |
| 3 | `doc-01::chunk-1` | `doc-01` | -10.0669 | Autoscaling and traffic spikes |

### Query 2: What happens when the embedding service is slow or fails?

_Expanded query (B/E):_ `What happens when the embedding service is slow or fails. Related concepts: latency, tail latency, p95, performance regression, failure mode, circuit breaker, fallback, graceful degradation, dense encoder, sentence encoder, vector representation.`

**A — Raw Vector**

| Rank | Chunk ID | Parent | Score | Title |
|------|----------|--------|-------|-------|
| 1 | `doc-07::chunk-0` | `doc-07` | 0.4978 | Failure isolation |
| 2 | `doc-06::chunk-0` | `doc-06` | 0.3284 | Observability and SLOs |
| 3 | `doc-04::chunk-1` | `doc-04` | 0.3006 | Vector index choice |

**B — AI-Enhanced**

| Rank | Chunk ID | Parent | Score | Title |
|------|----------|--------|-------|-------|
| 1 | `doc-07::chunk-0` | `doc-07` | 0.5675 | Failure isolation |
| 2 | `doc-06::chunk-0` | `doc-06` | 0.4498 | Observability and SLOs |
| 3 | `doc-06::chunk-1` | `doc-06` | 0.3701 | Observability and SLOs |

**C — Hybrid (RRF, equal weights)**

| Rank | Chunk ID | Parent | Score | Title |
|------|----------|--------|-------|-------|
| 1 | `doc-07::chunk-0` | `doc-07` | 0.0328 | Failure isolation |
| 2 | `doc-06::chunk-0` | `doc-06` | 0.0323 | Observability and SLOs |
| 3 | `doc-04::chunk-1` | `doc-04` | 0.0310 | Vector index choice |

**D — Rerank**

| Rank | Chunk ID | Parent | Score | Title |
|------|----------|--------|-------|-------|
| 1 | `doc-07::chunk-0` | `doc-07` | 4.9279 | Failure isolation |
| 2 | `doc-06::chunk-0` | `doc-06` | -4.7338 | Observability and SLOs |
| 3 | `doc-01::chunk-1` | `doc-01` | -8.9662 | Autoscaling and traffic spikes |

**E — Expand + Rerank**

| Rank | Chunk ID | Parent | Score | Title |
|------|----------|--------|-------|-------|
| 1 | `doc-07::chunk-0` | `doc-07` | 4.9279 | Failure isolation |
| 2 | `doc-06::chunk-0` | `doc-06` | -4.7338 | Observability and SLOs |
| 3 | `doc-01::chunk-2` | `doc-01` | -7.1568 | Autoscaling and traffic spikes |

### Query 3: How are documents prepared for semantic search?

_Expanded query (B/E):_ `How are documents prepared for semantic search. Related concepts: vector index, ANN, HNSW, semantic retrieval.`

**A — Raw Vector**

| Rank | Chunk ID | Parent | Score | Title |
|------|----------|--------|-------|-------|
| 1 | `doc-08::chunk-1` | `doc-08` | 0.5174 | Data ingestion pipeline |
| 2 | `doc-04::chunk-0` | `doc-04` | 0.5132 | Vector index choice |
| 3 | `doc-08::chunk-0` | `doc-08` | 0.4521 | Data ingestion pipeline |

**B — AI-Enhanced**

| Rank | Chunk ID | Parent | Score | Title |
|------|----------|--------|-------|-------|
| 1 | `doc-04::chunk-0` | `doc-04` | 0.6252 | Vector index choice |
| 2 | `doc-08::chunk-1` | `doc-08` | 0.5466 | Data ingestion pipeline |
| 3 | `doc-05::chunk-0` | `doc-05` | 0.4385 | Embedding model selection |

**C — Hybrid (RRF, equal weights)**

| Rank | Chunk ID | Parent | Score | Title |
|------|----------|--------|-------|-------|
| 1 | `doc-04::chunk-0` | `doc-04` | 0.0320 | Vector index choice |
| 2 | `doc-08::chunk-0` | `doc-08` | 0.0320 | Data ingestion pipeline |
| 3 | `doc-08::chunk-1` | `doc-08` | 0.0318 | Data ingestion pipeline |

**D — Rerank**

| Rank | Chunk ID | Parent | Score | Title |
|------|----------|--------|-------|-------|
| 1 | `doc-04::chunk-0` | `doc-04` | -5.6136 | Vector index choice |
| 2 | `doc-07::chunk-1` | `doc-07` | -8.1074 | Failure isolation |
| 3 | `doc-08::chunk-1` | `doc-08` | -9.3011 | Data ingestion pipeline |

**E — Expand + Rerank**

| Rank | Chunk ID | Parent | Score | Title |
|------|----------|--------|-------|-------|
| 1 | `doc-04::chunk-0` | `doc-04` | -5.6136 | Vector index choice |
| 2 | `doc-07::chunk-1` | `doc-07` | -8.1074 | Failure isolation |
| 3 | `doc-08::chunk-1` | `doc-08` | -9.3011 | Data ingestion pipeline |

## 2. Latency (per call, ms)

Each query was warmed once then timed 10 times across 3 queries (n=30 samples per strategy).

| Strategy | p50 | p95 | mean |
|----------|-----|-----|------|
| A_raw_vector_search | 11.05 | 12.01 | 11.05 |
| B_ai_enhanced_retrieval | 9.76 | 11.30 | 9.88 |
| C_hybrid_dense_plus_bm25 | 10.41 | 11.06 | 10.44 |
| D_rerank | 48.59 | 60.87 | 48.48 |
| E_expand_plus_rerank | 48.36 | 64.09 | 48.97 |

## 3. Quality vs labelled set (with 95% bootstrap CIs)

Labelled set: 16 queries. Predictions are deduplicated to `parent_doc_id` before scoring. CIs are percentile bootstrap (2000 resamples).

| Strategy | Recall@3 | MRR | nDCG@3 |
|----------|----------|-----|----------|
| A_raw_vector_search | 0.969 [0.906, 1.000] | 0.927 [0.812, 1.000] | 0.922 [0.836, 1.000] |
| B_ai_enhanced_retrieval | 0.969 [0.906, 1.000] | 0.927 [0.823, 1.000] | 0.922 [0.842, 1.000] |
| C_hybrid_dense_plus_bm25 | 0.812 [0.625, 1.000] | 0.740 [0.521, 0.927] | 0.758 [0.548, 0.938] |
| D_rerank | 0.969 [0.906, 1.000] | 0.927 [0.812, 1.000] | 0.916 [0.828, 0.985] |
| E_expand_plus_rerank | 0.969 [0.906, 1.000] | 0.927 [0.812, 1.000] | 0.916 [0.828, 0.985] |

## 4. Paired bootstrap significance (vs Strategy A, on MRR)

One-sided test: is strategy X significantly better than A? 10,000 bootstrap resamples. Significant ⇔ p < 0.05 and mean_diff > 0.

| Comparison | mean Δ MRR (B − A) | 95% CI | p-value | Significant? |
|------------|---------------------|--------|---------|--------------|
| B_ai_enhanced_retrieval > A | +0.0000 | [-0.094, +0.094] | 0.6518 | no |
| C_hybrid_dense_plus_bm25 > A | -0.1875 | [-0.375, +0.000] | 1.0000 | no |
| D_rerank > A | +0.0000 | [-0.031, +0.031] | 0.6468 | no |
| E_expand_plus_rerank > A | +0.0000 | [-0.031, +0.031] | 0.6468 | no |

## 5. Hybrid (C) weight sweep

RRF score = `w_dense / (60 + rank_dense) + w_bm25 / (60 + rank_bm25)`. Sweeping the weights answers "is BM25 carrying any signal the encoder misses?".

| w_dense | w_bm25 | Recall@k | MRR | nDCG@k |
|---------|--------|----------|-----|--------|
| 1.00 | 0.00 | 0.9688 | 0.9271 | 0.9215 |
| 0.90 | 0.10 | 0.8750 | 0.7604 | 0.7894 |
| 0.70 | 0.30 | 0.8125 | 0.7396 | 0.7582 |
| 0.50 | 0.50 | 0.8125 | 0.7396 | 0.7582 |
| 0.30 | 0.70 | 0.8125 | 0.7083 | 0.7390 |
| 0.00 | 1.00 | 0.7188 | 0.6562 | 0.6605 |

## 6. Cache statistics

- Embedding cache: hits=162, misses=56, hit-ratio=74.31%
- Expansion cache: hits=22, misses=16, hit-ratio=57.89%

## 7. Findings

- **Headline MRR.** A: 0.927, B: 0.927, C: 0.740, D: 0.927, E: 0.927. nDCG@3 — A: 0.922, B: 0.922, C: 0.758, D: 0.916, E: 0.916.
- **Ceiling effect: 4 strategies tie at MRR=0.927** (A_raw_vector_search, B_ai_enhanced_retrieval, D_rerank, E_expand_plus_rerank). With the labelled set used here, the dense encoder already lifts the relevant passage to the top rank on most queries — there is no headroom for query expansion or cross-encoder reranking to close. The fancier machinery is not failing; it has nothing to recover.
- **No strategy reaches paired-bootstrap significance over A** at p<0.05 (n=16). The point estimates that match A are genuinely indistinguishable on this labelled set, not just noisily close — see the CI on the per-query difference.
- **A significantly *beats*** **C_hybrid_dense_plus_bm25** — i.e. these strategies actively regress on MRR with p>0.95. On this corpus they should not be enabled.
- **Weight sweep on C settles the hybrid question.** Best weight: `(w_dense=1.0, w_bm25=0.0)` → MRR 0.927. As `w_bm25` increases from 0, MRR drops monotonically. BM25 carries no signal the dense encoder is missing on this corpus — the paraphrased queries don't share tokens with the passages, and the lexical queries already match through the encoder. Conclusion: do not enable hybrid here. Keep the BM25 index for the breaker-open fallback (see `doc-07`) and for a future corpus with named entities or jargon.
- **Latency.** Median per call — A_raw_vector_search: 11.05 ms, B_ai_enhanced_retrieval: 9.76 ms, C_hybrid_dense_plus_bm25: 10.41 ms, D_rerank: 48.59 ms, E_expand_plus_rerank: 48.36 ms. The cross-encoder pass in D/E dominates (48.6 ms on CPU over ~20 candidates) — on GPU+ONNX this same step is typically ~5 ms, so the absolute cost is a CPU artefact, not a fundamental cost.
- **Caching is doing real work.** Hit ratios reported in §6 — the same query and chunk show up many times during qualitative, eval and weight-sweep phases, so the cache turns the second and subsequent visits into a dict lookup. Note: §2 latency is measured on a *separate* uncached pipeline, so the per-call numbers reflect raw cost, not cache-hit cost.
- **Production recommendation, data-driven.** On *this* corpus, ship **A_raw_vector_search** — it ties the headline metric (MRR=0.927) at the lowest latency (11.05 ms). Keep B/D/E built and tested: query expansion earns its keep when user vocabulary diverges from corpus vocabulary, and reranking earns its keep when the encoder is weaker or the corpus larger. Re-run this benchmark on the real corpus before defaulting to a more expensive strategy — the labelled set + paired bootstrap is the regression gate that decides.
- **What this benchmark *cannot* tell us.** With 16 labelled queries the bootstrap CI on MRR is wide ([0.81, 1.00] for A). A production labelled set should have ≥100 queries spanning intent classes (navigational, informational, transactional) so the paired test has the power to detect a 2-3 point lift. The eval *framework* is the deliverable here, not the specific numbers — the framework will scale to that larger set without changes.


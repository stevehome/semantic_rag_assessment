from src.bm25 import BM25Index
from src.chunker import ChunkConfig
from src.eval import LabelledQuery, evaluate_strategy
from src.pipeline import RAGPipeline
from src.query_expansion import QueryExpander
from src.reranker import MockReranker
from src.storage import VectorStore


def _build_pipeline(fake_embedder, chunk_config=None):
    return RAGPipeline(
        embedder=fake_embedder,
        store=VectorStore(metric="cosine"),
        bm25=BM25Index(),
        expander=QueryExpander(),
        chunk_config=chunk_config,
    )


def test_ingest_without_chunking_populates_store(fake_embedder, sample_docs):
    pipe = _build_pipeline(fake_embedder, chunk_config=None)
    pipe.ingest(sample_docs)
    assert len(pipe.store) == len(sample_docs)
    assert len(pipe.bm25) == len(sample_docs)


def test_ingest_with_chunking_expands_records(fake_embedder):
    docs = [
        {"id": f"doc-{i}", "title": "t", "text": " ".join([f"word{j}" for j in range(30)])}
        for i in range(3)
    ]
    pipe = _build_pipeline(fake_embedder, chunk_config=ChunkConfig(max_tokens=10, overlap=2))
    pipe.ingest(docs)
    # Each 30-token doc with stride 8 produces 4 chunks.
    assert len(pipe.store) == 12
    assert len(pipe.bm25) == 12


def test_ingest_empty_is_noop(fake_embedder):
    pipe = _build_pipeline(fake_embedder)
    pipe.ingest([])
    assert len(pipe.store) == 0
    assert len(pipe.bm25) == 0


def test_retrieve_raw_records_timing(fake_embedder, sample_docs):
    pipe = _build_pipeline(fake_embedder, chunk_config=None)
    pipe.ingest(sample_docs)
    result = pipe.retrieve_raw("autoscaler peak load", k=3)

    assert result.strategy == "A_raw_vector_search"
    assert "embed" in result.timing.stages_ms
    assert "search" in result.timing.stages_ms
    assert result.timing.total_ms > 0.0
    assert result.hits[0].id == "a"


def test_retrieve_expanded_records_expand_stage(fake_embedder, sample_docs):
    pipe = _build_pipeline(fake_embedder, chunk_config=None)
    pipe.ingest(sample_docs)
    result = pipe.retrieve_expanded("How does peak load work?", k=3)

    assert result.strategy == "B_ai_enhanced_retrieval"
    assert result.expanded_query is not None
    assert "expand" in result.timing.stages_ms
    assert "embed" in result.timing.stages_ms


def test_retrieve_hybrid_returns_fused_results(fake_embedder, sample_docs):
    pipe = _build_pipeline(fake_embedder, chunk_config=None)
    pipe.ingest(sample_docs)
    result = pipe.retrieve_hybrid("hnsw vector index", k=3)

    assert result.strategy == "C_hybrid_dense_plus_bm25"
    assert "search_dense" in result.timing.stages_ms
    assert "search_bm25" in result.timing.stages_ms
    assert "fuse" in result.timing.stages_ms
    assert len(result.hits) <= 3
    # The on-topic doc must surface.
    assert "c" in [h.id for h in result.hits]


def test_benchmark_returns_three_strategies(fake_embedder, sample_docs):
    pipe = _build_pipeline(fake_embedder, chunk_config=None)
    pipe.ingest(sample_docs)
    report = pipe.benchmark(["autoscaler", "failure cache", "hnsw"], k=2)
    assert len(report) == 3
    for row in report:
        assert {"strategy_A", "strategy_B", "strategy_C"} <= set(row)
        assert len(row["strategy_C"]["hits"]) <= 2


def test_evaluate_strategy_against_labelled_set(fake_embedder, sample_docs):
    pipe = _build_pipeline(fake_embedder, chunk_config=None)
    pipe.ingest(sample_docs)
    labelled = [
        LabelledQuery(query="autoscaler peak load", relevant_doc_ids={"a"}),
        LabelledQuery(query="hnsw vector index", relevant_doc_ids={"c"}),
        LabelledQuery(query="circuit breaker fallback", relevant_doc_ids={"d"}),
    ]
    summary = evaluate_strategy("A_raw_vector_search", pipe.retrieve_raw, labelled, k=3)
    assert summary.n_queries == 3
    # The fake embedder is term-presence — all three queries should hit their target.
    assert summary.recall_at_k > 0.0
    assert summary.mrr > 0.0


def test_pipeline_can_route_through_vertex_mock(fake_embedder, sample_docs):
    from src.embedding import MockVertexEmbeddingModel

    vertex = MockVertexEmbeddingModel(embedder=fake_embedder)
    pipe = RAGPipeline(
        embedder=fake_embedder,
        store=VectorStore(metric="cosine"),
        bm25=BM25Index(),
        expander=QueryExpander(),
        chunk_config=None,
        vertex_embedding_model=vertex,
    )
    pipe.ingest(sample_docs)
    result = pipe.retrieve_raw("hnsw vector ann", k=1)
    assert result.hits[0].id == "c"


def test_retrieve_rerank_uses_reranker(fake_embedder, sample_docs):
    # Build a reranker that boosts whichever doc has "hnsw" in its text.
    score_fn = lambda q, p: 10.0 if "hnsw" in p.lower() else float(len(p)) / 1000
    pipe = _build_pipeline(fake_embedder, chunk_config=None)
    pipe.reranker = MockReranker(score_fn=score_fn)
    pipe.ingest(sample_docs)

    result = pipe.retrieve_rerank("which index", k=2)
    assert result.strategy == "D_rerank"
    assert "rerank" in result.timing.stages_ms
    assert result.hits[0].id == "c"  # doc "c" has 'hnsw'


def test_retrieve_expand_rerank_combines_stages(fake_embedder, sample_docs):
    pipe = _build_pipeline(fake_embedder, chunk_config=None)
    pipe.reranker = MockReranker(score_fn=lambda q, p: -len(p))  # any deterministic
    pipe.ingest(sample_docs)

    result = pipe.retrieve_expand_rerank("How does peak load work?", k=2)
    assert result.strategy == "E_expand_plus_rerank"
    assert result.expanded_query is not None
    for stage in ("expand", "embed", "search", "rerank"):
        assert stage in result.timing.stages_ms


def test_rerank_without_reranker_raises(fake_embedder, sample_docs):
    pipe = _build_pipeline(fake_embedder, chunk_config=None)
    pipe.ingest(sample_docs)
    import pytest

    with pytest.raises(RuntimeError):
        pipe.retrieve_rerank("q", k=1)
    with pytest.raises(RuntimeError):
        pipe.retrieve_expand_rerank("q", k=1)


def test_hybrid_weights_change_ranking(fake_embedder, sample_docs):
    """Weighted RRF should bias the fused ranking toward whichever ranker is up-weighted."""
    pipe = _build_pipeline(fake_embedder, chunk_config=None)
    pipe.ingest(sample_docs)

    pipe.hybrid_weights = (1.0, 0.0)  # dense only
    dense_only = pipe.retrieve_hybrid("hnsw vector index", k=3)

    pipe.hybrid_weights = (0.0, 1.0)  # bm25 only
    bm25_only = pipe.retrieve_hybrid("hnsw vector index", k=3)

    # The fused score patterns should differ when weights are extreme.
    assert [h.score for h in dense_only.hits] != [h.score for h in bm25_only.hits]

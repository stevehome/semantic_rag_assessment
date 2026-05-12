from src.bm25 import BM25Index
from src.pipeline import RAGPipeline
from src.persistence import load_pipeline_indexes, save_pipeline
from src.query_expansion import QueryExpander
from src.storage import VectorStore


def test_save_and_load_round_trip(tmp_path, fake_embedder, sample_docs):
    pipe = RAGPipeline(
        embedder=fake_embedder,
        store=VectorStore(metric="cosine"),
        bm25=BM25Index(),
        expander=QueryExpander(),
        chunk_config=None,
    )
    pipe.ingest(sample_docs)
    before = pipe.retrieve_raw("hnsw vector ann", k=2)

    save_pipeline(pipe, tmp_path)
    fresh = RAGPipeline(
        embedder=fake_embedder,
        store=VectorStore(metric="cosine"),
        bm25=BM25Index(),
        expander=QueryExpander(),
        chunk_config=None,
    )
    load_pipeline_indexes(fresh, tmp_path)

    after = fresh.retrieve_raw("hnsw vector ann", k=2)
    assert [h.id for h in before.hits] == [h.id for h in after.hits]
    # BM25 also round-tripped.
    before_bm25 = pipe.bm25.search("autoscaler", k=3)
    after_bm25 = fresh.bm25.search("autoscaler", k=3)
    assert [h.id for h in before_bm25] == [h.id for h in after_bm25]

from src.bm25 import BM25Index, reciprocal_rank_fusion
from src.storage import SearchHit


def _meta(text):
    return {"text": text}


def test_bm25_ranks_by_term_overlap():
    idx = BM25Index()
    idx.add(
        ids=["a", "b", "c"],
        texts=[
            "autoscaler handles traffic spikes peak load",
            "cache redis lru invalidation",
            "vector index hnsw ann recall",
        ],
        metadata=[_meta(t) for t in [
            "autoscaler handles traffic spikes peak load",
            "cache redis lru invalidation",
            "vector index hnsw ann recall",
        ]],
    )

    hits = idx.search("peak load autoscaler", k=3)
    assert hits, "BM25 should return at least one hit on overlapping terms"
    assert hits[0].id == "a"


def test_bm25_returns_empty_when_no_term_matches():
    idx = BM25Index()
    idx.add(ids=["a"], texts=["alpha beta gamma"], metadata=[_meta("a")])
    assert idx.search("zeta", k=3) == []


def test_bm25_handles_empty_index():
    idx = BM25Index()
    assert idx.search("anything", k=3) == []


def test_bm25_idf_demotes_common_terms():
    idx = BM25Index()
    idx.add(
        ids=["a", "b", "c"],
        texts=[
            "the the the the the the alpha",
            "the the the the the the beta",
            "the the the the the the gamma",
        ],
        metadata=[_meta("x"), _meta("y"), _meta("z")],
    )
    # "the" appears in every doc — IDF is ~0 — should not break ranking.
    hits = idx.search("the alpha", k=3)
    assert hits[0].id == "a"


def test_rrf_fuses_two_rankings():
    a = [
        SearchHit(id="x", score=10.0, metadata={"src": "dense"}),
        SearchHit(id="y", score=5.0, metadata={"src": "dense"}),
    ]
    b = [
        SearchHit(id="y", score=8.0, metadata={"src": "bm25"}),
        SearchHit(id="z", score=2.0, metadata={"src": "bm25"}),
    ]
    fused = reciprocal_rank_fusion([a, b], k=3, rrf_k=60)
    fused_ids = [h.id for h in fused]
    # y appears in both at rank 2 and 1 → highest fused score.
    assert fused_ids[0] == "y"
    assert set(fused_ids) == {"x", "y", "z"}


def test_rrf_metadata_taken_from_first_surfacer():
    a = [SearchHit(id="x", score=1.0, metadata={"src": "dense"})]
    b = [SearchHit(id="x", score=1.0, metadata={"src": "bm25"})]
    [hit] = reciprocal_rank_fusion([a, b], k=1)
    assert hit.metadata == {"src": "dense"}


def test_rrf_weights_shift_winner():
    a = [SearchHit(id="x", score=1.0, metadata={}),
         SearchHit(id="y", score=1.0, metadata={})]
    b = [SearchHit(id="y", score=1.0, metadata={}),
         SearchHit(id="x", score=1.0, metadata={})]

    # Equal weights → x wins by tie-breaking on first-appearance.
    eq = reciprocal_rank_fusion([a, b], k=2, weights=[1.0, 1.0])
    # When ranker b is weighted 10x more, y should rise.
    weighted = reciprocal_rank_fusion([a, b], k=2, weights=[1.0, 10.0])
    assert weighted[0].id == "y"
    # And equal-weight fused scores are identical (sanity).
    assert eq[0].score == eq[1].score


def test_rrf_weights_length_must_match_rankings():
    import pytest

    a = [SearchHit(id="x", score=1.0, metadata={})]
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([a], k=1, weights=[1.0, 2.0])

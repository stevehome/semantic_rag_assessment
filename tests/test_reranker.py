from src.reranker import MockReranker
from src.storage import SearchHit


def _hit(id_, text):
    return SearchHit(id=id_, score=0.0, metadata={"text": text})


def test_mock_reranker_orders_by_score_fn():
    candidates = [_hit("a", "alpha"), _hit("b", "beta"), _hit("c", "gamma")]
    score_fn = lambda q, p: {"alpha": 1.0, "beta": 3.0, "gamma": 2.0}[p]
    reranker = MockReranker(score_fn=score_fn)
    out = reranker.rerank("anything", candidates, top_k=3)
    assert [h.id for h in out] == ["b", "c", "a"]
    assert out[0].score == 3.0


def test_mock_reranker_truncates_to_top_k():
    candidates = [_hit(str(i), f"t{i}") for i in range(5)]
    reranker = MockReranker(score_fn=lambda q, p: -len(p))
    out = reranker.rerank("q", candidates, top_k=2)
    assert len(out) == 2


def test_mock_reranker_handles_empty_candidates():
    reranker = MockReranker(score_fn=lambda q, p: 1.0)
    assert reranker.rerank("q", [], top_k=3) == []

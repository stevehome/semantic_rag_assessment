import math

import pytest

from src.eval import (
    LabelledQuery,
    dedupe_to_parent,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.storage import SearchHit


def _hits(*pairs):
    return [SearchHit(id=i, score=0.0, metadata={"parent_doc_id": p}) for i, p in pairs]


def test_dedupe_to_parent_keeps_first_occurrence():
    hits = _hits(("c1", "d1"), ("c2", "d1"), ("c3", "d2"))
    assert dedupe_to_parent(hits) == ["d1", "d2"]


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"a", "x"}, k=3) == 0.5
    assert recall_at_k(["a", "b"], {"c"}, k=2) == 0.0
    assert recall_at_k([], set(), k=3) == 0.0  # no relevant → 0 by convention


def test_precision_at_k():
    assert precision_at_k(["a", "b", "c"], {"a", "c"}, k=3) == pytest.approx(2 / 3)
    assert precision_at_k(["a", "b"], {"a"}, k=0) == 0.0


def test_mrr_first_hit_rank():
    assert mrr(["x", "y", "a"], {"a"}) == pytest.approx(1.0 / 3)
    assert mrr(["x", "y"], {"a"}) == 0.0
    assert mrr(["a", "b"], {"a"}) == 1.0


def test_ndcg_perfect_ranking_is_one():
    predicted = ["a", "b"]
    relevant = {"a", "b"}
    assert ndcg_at_k(predicted, relevant, k=2) == pytest.approx(1.0)


def test_ndcg_swapped_ranking_below_one():
    predicted = ["x", "a"]  # only one relevant, ranked 2nd
    relevant = {"a"}
    expected = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
    assert ndcg_at_k(predicted, relevant, k=2) == pytest.approx(expected)


def test_labelled_query_dataclass():
    lq = LabelledQuery(query="q", relevant_doc_ids={"a", "b"})
    assert lq.relevant_doc_ids == {"a", "b"}

import numpy as np
import pytest

from src.storage import VectorStore


def test_cosine_ranks_by_similarity():
    store = VectorStore(metric="cosine")
    embs = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
    store.add(["a", "b", "c"], embs, [{}, {}, {}])

    hits = store.search(np.array([1, 0, 0], dtype=np.float32), k=3)
    assert [h.id for h in hits] == ["a", "c", "b"]
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_euclidean_ranks_by_distance():
    store = VectorStore(metric="euclidean")
    embs = np.array([[0, 0], [1, 0], [5, 5]], dtype=np.float32)
    store.add(["origin", "near", "far"], embs, [{}, {}, {}])

    hits = store.search(np.array([0, 0], dtype=np.float32), k=3)
    assert [h.id for h in hits] == ["origin", "near", "far"]


def test_search_on_empty_store_returns_empty():
    store = VectorStore()
    assert store.search(np.zeros(4, dtype=np.float32), k=3) == []


def test_unsupported_metric_rejected():
    with pytest.raises(ValueError):
        VectorStore(metric="manhattan")


def test_metadata_round_trips():
    store = VectorStore()
    store.add(
        ["x"],
        np.array([[1, 0]], dtype=np.float32),
        [{"title": "T", "text": "hello"}],
    )
    [hit] = store.search(np.array([1, 0], dtype=np.float32), k=1)
    assert hit.metadata == {"title": "T", "text": "hello"}

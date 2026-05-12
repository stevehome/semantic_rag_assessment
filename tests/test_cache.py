import numpy as np

from src.cache import CachedEmbedder, CachedQueryExpander


class _CountingEmbedder:
    def __init__(self):
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        return np.asarray([[float(hash(t) % 7)] for t in texts], dtype=np.float32)


class _CountingExpander:
    def __init__(self):
        self.calls = 0

    def expand(self, q):
        self.calls += 1
        return q + " (expanded)"


def test_cached_embedder_serves_repeated_queries_from_cache():
    inner = _CountingEmbedder()
    cached = CachedEmbedder(embedder=inner, maxsize=128)

    cached.encode(["alpha", "beta"])
    cached.encode(["alpha", "beta"])
    cached.encode(["alpha", "gamma"])

    # Only 2 underlying encode batches: one for "alpha,beta", one for "gamma".
    assert inner.calls == 2
    assert cached.stats.hits == 3
    assert cached.stats.misses == 3


def test_cached_embedder_preserves_input_order():
    inner = _CountingEmbedder()
    cached = CachedEmbedder(embedder=inner)
    out1 = cached.encode(["x", "y", "z"])
    out2 = cached.encode(["z", "y", "x"])  # all cached
    # Reversed input → reversed rows.
    assert np.allclose(out1[::-1], out2)


def test_cached_embedder_evicts_when_full():
    inner = _CountingEmbedder()
    cached = CachedEmbedder(embedder=inner, maxsize=2)
    cached.encode(["a", "b", "c"])  # all miss; cache now holds 2 of them
    assert len(cached._cache) == 2


def test_cached_query_expander_hits():
    inner = _CountingExpander()
    cached = CachedQueryExpander(expander=inner)
    assert cached.expand("hello") == "hello (expanded)"
    assert cached.expand("hello") == "hello (expanded)"
    assert inner.calls == 1
    assert cached.stats.hits == 1 and cached.stats.misses == 1

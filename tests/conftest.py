"""Shared test fixtures.

Tests do not load the real sentence-transformers model — they install a
deterministic fake embedder so the suite runs in milliseconds and is
network-free.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np
import pytest


class FakeEmbedder:
    """Deterministic toy embedder.

    Token-presence hashing into a fixed-dim vector. Two passages that share
    vocabulary will have non-trivial cosine similarity, which is enough for
    retrieval correctness tests.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in t.lower().split():
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                idx = h % self.dim
                sign = 1.0 if (h >> 8) & 1 else -1.0
                out[i, idx] += sign
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def sample_docs():
    return [
        {"id": "a", "title": "scaling", "text": "autoscaler handles traffic spikes peak load"},
        {"id": "b", "title": "cache", "text": "two tier cache redis lru invalidation"},
        {"id": "c", "title": "index", "text": "hnsw vector index ann search recall"},
        {"id": "d", "title": "failure", "text": "circuit breaker fallback bm25 lexical"},
    ]

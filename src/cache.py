"""LRU caches for expensive RAG stages.

`CachedEmbedder` and `CachedQueryExpander` decorate their underlying objects.
Hit ratios are exposed via `.stats` so they can be logged or asserted on in
tests. In production, replace the dict with Memorystore / Redis — the cache
key is text → vector or text → text.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def to_dict(self) -> Dict[str, float | int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": round(self.hit_ratio, 4),
        }


@dataclass
class CachedEmbedder:
    """LRU cache around an embedder's `encode(texts) -> np.ndarray`.

    Mixes cache hits and fresh computes within a single call so the
    consumer's batch interface is preserved.
    """

    embedder: object  # duck-typed: must have .encode(list[str]) -> np.ndarray
    maxsize: int = 1024
    _cache: "OrderedDict[str, np.ndarray]" = field(default_factory=OrderedDict)
    stats: CacheStats = field(default_factory=CacheStats)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        results: List[np.ndarray | None] = [None] * len(texts)
        to_compute_idx: List[int] = []
        to_compute_texts: List[str] = []

        for i, t in enumerate(texts):
            if t in self._cache:
                self._cache.move_to_end(t)
                results[i] = self._cache[t]
                self.stats.hits += 1
            else:
                to_compute_idx.append(i)
                to_compute_texts.append(t)
                self.stats.misses += 1

        if to_compute_texts:
            fresh = self.embedder.encode(to_compute_texts)
            for slot, t, vec in zip(to_compute_idx, to_compute_texts, fresh, strict=False):
                self._cache[t] = vec
                results[slot] = vec
                while len(self._cache) > self.maxsize:
                    self._cache.popitem(last=False)

        return np.asarray(results, dtype=np.float32)


@dataclass
class CachedQueryExpander:
    """LRU cache around a `QueryExpander.expand(query) -> str`."""

    expander: object  # duck-typed: must have .expand(query) -> str
    maxsize: int = 512
    _cache: "OrderedDict[str, str]" = field(default_factory=OrderedDict)
    stats: CacheStats = field(default_factory=CacheStats)

    def expand(self, query: str) -> str:
        if query in self._cache:
            self._cache.move_to_end(query)
            self.stats.hits += 1
            return self._cache[query]
        result = self.expander.expand(query)
        self._cache[query] = result
        self.stats.misses += 1
        while len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)
        return result

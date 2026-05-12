"""Minimal Okapi BM25 index.

Implements the standard BM25 scoring formula (k1, b) over whitespace-tokenised
text. Exposes the same `add` / `search` shape as `VectorStore` so the pipeline
can compose vector and lexical hits via Reciprocal Rank Fusion.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from .storage import SearchHit


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class BM25Index:
    k1: float = 1.5
    b: float = 0.75
    _ids: List[str] = field(default_factory=list)
    _metadata: List[Dict[str, Any]] = field(default_factory=list)
    _docs: List[List[str]] = field(default_factory=list)
    _df: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _avgdl: float = 0.0

    def add(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        metadata: Sequence[Dict[str, Any]],
    ) -> None:
        if not (len(ids) == len(texts) == len(metadata)):
            raise ValueError("ids, texts and metadata must align")
        for id_, text in zip(ids, texts):
            tokens = _tokenize(text)
            self._ids.append(id_)
            self._docs.append(tokens)
            for term in set(tokens):
                self._df[term] += 1
        self._metadata.extend(metadata)
        if self._docs:
            self._avgdl = sum(len(d) for d in self._docs) / len(self._docs)

    def search(self, query: str, k: int = 3) -> List[SearchHit]:
        if not self._docs:
            return []
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        N = len(self._docs)
        results: List[SearchHit] = []
        for idx, tokens in enumerate(self._docs):
            tf = Counter(tokens)
            dl = len(tokens)
            score = 0.0
            for term in q_tokens:
                if term not in tf:
                    continue
                df = self._df.get(term, 0)
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
                f = tf[term]
                num = f * (self.k1 + 1)
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1.0))
                score += idf * num / denom
            if score > 0.0:
                results.append(
                    SearchHit(
                        id=self._ids[idx],
                        score=float(score),
                        metadata=dict(self._metadata[idx]),
                    )
                )
        results.sort(key=lambda h: -h.score)
        return results[: max(0, k)]

    def __len__(self) -> int:
        return len(self._ids)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[SearchHit]],
    k: int = 3,
    rrf_k: int = 60,
    weights: Sequence[float] | None = None,
) -> List[SearchHit]:
    """Fuse multiple ranked lists via (weighted) Reciprocal Rank Fusion.

    Score per id: ``sum_i weight_i / (rrf_k + rank_i)``. Weights default to
    1.0 for every ranker. Weighted RRF is the simplest way to bias toward a
    ranker you trust more (e.g. dense embeddings over BM25 on a paraphrased
    corpus) without having to normalise scores across rankers.

    Metadata is taken from the first ranker that surfaced the id.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError(
            f"weights ({len(weights)}) must match rankings ({len(rankings)})"
        )

    fused: Dict[str, float] = defaultdict(float)
    first_meta: Dict[str, Dict[str, Any]] = {}

    for ranking, weight in zip(rankings, weights, strict=False):
        for rank, hit in enumerate(ranking, start=1):
            fused[hit.id] += float(weight) / (rrf_k + rank)
            if hit.id not in first_meta:
                first_meta[hit.id] = hit.metadata

    ordered = sorted(fused.items(), key=lambda kv: -kv[1])
    return [
        SearchHit(id=id_, score=float(score), metadata=dict(first_meta[id_]))
        for id_, score in ordered[: max(0, k)]
    ]

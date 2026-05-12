"""Cross-encoder reranking.

A cross-encoder scores `(query, passage)` pairs jointly, so it captures
interactions a bi-encoder can't. The standard production pattern is:

    top-20 from ANN  →  cross-encoder rerank  →  top-k

which routinely recovers 5-15 nDCG points lost to ANN approximation and
bi-encoder coarseness.

`LocalCrossEncoderReranker` wraps `sentence_transformers.CrossEncoder`.
`MockReranker` lets tests inject deterministic scores. The protocol is the
same shape used by Vertex AI Discovery Engine's Ranking API
(`projects.locations.rankingConfigs.rank`) — a list of records in, a list of
records with new scores out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Protocol, Sequence

from .storage import SearchHit


class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: Sequence[SearchHit], top_k: int
    ) -> List[SearchHit]:
        ...


@dataclass
class LocalCrossEncoderReranker:
    """Production-grade reranker backed by a local cross-encoder."""

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    _model: object = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self, query: str, candidates: Sequence[SearchHit], top_k: int
    ) -> List[SearchHit]:
        if not candidates:
            return []
        model = self._ensure_model()
        pairs = [[query, c.metadata.get("text", "")] for c in candidates]
        scores = model.predict(pairs, show_progress_bar=False)
        scored = sorted(
            zip(candidates, scores, strict=False), key=lambda x: -float(x[1])
        )
        return [
            SearchHit(id=c.id, score=float(s), metadata=dict(c.metadata))
            for c, s in scored[: max(0, top_k)]
        ]


@dataclass
class MockReranker:
    """Deterministic reranker for tests.

    Takes a `score_fn(query, passage) -> float` so tests can control ordering.
    """

    score_fn: Callable[[str, str], float]

    def rerank(
        self, query: str, candidates: Sequence[SearchHit], top_k: int
    ) -> List[SearchHit]:
        if not candidates:
            return []
        scored = sorted(
            (
                (c, float(self.score_fn(query, c.metadata.get("text", ""))))
                for c in candidates
            ),
            key=lambda x: -x[1],
        )
        return [
            SearchHit(id=c.id, score=score, metadata=dict(c.metadata))
            for c, score in scored[: max(0, top_k)]
        ]

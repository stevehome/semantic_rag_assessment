"""Lightweight NumPy-backed vector store.

Chosen over FAISS/Chroma for this assessment because:
  * Zero native dependencies → runs anywhere pytest runs.
  * Brute-force search is exact, which makes correctness assertions in tests
    deterministic.
  * The public API (`add`, `search`) mirrors what an ANN index exposes, so
    swapping in FAISS/HNSW or Vertex AI Matching Engine is a one-class change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class SearchHit:
    id: str
    score: float
    metadata: Dict[str, Any]


@dataclass
class VectorStore:
    metric: str = "cosine"  # "cosine" or "euclidean"
    _ids: List[str] = field(default_factory=list)
    _metadata: List[Dict[str, Any]] = field(default_factory=list)
    _matrix: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.metric not in {"cosine", "euclidean"}:
            raise ValueError(f"Unsupported metric: {self.metric}")

    # --------------------------------------------------------------------- #
    # Mutators                                                              #
    # --------------------------------------------------------------------- #
    def add(
        self,
        ids: Sequence[str],
        embeddings: np.ndarray,
        metadata: Sequence[Dict[str, Any]],
    ) -> None:
        if len(ids) != len(embeddings) or len(ids) != len(metadata):
            raise ValueError("ids, embeddings and metadata must align")

        embeddings = np.asarray(embeddings, dtype=np.float32)
        if self.metric == "cosine":
            embeddings = _l2_normalize(embeddings)

        if self._matrix is None:
            self._matrix = embeddings
        else:
            self._matrix = np.vstack([self._matrix, embeddings])

        self._ids.extend(ids)
        self._metadata.extend(metadata)

    # --------------------------------------------------------------------- #
    # Query                                                                 #
    # --------------------------------------------------------------------- #
    def search(self, query: np.ndarray, k: int = 3) -> List[SearchHit]:
        if self._matrix is None or len(self._ids) == 0:
            return []

        query = np.asarray(query, dtype=np.float32).reshape(-1)

        if self.metric == "cosine":
            query = _l2_normalize(query[None, :])[0]
            scores = self._matrix @ query  # higher = more similar
            order = np.argsort(-scores)
        else:  # euclidean
            diff = self._matrix - query[None, :]
            scores = -np.linalg.norm(diff, axis=1)  # negate so higher = better
            order = np.argsort(-scores)

        top = order[: max(0, k)]
        return [
            SearchHit(
                id=self._ids[int(i)],
                score=float(scores[int(i)]),
                metadata=dict(self._metadata[int(i)]),
            )
            for i in top
        ]

    def __len__(self) -> int:
        return len(self._ids)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return x / norms

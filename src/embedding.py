"""Embedding layer.

Provides a `LocalEmbedder` backed by `sentence-transformers` and a
`MockVertexEmbeddingModel` that mirrors the surface of
`vertexai.language_models.TextEmbeddingModel` so production code that targets
Vertex AI can be drop-in replaced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class LocalEmbedder:
    """Local sentence-transformers wrapper.

    Lazily loads the model so importing this module is cheap (important for
    tests, which mock the model entirely).
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        model = self._ensure_model()
        vectors = model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)

    @property
    def dim(self) -> int:
        return int(self._ensure_model().get_sentence_embedding_dimension())


# --------------------------------------------------------------------------- #
# Vertex AI surface mocks                                                     #
# --------------------------------------------------------------------------- #

@dataclass
class MockTextEmbedding:
    """Shape-compatible with `vertexai.language_models.TextEmbedding`."""

    values: List[float]


class MockVertexEmbeddingModel:
    """Mock of `vertexai.language_models.TextEmbeddingModel`.

    Production code calls
        model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
        embs  = model.get_embeddings(["text"])
        vec   = embs[0].values

    This mock preserves that exact interface and delegates to a local encoder.
    """

    def __init__(self, embedder: LocalEmbedder | None = None):
        self._embedder = embedder or LocalEmbedder()

    @classmethod
    def from_pretrained(cls, model_name: str) -> "MockVertexEmbeddingModel":
        # model_name is accepted for parity but ignored — the local encoder is fixed.
        _ = model_name
        return cls()

    def get_embeddings(self, texts: Iterable[str]) -> List[MockTextEmbedding]:
        vectors = self._embedder.encode(list(texts))
        return [MockTextEmbedding(values=vec.tolist()) for vec in vectors]

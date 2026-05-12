"""On-disk persistence for the dense and lexical indexes.

Layout under `<root>/`:
  ├── vectors.npz   # matrix + ids (numpy)
  ├── metadata.json # per-chunk metadata aligned to ids
  └── bm25.json     # serialised BM25Index state

The format is intentionally simple — readable, diffable, easy to ship in a
container image. For production scale swap `vectors.npz` for the Matching
Engine index and keep `bm25.json` server-side.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .bm25 import BM25Index
from .pipeline import RAGPipeline
from .storage import VectorStore


def save_pipeline(pipeline: RAGPipeline, root: str | Path) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    matrix = pipeline.store._matrix if pipeline.store._matrix is not None else np.zeros(
        (0, 0), dtype=np.float32
    )
    np.savez(
        root / "vectors.npz",
        matrix=matrix,
        ids=np.array(pipeline.store._ids, dtype=object),
    )
    (root / "metadata.json").write_text(
        json.dumps(pipeline.store._metadata, ensure_ascii=False), encoding="utf-8"
    )

    bm25_state: Dict[str, Any] = {
        "k1": pipeline.bm25.k1,
        "b": pipeline.bm25.b,
        "ids": pipeline.bm25._ids,
        "docs": pipeline.bm25._docs,
        "df": dict(pipeline.bm25._df),
        "avgdl": pipeline.bm25._avgdl,
        "metadata": pipeline.bm25._metadata,
    }
    (root / "bm25.json").write_text(
        json.dumps(bm25_state, ensure_ascii=False), encoding="utf-8"
    )
    (root / "store_metric.txt").write_text(pipeline.store.metric, encoding="utf-8")
    return root


def load_pipeline_indexes(pipeline: RAGPipeline, root: str | Path) -> RAGPipeline:
    """Hydrate `pipeline.store` and `pipeline.bm25` from a saved root.

    The pipeline's embedder, expander and reranker are *not* persisted — they
    are caller-side concerns. This mirrors how Matching Engine vs the model
    catalog are separated in production.
    """
    root = Path(root)
    archive = np.load(root / "vectors.npz", allow_pickle=True)
    matrix = archive["matrix"]
    ids = [str(x) for x in archive["ids"].tolist()]
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    metric = (root / "store_metric.txt").read_text(encoding="utf-8").strip()

    store = VectorStore(metric=metric)
    if matrix.size > 0:
        store.add(ids=ids, embeddings=matrix.astype(np.float32), metadata=metadata)
    pipeline.store = store

    bm25_state = json.loads((root / "bm25.json").read_text(encoding="utf-8"))
    bm25 = BM25Index(k1=bm25_state["k1"], b=bm25_state["b"])
    if bm25_state["ids"]:
        bm25.add(
            ids=bm25_state["ids"],
            texts=[" ".join(toks) for toks in bm25_state["docs"]],
            metadata=bm25_state["metadata"],
        )
    pipeline.bm25 = bm25
    return pipeline

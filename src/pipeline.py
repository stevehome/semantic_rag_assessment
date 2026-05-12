"""RAG pipeline orchestration.

`RAGPipeline` owns ingestion (with optional chunking), embedding, the vector
store, the BM25 lexical index, query expansion, an optional cross-encoder
reranker, and five retrieval strategies:

  A. Raw vector search (dense only).
  B. AI-enhanced retrieval (query expansion → dense).
  C. Hybrid (dense ∪ BM25 fused with weighted Reciprocal Rank Fusion).
  D. Rerank (dense top-N → cross-encoder → top-k).
  E. Expand + rerank (B + D combined — the strongest production pattern).

Every retrieval returns a `RetrievalResult` carrying a `Timing` breakdown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

import numpy as np

from .bm25 import BM25Index, reciprocal_rank_fusion
from .chunker import ChunkConfig, chunk_documents
from .embedding import LocalEmbedder, MockVertexEmbeddingModel
from .query_expansion import QueryExpander
from .storage import SearchHit, VectorStore
from .timing import Timing


class _RerankerProto(Protocol):
    def rerank(
        self, query: str, candidates: Sequence[SearchHit], top_k: int
    ) -> List[SearchHit]:
        ...


@dataclass
class RetrievalResult:
    strategy: str
    query: str
    expanded_query: Optional[str]
    hits: List[SearchHit]
    timing: Timing = field(default_factory=Timing)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "query": self.query,
            "expanded_query": self.expanded_query,
            "timing_ms": self.timing.to_dict(),
            "hits": [
                {
                    "id": h.id,
                    "parent_doc_id": h.metadata.get("parent_doc_id", h.id),
                    "score": round(h.score, 4),
                    "title": h.metadata.get("title"),
                    "snippet": h.metadata.get("text", "")[:160],
                }
                for h in self.hits
            ],
        }


@dataclass
class RAGPipeline:
    embedder: object = field(default_factory=LocalEmbedder)  # .encode(list[str]) -> np.ndarray
    store: VectorStore = field(default_factory=lambda: VectorStore(metric="cosine"))
    bm25: BM25Index = field(default_factory=BM25Index)
    expander: object = field(default_factory=QueryExpander)  # .expand(query) -> str
    reranker: Optional[_RerankerProto] = None
    chunk_config: Optional[ChunkConfig] = field(default_factory=ChunkConfig)
    vertex_embedding_model: Optional[MockVertexEmbeddingModel] = None
    # Hybrid (Strategy C) configuration.
    hybrid_fanout: int = 10
    rrf_k: int = 60
    hybrid_weights: tuple[float, float] = (1.0, 1.0)  # (dense, bm25)
    # Rerank (Strategy D / E) configuration.
    rerank_fanout: int = 20

    # ------------------------------------------------------------------ #
    # Ingestion                                                          #
    # ------------------------------------------------------------------ #
    def ingest(self, documents: Sequence[Dict[str, Any]]) -> None:
        if not documents:
            return

        if self.chunk_config is not None:
            records = chunk_documents(documents, config=self.chunk_config)
        else:
            records = [
                {
                    "id": d["id"],
                    "parent_doc_id": d["id"],
                    "title": d.get("title", ""),
                    "text": d["text"],
                    "chunk_index": 0,
                }
                for d in documents
            ]
        if not records:
            return

        ids = [r["id"] for r in records]
        texts = [r["text"] for r in records]
        metadata = [
            {
                "title": r["title"],
                "text": r["text"],
                "parent_doc_id": r["parent_doc_id"],
                "chunk_index": r["chunk_index"],
            }
            for r in records
        ]

        embeddings = self._embed(texts)
        self.store.add(ids=ids, embeddings=embeddings, metadata=metadata)
        self.bm25.add(ids=ids, texts=texts, metadata=metadata)

    # ------------------------------------------------------------------ #
    # Retrieval strategies                                               #
    # ------------------------------------------------------------------ #
    def retrieve_raw(self, query: str, k: int = 3) -> RetrievalResult:
        timing = Timing()
        with timing.stage("embed"):
            vec = self._embed([query])[0]
        with timing.stage("search"):
            hits = self.store.search(vec, k=k)
        return RetrievalResult(
            strategy="A_raw_vector_search",
            query=query, expanded_query=None, hits=hits, timing=timing,
        )

    def retrieve_expanded(self, query: str, k: int = 3) -> RetrievalResult:
        timing = Timing()
        with timing.stage("expand"):
            expanded = self.expander.expand(query)
        with timing.stage("embed"):
            vec = self._embed([expanded])[0]
        with timing.stage("search"):
            hits = self.store.search(vec, k=k)
        return RetrievalResult(
            strategy="B_ai_enhanced_retrieval",
            query=query, expanded_query=expanded, hits=hits, timing=timing,
        )

    def retrieve_hybrid(self, query: str, k: int = 3) -> RetrievalResult:
        timing = Timing()
        with timing.stage("embed"):
            vec = self._embed([query])[0]
        with timing.stage("search_dense"):
            dense_hits = self.store.search(vec, k=self.hybrid_fanout)
        with timing.stage("search_bm25"):
            bm25_hits = self.bm25.search(query, k=self.hybrid_fanout)
        with timing.stage("fuse"):
            fused = reciprocal_rank_fusion(
                [dense_hits, bm25_hits],
                k=k,
                rrf_k=self.rrf_k,
                weights=list(self.hybrid_weights),
            )
        return RetrievalResult(
            strategy="C_hybrid_dense_plus_bm25",
            query=query, expanded_query=None, hits=fused, timing=timing,
        )

    def retrieve_rerank(self, query: str, k: int = 3) -> RetrievalResult:
        if self.reranker is None:
            raise RuntimeError("retrieve_rerank requires a reranker")
        timing = Timing()
        with timing.stage("embed"):
            vec = self._embed([query])[0]
        with timing.stage("search"):
            candidates = self.store.search(vec, k=self.rerank_fanout)
        with timing.stage("rerank"):
            hits = self.reranker.rerank(query, candidates, top_k=k)
        return RetrievalResult(
            strategy="D_rerank",
            query=query, expanded_query=None, hits=hits, timing=timing,
        )

    def retrieve_expand_rerank(self, query: str, k: int = 3) -> RetrievalResult:
        if self.reranker is None:
            raise RuntimeError("retrieve_expand_rerank requires a reranker")
        timing = Timing()
        with timing.stage("expand"):
            expanded = self.expander.expand(query)
        with timing.stage("embed"):
            vec = self._embed([expanded])[0]
        with timing.stage("search"):
            candidates = self.store.search(vec, k=self.rerank_fanout)
        # Reranker scores against the *original* user query, not the expanded
        # form — expansion helps recall, the cross-encoder scores intent.
        with timing.stage("rerank"):
            hits = self.reranker.rerank(query, candidates, top_k=k)
        return RetrievalResult(
            strategy="E_expand_plus_rerank",
            query=query, expanded_query=expanded, hits=hits, timing=timing,
        )

    # ------------------------------------------------------------------ #
    # Benchmark (qualitative comparison; quantitative eval lives in eval.py) #
    # ------------------------------------------------------------------ #
    def benchmark(self, queries: Sequence[str], k: int = 3) -> List[Dict[str, Any]]:
        report: List[Dict[str, Any]] = []
        for q in queries:
            row: Dict[str, Any] = {"query": q}
            row["strategy_A"] = self.retrieve_raw(q, k=k).to_dict()
            row["strategy_B"] = self.retrieve_expanded(q, k=k).to_dict()
            row["strategy_C"] = self.retrieve_hybrid(q, k=k).to_dict()
            if self.reranker is not None:
                row["strategy_D"] = self.retrieve_rerank(q, k=k).to_dict()
                row["strategy_E"] = self.retrieve_expand_rerank(q, k=k).to_dict()
            report.append(row)
        return report

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #
    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        if self.vertex_embedding_model is not None:
            embs = self.vertex_embedding_model.get_embeddings(list(texts))
            return np.asarray([e.values for e in embs], dtype=np.float32)
        return self.embedder.encode(list(texts))

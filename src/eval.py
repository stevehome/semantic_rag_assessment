"""Retrieval evaluation metrics.

Operates on parent-document granularity: predictions are deduplicated to
parent_doc_id before scoring, so the same metric is comparable across
chunk-level and document-level retrieval.

`evaluate_strategy` returns both the aggregate `EvalSummary` *and* the
per-query metric arrays, so paired bootstrap significance tests
(`src/significance.py`) can compare strategies head-to-head.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Sequence, Set

from .pipeline import RetrievalResult


@dataclass
class LabelledQuery:
    query: str
    relevant_doc_ids: Set[str]


def dedupe_to_parent(hits: Iterable) -> List[str]:
    seen: List[str] = []
    seen_set: Set[str] = set()
    for h in hits:
        parent = h.metadata.get("parent_doc_id") or h.id
        if parent not in seen_set:
            seen.append(parent)
            seen_set.add(parent)
    return seen


def recall_at_k(predicted: Sequence[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for p in predicted[:k] if p in relevant) / len(relevant)


def precision_at_k(predicted: Sequence[str], relevant: Set[str], k: int) -> float:
    if k == 0:
        return 0.0
    return sum(1 for p in predicted[:k] if p in relevant) / k


def mrr(predicted: Sequence[str], relevant: Set[str]) -> float:
    for rank, p in enumerate(predicted, start=1):
        if p in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(predicted: Sequence[str], relevant: Set[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, p in enumerate(predicted[:k], start=1)
        if p in relevant
    )
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg


@dataclass
class PerQueryRow:
    query: str
    recall: float
    precision: float
    rr: float  # reciprocal rank (per-query MRR contribution)
    ndcg: float
    predicted_parents: List[str]


@dataclass
class EvalSummary:
    strategy: str
    k: int
    n_queries: int
    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float
    per_query: List[PerQueryRow] = field(default_factory=list)

    def to_dict(self) -> Dict[str, float | str | int]:
        return {
            "strategy": self.strategy,
            "k": self.k,
            "n_queries": self.n_queries,
            "recall@k": round(self.recall_at_k, 4),
            "precision@k": round(self.precision_at_k, 4),
            "MRR": round(self.mrr, 4),
            "nDCG@k": round(self.ndcg_at_k, 4),
        }

    def metric_array(self, name: str) -> List[float]:
        if name == "recall":
            return [r.recall for r in self.per_query]
        if name == "precision":
            return [r.precision for r in self.per_query]
        if name == "rr":
            return [r.rr for r in self.per_query]
        if name == "ndcg":
            return [r.ndcg for r in self.per_query]
        raise KeyError(name)


def evaluate_strategy(
    strategy_name: str,
    retrieve_fn: Callable[[str, int], RetrievalResult],
    labelled: Sequence[LabelledQuery],
    k: int = 3,
) -> EvalSummary:
    per_query: List[PerQueryRow] = []
    for lq in labelled:
        result = retrieve_fn(lq.query, k)
        predicted = dedupe_to_parent(result.hits)
        per_query.append(
            PerQueryRow(
                query=lq.query,
                recall=recall_at_k(predicted, lq.relevant_doc_ids, k),
                precision=precision_at_k(predicted, lq.relevant_doc_ids, k),
                rr=mrr(predicted, lq.relevant_doc_ids),
                ndcg=ndcg_at_k(predicted, lq.relevant_doc_ids, k),
                predicted_parents=predicted,
            )
        )
    n = len(per_query)

    def avg(getter: Callable[[PerQueryRow], float]) -> float:
        return sum(getter(r) for r in per_query) / n if n else 0.0

    return EvalSummary(
        strategy=strategy_name,
        k=k,
        n_queries=n,
        recall_at_k=avg(lambda r: r.recall),
        precision_at_k=avg(lambda r: r.precision),
        mrr=avg(lambda r: r.rr),
        ndcg_at_k=avg(lambda r: r.ndcg),
        per_query=per_query,
    )

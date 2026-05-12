"""Query-expansion layer with a mocked Vertex AI GenerativeModel.

In production this would call `vertexai.generative_models.GenerativeModel`
("gemini-1.5-pro" or similar) to rewrite a terse user query into a richer,
embedding-friendly form. For this assessment we mock the surface so the
pipeline runs offline and is deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List


# Domain-specific rewrite hints. The mocked LLM uses these to add synonyms a
# semantic encoder benefits from (e.g. "peak load" → "traffic spikes,
# autoscaling, capacity"). In production the equivalent would be a Gemini
# prompt: "Rewrite this query as a search-optimised passage."
_SYNONYM_HINTS: Dict[str, List[str]] = {
    "peak load": ["traffic spikes", "autoscaling", "high concurrency", "capacity"],
    "scale": ["horizontal scaling", "autoscaler", "throughput"],
    "slow": ["latency", "tail latency", "p95", "performance regression"],
    "fail": ["failure mode", "circuit breaker", "fallback", "graceful degradation"],
    "outage": ["failure isolation", "circuit breaker", "fallback path"],
    "fast": ["low latency", "p95 latency", "warm cache"],
    "cache": ["caching tier", "LRU", "Redis", "invalidation"],
    "search": ["vector index", "ANN", "HNSW", "semantic retrieval"],
    "embedding": ["dense encoder", "sentence encoder", "vector representation"],
    "ingest": ["chunking", "document pipeline", "metadata", "overlap window"],
    "monitor": ["observability", "tracing", "SLO", "p95 latency"],
    "start": ["cold start", "warmup probe", "JIT cache"],
}


@dataclass
class _MockPart:
    text: str


@dataclass
class _MockCandidate:
    content: "_MockContent"


@dataclass
class _MockContent:
    parts: List[_MockPart]


@dataclass
class MockGenerationResponse:
    """Shape-compatible with `vertexai.generative_models.GenerationResponse`."""

    text: str

    @property
    def candidates(self) -> List[_MockCandidate]:
        return [_MockCandidate(content=_MockContent(parts=[_MockPart(text=self.text)]))]


class MockGenerativeModel:
    """Mock of `vertexai.generative_models.GenerativeModel`.

    The real call site is:
        model = GenerativeModel("gemini-1.5-pro")
        resp  = model.generate_content(prompt)
        text  = resp.text

    We preserve that surface but synthesise the rewrite from a rule table so
    tests are deterministic and offline.
    """

    def __init__(self, model_name: str = "gemini-1.5-pro-mock"):
        self.model_name = model_name

    def generate_content(self, prompt: str) -> MockGenerationResponse:
        # Extract the user query from the prompt — we accept either the raw
        # query or a templated prompt of the form "... Query: <q>".
        query = prompt
        match = re.search(r"Query:\s*(.+)$", prompt, flags=re.IGNORECASE | re.DOTALL)
        if match:
            query = match.group(1).strip()

        rewritten = _expand(query)
        return MockGenerationResponse(text=rewritten)


def _expand(query: str) -> str:
    lowered = query.lower()
    extras: List[str] = []
    for trigger, synonyms in _SYNONYM_HINTS.items():
        if trigger in lowered:
            extras.extend(synonyms)

    # De-duplicate while preserving order.
    seen = set()
    uniq = [s for s in extras if not (s in seen or seen.add(s))]

    if not uniq:
        # No trigger hit — at least normalise: drop interrogative framing so the
        # encoder sees a declarative passage.
        return _strip_question_framing(query)

    rewritten = (
        f"{_strip_question_framing(query)}. "
        f"Related concepts: {', '.join(uniq)}."
    )
    return rewritten


def _strip_question_framing(q: str) -> str:
    q = q.strip().rstrip("?")
    q = re.sub(r"^(how does|how do|what is|what are|why does|why do|can you tell me)\s+",
               "", q, flags=re.IGNORECASE)
    q = re.sub(r"^(the system|the platform)\s+", "", q, flags=re.IGNORECASE)
    return q.strip()


class QueryExpander:
    """Thin wrapper that hides the LLM prompt template from the pipeline."""

    PROMPT_TEMPLATE = (
        "You rewrite user questions into dense passages optimised for semantic "
        "vector search. Preserve intent; add domain synonyms. Query: {query}"
    )

    def __init__(self, model: MockGenerativeModel | None = None):
        self.model = model or MockGenerativeModel()

    def expand(self, query: str) -> str:
        prompt = self.PROMPT_TEMPLATE.format(query=query)
        response = self.model.generate_content(prompt)
        return response.text

"""Token-window chunking with overlap.

Whitespace-token granularity (no `tiktoken` dependency). Each chunk carries a
`parent_doc_id` so retrieval results can be deduplicated back to source
documents for evaluation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence


@dataclass(frozen=True)
class ChunkConfig:
    max_tokens: int = 25
    overlap: int = 6

    def __post_init__(self):
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if self.overlap < 0 or self.overlap >= self.max_tokens:
            raise ValueError("overlap must satisfy 0 <= overlap < max_tokens")


_WORD_RE = re.compile(r"\S+")


def _tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text)


def chunk_document(doc: Dict[str, Any], config: ChunkConfig | None = None) -> List[Dict[str, Any]]:
    config = config or ChunkConfig()
    tokens = _tokenize(doc["text"])
    if not tokens:
        return []

    stride = config.max_tokens - config.overlap
    chunks: List[Dict[str, Any]] = []
    start = 0
    chunk_index = 0
    while start < len(tokens):
        end = min(start + config.max_tokens, len(tokens))
        text = " ".join(tokens[start:end])
        chunks.append(
            {
                "id": f"{doc['id']}::chunk-{chunk_index}",
                "parent_doc_id": doc["id"],
                "title": doc.get("title", ""),
                "text": text,
                "chunk_index": chunk_index,
            }
        )
        chunk_index += 1
        if end == len(tokens):
            break
        start += stride
    return chunks


def chunk_documents(
    docs: Sequence[Dict[str, Any]], config: ChunkConfig | None = None
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for d in docs:
        out.extend(chunk_document(d, config=config))
    return out

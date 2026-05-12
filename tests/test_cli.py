"""CLI smoke tests.

Exercise the argparse wiring without invoking the real models. The fake
embedder is patched into RAGPipeline so no network or model download is
required.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np

from src import cli


class _FakeEmbedder:
    def encode(self, texts):
        rng = np.random.default_rng(abs(hash("|".join(texts))) % (2**32))
        return rng.standard_normal((len(texts), 16)).astype(np.float32)


class _ScriptedReranker:
    def rerank(self, query, candidates, top_k):
        return list(candidates)[:top_k]


def _patched_build(with_reranker: bool):
    from src.pipeline import RAGPipeline
    from src.query_expansion import QueryExpander

    pipe = RAGPipeline(embedder=_FakeEmbedder(), expander=QueryExpander(), chunk_config=None)
    if with_reranker:
        pipe.reranker = _ScriptedReranker()
    return pipe


def test_cli_search_prints_json(capsys):
    with patch.object(cli, "_build_pipeline", _patched_build):
        cli.main(["search", "autoscaler peak load", "--strategy", "A", "--k", "2"])
    out = capsys.readouterr().out.strip()
    parsed = json.loads(out)
    assert parsed["strategy"] == "A_raw_vector_search"
    assert len(parsed["hits"]) == 2


def test_cli_ingest_reports_counts(capsys, tmp_path):
    with patch.object(cli, "_build_pipeline", _patched_build):
        cli.main(["ingest", "--save", str(tmp_path)])
    out = capsys.readouterr().out
    assert "Ingested" in out and "Saved indexes to" in out
    assert (tmp_path / "vectors.npz").exists()
    assert (tmp_path / "bm25.json").exists()


def test_cli_rejects_unknown_strategy():
    import pytest

    with pytest.raises(SystemExit):
        cli.main(["search", "q", "--strategy", "Z"])

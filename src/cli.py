"""Command-line interface.

Usage:
    python -m src.cli ingest [--save PATH]
    python -m src.cli search "<query>" [--strategy A|B|C|D|E] [--k 3] [--load PATH]
    python -m src.cli benchmark
    python -m src.cli eval

For `search`, strategies D and E require the cross-encoder reranker, which
will be downloaded on first use.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict

from .data import DOCUMENTS
from .pipeline import RAGPipeline, RetrievalResult


def _build_pipeline(with_reranker: bool) -> RAGPipeline:
    pipe = RAGPipeline()
    if with_reranker:
        from .reranker import LocalCrossEncoderReranker

        pipe.reranker = LocalCrossEncoderReranker()
    return pipe


def _strategy_fn(pipe: RAGPipeline, name: str) -> Callable[[str, int], RetrievalResult]:
    table: Dict[str, Callable[[str, int], RetrievalResult]] = {
        "A": pipe.retrieve_raw,
        "B": pipe.retrieve_expanded,
        "C": pipe.retrieve_hybrid,
        "D": pipe.retrieve_rerank,
        "E": pipe.retrieve_expand_rerank,
    }
    if name not in table:
        raise SystemExit(f"Unknown strategy: {name}")
    return table[name]


def cmd_ingest(args: argparse.Namespace) -> None:
    pipe = _build_pipeline(with_reranker=False)
    pipe.ingest(DOCUMENTS)
    print(f"Ingested {len(DOCUMENTS)} documents → "
          f"{len(pipe.store)} chunks in vector store, "
          f"{len(pipe.bm25)} chunks in BM25.")
    if args.save:
        from .persistence import save_pipeline

        root = save_pipeline(pipe, args.save)
        print(f"Saved indexes to {root}")


def cmd_search(args: argparse.Namespace) -> None:
    needs_reranker = args.strategy in {"D", "E"}
    pipe = _build_pipeline(with_reranker=needs_reranker)
    if args.load:
        from .persistence import load_pipeline_indexes

        load_pipeline_indexes(pipe, args.load)
    else:
        pipe.ingest(DOCUMENTS)

    fn = _strategy_fn(pipe, args.strategy)
    result = fn(args.query, args.k)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


def cmd_benchmark(args: argparse.Namespace) -> None:
    # Defer to main.py for the full benchmark + report.
    from main import main as run_main  # type: ignore

    run_main()


def cmd_eval(args: argparse.Namespace) -> None:
    from .data import LABELLED_QUERIES
    from .eval import LabelledQuery, evaluate_strategy

    pipe = _build_pipeline(with_reranker=True)
    pipe.ingest(DOCUMENTS)
    labelled = [
        LabelledQuery(query=q["query"], relevant_doc_ids=set(q["relevant_doc_ids"]))
        for q in LABELLED_QUERIES
    ]
    summaries = {
        "A": evaluate_strategy("A", pipe.retrieve_raw, labelled, k=args.k),
        "B": evaluate_strategy("B", pipe.retrieve_expanded, labelled, k=args.k),
        "C": evaluate_strategy("C", pipe.retrieve_hybrid, labelled, k=args.k),
        "D": evaluate_strategy("D", pipe.retrieve_rerank, labelled, k=args.k),
        "E": evaluate_strategy("E", pipe.retrieve_expand_rerank, labelled, k=args.k),
    }
    print(json.dumps(
        {name: s.to_dict() for name, s in summaries.items()},
        indent=2,
    ))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest the sample corpus")
    p_ingest.add_argument("--save", type=Path, help="Persist indexes to this directory")
    p_ingest.set_defaults(func=cmd_ingest)

    p_search = sub.add_parser("search", help="Run a single query through one strategy")
    p_search.add_argument("query")
    p_search.add_argument("--strategy", choices=list("ABCDE"), default="A")
    p_search.add_argument("--k", type=int, default=3)
    p_search.add_argument("--load", type=Path, help="Load indexes from this directory")
    p_search.set_defaults(func=cmd_search)

    p_bench = sub.add_parser("benchmark", help="Run the full benchmark + write report")
    p_bench.set_defaults(func=cmd_benchmark)

    p_eval = sub.add_parser("eval", help="Run labelled-set evaluation for all strategies")
    p_eval.add_argument("--k", type=int, default=3)
    p_eval.set_defaults(func=cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])

import argparse
import csv
import json
import re
from pathlib import Path

from scripts.eval_retrieval import read_json, write_json
from scripts.retrieval_experiments import (
    RERANKER,
    apply_reranker,
    bm25_scores,
    dense_scores,
    details_from_rankings,
    minmax,
    rankings_from_scores,
)


ROOT = Path(__file__).resolve().parents[1]
CHFP = ROOT / "data/processed/preprocessed_chunks/prep_heads.json"
FALLBACK_CHFP = ROOT / "data/processed/chunk_experiments/chunks_fixed_chars_700_100.json"
QUESTIONS = ROOT / "data/eval/gold_qa_extended.json"
RESULTS = ROOT / "results/rerank"
EMBMDL = "intfloat/multilingual-e5-small"
EXTRA_RERANKER = "cross-encoder/ms-marco-MiniLM-L6-v2"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_label(text: str) -> str:
    names = {
        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1": "mmarco",
        "cross-encoder/ms-marco-MiniLM-L6-v2": "msmarco",
    }
    return names.get(text, re.sub(r"[^a-zA-Z0-9_.-]+", "_", text))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, default=CHFP)
    parser.add_argument("--eval", type=Path, default=QUESTIONS)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--embedder", default=EMBMDL)
    parser.add_argument("--hybrid-alpha", type=float, default=0.65)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-n", nargs="+", type=int, default=[10, 15, 25, 50])
    parser.add_argument("--rerankers", nargs="+", default=[RERANKER, EXTRA_RERANKER])
    args = parser.parse_args()

    chunks_path = args.chunks if args.chunks.exists() else FALLBACK_CHFP
    chunks = read_json(chunks_path)
    examples = read_json(args.eval)

    dense = dense_scores(chunks, examples, args.embedder, args.batch_size)
    bm25 = bm25_scores(chunks, examples)
    hybrid = [
        args.hybrid_alpha * minmax(dense_row) + (1 - args.hybrid_alpha) * minmax(bm25_row)
        for dense_row, bm25_row in zip(dense, bm25)
    ]

    summaries = []
    base_summary, base_details = details_from_rankings(
        label="base",
        retriever="hybrid_dense_bm25",
        model_name=args.embedder,
        chunks=chunks,
        examples=examples,
        rankings=rankings_from_scores(hybrid),
        score_rows=hybrid,
    )
    summaries.append(base_summary)
    write_json(args.results_dir / "base_details.json", base_details)
    print(json.dumps(base_summary, ensure_ascii=False))

    for reranker in args.rerankers:
        for top_n in args.top_n:
            label = f"{safe_label(reranker)}_t{top_n}"
            rankings = apply_reranker(
                chunks=chunks,
                examples=examples,
                score_rows=hybrid,
                model_name=reranker,
                top_n=top_n,
            )
            summary, details = details_from_rankings(
                label=label,
                retriever="hybrid_dense_bm25_rerank",
                model_name=reranker,
                chunks=chunks,
                examples=examples,
                rankings=rankings,
                score_rows=None,
            )
            summary["rerank_top_n"] = top_n
            summaries.append(summary)
            write_json(args.results_dir / f"{label}_details.json", details)
            print(json.dumps(summary, ensure_ascii=False))

    write_json(args.results_dir / "summary.json", summaries)
    write_csv(args.results_dir / "summary.csv", summaries)
    print(f"сводная таблица сохранена в {args.results_dir / 'summary.csv'}")

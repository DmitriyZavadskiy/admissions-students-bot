import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from scripts.eval_retrieval import read_json, write_json
from scripts.retrieval_experiments import (
    BM25,
    details_from_rankings,
    minmax,
    prepare_for_embedder,
    rankings_from_scores,
    tokenize,
)


ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "data/eval/gold_qa_extended.json"
RESULTS = ROOT / "results/retrieval_enriched"
EMBMDL = "intfloat/multilingual-e5-small"


def field_texts(chunks: list[dict], field: str) -> list[str]:
    return [chunk.get(field) or chunk["text"] for chunk in chunks]


def dense_scores(chunks, examples, model_name, field, batch_size):
    model = SentenceTransformer(model_name)
    chunk_texts = prepare_for_embedder(model_name, field_texts(chunks, field), "passage")
    question_texts = prepare_for_embedder(model_name, [example["question"] for example in examples], "query")
    chunk_embeddings = model.encode(chunk_texts, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False)
    question_embeddings = model.encode(question_texts, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False)
    return [np.asarray(chunk_embeddings @ query_embedding) for query_embedding in question_embeddings]


def bm25_scores(chunks, examples, field):
    bm25 = BM25([tokenize(text) for text in field_texts(chunks, field)])
    return [bm25.score(tokenize(example["question"])) for example in examples]


def summarize_by_category(details: list[dict], label: str) -> list[dict]:
    groups = defaultdict(list)
    for row in details:
        groups[row.get("category") or "unknown"].append(row)
    summaries = []
    for category, rows in sorted(groups.items()):
        n = len(rows)
        summaries.append(
            {
                "label": label,
                "category": category,
                "questions": n,
                "hit_1": sum(row["hit_1"] for row in rows) / n,
                "hit_5": sum(row["hit_5"] for row in rows) / n,
                "mrr": sum(row["mrr"] for row in rows) / n,
                "map": sum(row["ap"] for row in rows) / n,
                "ndcg_5": sum(row["ndcg_5"] for row in rows) / n,
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--field", default="text")
    parser.add_argument("--label", required=True)
    parser.add_argument("--eval", type=Path, default=QUESTIONS)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--embedder", default=EMBMDL)
    parser.add_argument("--hybrid-alpha", type=float, default=0.65)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    chunks = read_json(args.chunks)
    examples = read_json(args.eval)

    dense = dense_scores(chunks, examples, args.embedder, args.field, args.batch_size)
    bm25 = bm25_scores(chunks, examples, args.field)
    hybrid = [
        args.hybrid_alpha * minmax(dense_row) + (1 - args.hybrid_alpha) * minmax(bm25_row)
        for dense_row, bm25_row in zip(dense, bm25)
    ]

    summary, details = details_from_rankings(
        label=args.label,
        retriever="hybrid_dense_bm25",
        model_name=args.embedder,
        chunks=chunks,
        examples=examples,
        rankings=rankings_from_scores(hybrid),
        score_rows=hybrid,
    )
    categories = summarize_by_category(details, args.label)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.results_dir / f"{args.label}_summary.json", summary)
    write_json(args.results_dir / f"{args.label}_details.json", details)
    write_json(args.results_dir / f"{args.label}_by_category.json", categories)

    print(json.dumps(summary, ensure_ascii=False))
    for row in categories:
        print(f"{row['category']:11} n={row['questions']:2} hit1={row['hit_1']:.3f} hit5={row['hit_5']:.3f} mrr={row['mrr']:.3f} ndcg5={row['ndcg_5']:.3f}")

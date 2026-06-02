import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder

from scripts.eval_retrieval import read_json, write_json
from scripts.retrieval_experiments import RERANKER, bm25_scores, dense_scores, minmax


ROOT = Path(__file__).resolve().parents[1]
CHFP = ROOT / "data/processed/preprocessed_chunks/prep_heads.json"
QUESTIONS = ROOT / "data/eval/gold_qa_extended.json"
RESULTS = ROOT / "results/reranker_latency"
EMBMDL = "intfloat/multilingual-e5-small"
EXTRA_RERANKER = "cross-encoder/ms-marco-MiniLM-L6-v2"


def short_name(model_name: str) -> str:
    if "mmarco-mMiniLM" in model_name:
        return "mmarco"
    if "ms-marco-MiniLM" in model_name:
        return "msmarco"
    return model_name.split("/")[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, default=CHFP)
    parser.add_argument("--eval", type=Path, default=QUESTIONS)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--embedder", default=EMBMDL)
    parser.add_argument("--hybrid-alpha", type=float, default=0.65)
    parser.add_argument("--top-n", nargs="+", type=int, default=[10, 15, 25, 50])
    parser.add_argument("--rerankers", nargs="+", default=[RERANKER, EXTRA_RERANKER])
    args = parser.parse_args()

    chunks = read_json(args.chunks)
    examples = read_json(args.eval)

    dense = dense_scores(chunks, examples, args.embedder, 32)
    bm25 = bm25_scores(chunks, examples)
    hybrid = [
        args.hybrid_alpha * minmax(dense_row) + (1 - args.hybrid_alpha) * minmax(bm25_row)
        for dense_row, bm25_row in zip(dense, bm25)
    ]
    base_rankings = [list(np.argsort(-scores)) for scores in hybrid]

    rows = []
    for reranker_name in args.rerankers:
        reranker = CrossEncoder(reranker_name)
        for top_n in args.top_n:
            warmup = base_rankings[0][:top_n]
            reranker.predict([(examples[0]["question"], chunks[idx]["text"]) for idx in warmup], show_progress_bar=False)

            latencies = []
            for example, ranking in zip(examples, base_rankings):
                candidates = ranking[:top_n]
                pairs = [(example["question"], chunks[idx]["text"]) for idx in candidates]
                start = time.perf_counter()
                reranker.predict(pairs, show_progress_bar=False)
                latencies.append((time.perf_counter() - start) * 1000.0)

            row = {
                "reranker": short_name(reranker_name),
                "top_n": top_n,
                "queries": len(latencies),
                "latency_ms_mean": round(sum(latencies) / len(latencies), 1),
                "latency_ms_p50": round(float(np.percentile(latencies, 50)), 1),
                "latency_ms_p95": round(float(np.percentile(latencies, 95)), 1),
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))

    args.results_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.results_dir / "latency.json", rows)
    with (args.results_dir / "latency.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

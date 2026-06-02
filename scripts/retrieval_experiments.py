import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from scripts.eval_retrieval import chunk_relevant, dcg, norm, read_json, write_json


ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "data/eval/gold_qa_enriched.json"
CHFP = ROOT / "data/processed/chunk_experiments/chunks_fixed_chars_700_100.json"
FALLBACK_CHFP = ROOT / "data/processed/chunks.json"
RESULTS = ROOT / "results/retrieval"

EMBMDL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDERS = [
    EMBMDL,
    "intfloat/multilingual-e5-small",
    "sentence-transformers/distiluse-base-multilingual-cased-v2",
]
RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def model_label(model_name: str) -> str:
    names = {
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": "mini",
        "intfloat/multilingual-e5-small": "e5",
        "sentence-transformers/distiluse-base-multilingual-cased-v2": "distil",
    }
    return names.get(model_name, model_name.split("/")[-1].replace("-", "_"))


def tokenize(text: str) -> list[str]:
    return [token for token in norm(text).split() if len(token) > 1]


def needs_e5_prefix(model_name: str) -> bool:
    return "multilingual-e5" in model_name.lower()


def prepare_for_embedder(model_name: str, texts: list[str], kind: str) -> list[str]:
    if not needs_e5_prefix(model_name):
        return texts
    prefix = "query: " if kind == "query" else "passage: "
    return [prefix + text for text in texts]


class BM25:
    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_count = len(documents)
        self.lengths = np.array([len(doc) for doc in documents], dtype=np.float64)
        self.avg_len = float(self.lengths.mean()) if self.doc_count else 0.0
        self.term_freqs = [Counter(doc) for doc in documents]

        doc_freq = Counter()
        for doc in documents:
            doc_freq.update(set(doc))
        self.idf = {
            term: math.log(1 + (self.doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def score(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.doc_count, dtype=np.float64)
        if not query_tokens or not self.doc_count:
            return scores

        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for idx, freqs in enumerate(self.term_freqs):
                freq = freqs.get(term, 0)
                if not freq:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * self.lengths[idx] / self.avg_len)
                scores[idx] += idf * freq * (self.k1 + 1) / denom
        return scores


def minmax(scores: np.ndarray) -> np.ndarray:
    lo = float(scores.min()) if scores.size else 0.0
    hi = float(scores.max()) if scores.size else 0.0
    if hi <= lo:
        return np.zeros_like(scores, dtype=np.float64)
    return (scores - lo) / (hi - lo)


def dense_scores(
    chunks: list[dict],
    examples: list[dict],
    model_name: str,
    batch_size: int,
) -> list[np.ndarray]:
    model = SentenceTransformer(model_name)
    chunk_texts = prepare_for_embedder(model_name, [chunk["text"] for chunk in chunks], "passage")
    question_texts = prepare_for_embedder(model_name, [example["question"] for example in examples], "query")

    chunk_embeddings = model.encode(
        chunk_texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    question_embeddings = model.encode(
        question_texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    return [np.asarray(chunk_embeddings @ query_embedding) for query_embedding in question_embeddings]


def bm25_scores(chunks: list[dict], examples: list[dict]) -> list[np.ndarray]:
    bm25 = BM25([tokenize(chunk["text"]) for chunk in chunks])
    return [bm25.score(tokenize(example["question"])) for example in examples]


def apply_reranker(
    chunks: list[dict],
    examples: list[dict],
    score_rows: list[np.ndarray],
    model_name: str,
    top_n: int,
) -> list[list[int]]:
    reranker = CrossEncoder(model_name)
    rankings = []
    for example, scores in zip(examples, score_rows):
        base_ranking = list(np.argsort(-scores))
        candidates = base_ranking[:top_n]
        pairs = [(example["question"], chunks[idx]["text"]) for idx in candidates]
        rerank_scores = reranker.predict(pairs, show_progress_bar=False)
        reranked = [
            idx
            for idx, _ in sorted(
                zip(candidates, rerank_scores),
                key=lambda item: float(item[1]),
                reverse=True,
            )
        ]
        rankings.append(reranked + base_ranking[top_n:])
    return rankings


def details_from_rankings(
    label: str,
    retriever: str,
    model_name: str,
    chunks: list[dict],
    examples: list[dict],
    rankings: list[list[int]],
    score_rows: list[np.ndarray] | None = None,
) -> tuple[dict, list[dict]]:
    details = []
    for example, ranking in zip(examples, rankings):
        relevant_by_rank = [1 if chunk_relevant(example, chunks[idx]) else 0 for idx in ranking]
        total_relevant = sum(relevant_by_rank)
        first_rank = next((idx + 1 for idx, rel in enumerate(relevant_by_rank) if rel), 0)

        precision_sum = 0.0
        relevant_seen = 0
        for rank_idx, rel in enumerate(relevant_by_rank, start=1):
            if rel:
                relevant_seen += 1
                precision_sum += relevant_seen / rank_idx

        top5_relevance = relevant_by_rank[:5]
        ideal_relevance = [1] * min(total_relevant, 5)
        ndcg5 = dcg(top5_relevance) / dcg(ideal_relevance) if ideal_relevance else 0.0

        top_chunks = []
        for idx in ranking[:5]:
            score = None
            if score_rows is not None:
                score = round(float(score_rows[len(details)][idx]), 6)
            top_chunks.append(
                {
                    "chunk_id": int(chunks[idx]["chunk_id"]),
                    "score": score,
                    "title": chunks[idx].get("title", ""),
                    "source": chunks[idx].get("source", ""),
                    "is_relevant": chunk_relevant(example, chunks[idx]),
                }
            )

        details.append(
            {
                "id": example["id"],
                "category": example.get("category", ""),
                "question": example["question"],
                "expected_doc": example.get("expected_doc", ""),
                "total_relevant": total_relevant,
                "first_relevant_rank": first_rank,
                "hit_1": int(bool(relevant_by_rank[:1] and relevant_by_rank[0])),
                "hit_5": int(any(relevant_by_rank[:5])),
                "mrr": 1 / first_rank if first_rank else 0.0,
                "ap": precision_sum / total_relevant if total_relevant else 0.0,
                "ndcg_5": ndcg5,
                "top_chunks": top_chunks,
            }
        )

    n = len(details)
    summary = {
        "label": label,
        "model": model_name,
        "retriever": retriever,
        "questions": n,
        "chunks": len(chunks),
        "hit_1": sum(row["hit_1"] for row in details) / n,
        "hit_5": sum(row["hit_5"] for row in details) / n,
        "mrr": sum(row["mrr"] for row in details) / n,
        "map": sum(row["ap"] for row in details) / n,
        "ndcg_5": sum(row["ndcg_5"] for row in details) / n,
    }
    return summary, details


def save_result(results_dir: Path, summary: dict, details: list[dict]) -> None:
    safe_label = re.sub(r"[^a-zA-Z0-9_.-]+", "_", summary["label"])
    write_json(results_dir / f"{safe_label}_summary.json", summary)
    write_json(results_dir / f"{safe_label}_details.json", details)


def rankings_from_scores(score_rows: list[np.ndarray]) -> list[list[int]]:
    return [list(np.argsort(-scores)) for scores in score_rows]


def run(args: argparse.Namespace) -> None:
    chunks_path = args.chunks if args.chunks.exists() else FALLBACK_CHFP
    chunks = read_json(chunks_path)
    examples = read_json(args.eval)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    summaries = []

    dense_cache: dict[str, list[np.ndarray]] = {}
    for embedder in args.embedders:
        label = "dense_" + model_label(embedder)
        scores = dense_scores(chunks, examples, embedder, args.batch_size)
        dense_cache[embedder] = scores
        summary, details = details_from_rankings(
            label=label,
            retriever="dense",
            model_name=embedder,
            chunks=chunks,
            examples=examples,
            rankings=rankings_from_scores(scores),
            score_rows=scores,
        )
        summaries.append(summary)
        save_result(args.results_dir, summary, details)
        print(json.dumps(summary, ensure_ascii=False))

    bm25 = bm25_scores(chunks, examples)
    summary, details = details_from_rankings(
        label="bm25",
        retriever="bm25",
        model_name="bm25",
        chunks=chunks,
        examples=examples,
        rankings=rankings_from_scores(bm25),
        score_rows=bm25,
    )
    summaries.append(summary)
    save_result(args.results_dir, summary, details)
    print(json.dumps(summary, ensure_ascii=False))

    for hybrid_embedder in args.hybrid_embedders:
        if hybrid_embedder not in dense_cache:
            dense_cache[hybrid_embedder] = dense_scores(chunks, examples, hybrid_embedder, args.batch_size)

        base_dense = dense_cache[hybrid_embedder]
        hybrid_scores = [
            args.hybrid_alpha * minmax(dense_row) + (1 - args.hybrid_alpha) * minmax(bm25_row)
            for dense_row, bm25_row in zip(base_dense, bm25)
        ]
        hybrid_label = f"hybrid_{model_label(hybrid_embedder)}"
        summary, details = details_from_rankings(
            label=hybrid_label,
            retriever="hybrid_dense_bm25",
            model_name=hybrid_embedder,
            chunks=chunks,
            examples=examples,
            rankings=rankings_from_scores(hybrid_scores),
            score_rows=hybrid_scores,
        )
        summaries.append(summary)
        save_result(args.results_dir, summary, details)
        print(json.dumps(summary, ensure_ascii=False))

        if args.rerank:
            reranked_rankings = apply_reranker(
                chunks=chunks,
                examples=examples,
                score_rows=hybrid_scores,
                model_name=args.reranker,
                top_n=args.rerank_top_n,
            )
            summary, details = details_from_rankings(
                label=f"{hybrid_label}_rerank{args.rerank_top_n}",
                retriever="hybrid_dense_bm25_rerank",
                model_name=args.reranker,
                chunks=chunks,
                examples=examples,
                rankings=reranked_rankings,
                score_rows=None,
            )
            summaries.append(summary)
            save_result(args.results_dir, summary, details)
            print(json.dumps(summary, ensure_ascii=False))

    write_json(args.results_dir / "summary.json", summaries)
    write_csv(args.results_dir / "summary.csv", summaries)
    print(f"сводная таблица сохранена в {args.results_dir / 'summary.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, default=CHFP)
    parser.add_argument("--eval", type=Path, default=QUESTIONS)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--embedders", nargs="+", default=EMBEDDERS)
    parser.add_argument("--hybrid-embedders", nargs="+", default=[EMBMDL, "intfloat/multilingual-e5-small"])
    parser.add_argument("--hybrid-alpha", type=float, default=0.65)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--reranker", default=RERANKER)
    parser.add_argument("--rerank-top-n", type=int, default=25)
    args = parser.parse_args()
    run(args)

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
CHFP = ROOT / "data/processed/chunks.json"
QUESTIONS = ROOT / "data/eval/gold_qa_enriched.json"
RESULTS = ROOT / "results"
EMBMDL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def norm(text: str) -> str:
    text = (text or "").lower().replace("ё", "е")
    text = re.sub(r"[^\w\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)
        fp.write("\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def chunk_doc_matches(example: dict, chunk: dict) -> bool:
    expected_doc = norm(example.get("expected_doc", ""))
    title_source = norm(chunk.get("title", "") + " " + chunk.get("source", ""))
    return bool(expected_doc and expected_doc in title_source)


def chunk_relevant(example: dict, chunk: dict) -> bool:
    terms = [norm(term) for term in example.get("relevant_terms", []) if norm(term)]
    if not chunk_doc_matches(example, chunk):
        return False
    if not terms:
        ids = set(example.get("expected_chunk_ids", []))
        return int(chunk.get("chunk_id", -1)) in ids
    text = norm(chunk.get("text", ""))
    return all(term in text for term in terms)


def dcg(relevance: list[int]) -> float:
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevance))


@dataclass
class EvalConfig:
    label: str
    model_name: str = EMBMDL
    top_k: int = 5
    batch_size: int = 32


def evaluate(
    chunks: list[dict],
    examples: list[dict],
    model: SentenceTransformer,
    config: EvalConfig,
) -> tuple[dict, list[dict]]:
    texts = [chunk["text"] for chunk in chunks]
    chunk_embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=config.batch_size,
        show_progress_bar=False,
    )

    questions = [example["question"] for example in examples]
    question_embeddings = model.encode(
        questions,
        normalize_embeddings=True,
        batch_size=config.batch_size,
        show_progress_bar=False,
    )

    details = []
    for example, query_embedding in zip(examples, question_embeddings):
        scores = np.asarray(chunk_embeddings @ query_embedding)
        ranking = np.argsort(-scores)
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
        for idx in ranking[: config.top_k]:
            chunk = chunks[idx]
            top_chunks.append(
                {
                    "chunk_id": int(chunk["chunk_id"]),
                    "score": round(float(scores[idx]), 6),
                    "title": chunk.get("title", ""),
                    "source": chunk.get("source", ""),
                    "is_relevant": chunk_relevant(example, chunk),
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
        "label": config.label,
        "model": config.model_name,
        "retriever": "dense_local_cosine",
        "questions": n,
        "chunks": len(chunks),
        "hit_1": sum(row["hit_1"] for row in details) / n,
        "hit_5": sum(row["hit_5"] for row in details) / n,
        "mrr": sum(row["mrr"] for row in details) / n,
        "map": sum(row["ap"] for row in details) / n,
        "ndcg_5": sum(row["ndcg_5"] for row in details) / n,
    }
    return summary, details


def evaluate_from_files(
    chunks_path: Path,
    eval_path: Path,
    label: str,
    model_name: str = EMBMDL,
    results_dir: Path = RESULTS,
) -> dict:
    chunks = read_json(chunks_path)
    examples = read_json(eval_path)
    model = SentenceTransformer(model_name)
    config = EvalConfig(label=label, model_name=model_name)
    summary, details = evaluate(chunks, examples, model, config)

    safe_label = re.sub(r"[^a-zA-Z0-9_.-]+", "_", label)
    write_json(results_dir / f"{safe_label}_summary.json", summary)
    write_json(results_dir / f"{safe_label}_details.json", details)
    write_csv(results_dir / f"{safe_label}_summary.csv", [summary])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, default=CHFP)
    parser.add_argument("--eval", type=Path, default=QUESTIONS)
    parser.add_argument("--model", default=EMBMDL)
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    args = parser.parse_args()

    summary = evaluate_from_files(
        chunks_path=args.chunks,
        eval_path=args.eval,
        label=args.label,
        model_name=args.model,
        results_dir=args.results_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))

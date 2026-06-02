import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from scripts.eval_retrieval import read_json, write_json
from scripts.generation_eval import generate, score_answer
from scripts.retrieval_experiments import BM25, minmax, prepare_for_embedder, tokenize


ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "data/eval/gold_qa_extended.json"
RESULTS = ROOT / "results/gen_enriched"
EMBMDL = "intfloat/multilingual-e5-small"
GGUF = ROOT / "models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"


def field_texts(chunks, field):
    return [chunk.get(field) or chunk["text"] for chunk in chunks]


def build_context(chunks, ranking, chunk_by_id, top_k, max_chars, use_neighbors):
    blocks = []
    used = 0
    seen = set()
    number = 0
    for idx in ranking[:top_k]:
        chunk = chunks[idx]
        ids = [int(chunk["chunk_id"])]
        if use_neighbors:
            ids += chunk.get("neighbor_ids", [])
        texts = []
        for chunk_id in ids:
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            neighbor = chunk_by_id.get(chunk_id)
            if neighbor:
                texts.append(neighbor["text"])
        if not texts:
            continue
        number += 1
        block = (
            f"Фрагмент {number}\n"
            f"Документ: {chunk.get('title', '')}\n"
            f"Источник: {chunk.get('source', '')}\n"
            f"Фрагмент:\n{' '.join(texts)}\n"
        )
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n---\n".join(blocks)


def summarize_by_category(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("category") or "unknown"].append(row)
    summaries = []
    for category, group in sorted(groups.items()):
        n = len(group)
        summaries.append(
            {
                "category": category,
                "questions": n,
                "accuracy": sum(row["accuracy"] for row in group) / n,
                "completeness": sum(row["completeness"] for row in group) / n,
                "hallucination_free": sum(row["hallucination_free"] for row in group) / n,
                "source_reference": sum(row["source_reference"] for row in group) / n,
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--field", default="index_text")
    parser.add_argument("--label", default="gen_enriched")
    parser.add_argument("--eval", type=Path, default=QUESTIONS)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--embedder", default=EMBMDL)
    parser.add_argument("--gguf", type=Path, default=GGUF)
    parser.add_argument("--hybrid-alpha", type=float, default=0.65)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-context-chars", type=int, default=6000)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--no-neighbors", action="store_true")
    args = parser.parse_args()

    chunks = read_json(args.chunks)
    examples = read_json(args.eval)
    chunk_by_id = {int(chunk["chunk_id"]): chunk for chunk in chunks}

    model = SentenceTransformer(args.embedder)
    chunk_embeddings = model.encode(
        prepare_for_embedder(args.embedder, field_texts(chunks, args.field), "passage"),
        normalize_embeddings=True, batch_size=32, show_progress_bar=False,
    )
    question_embeddings = model.encode(
        prepare_for_embedder(args.embedder, [example["question"] for example in examples], "query"),
        normalize_embeddings=True, batch_size=32, show_progress_bar=False,
    )
    bm25 = BM25([tokenize(text) for text in field_texts(chunks, args.field)])

    from llama_cpp import Llama

    llm = Llama(model_path=str(args.gguf), n_ctx=8192,
                n_threads=max(2, os.cpu_count() or 4), n_gpu_layers=-1, verbose=False)

    rows = []
    for example, query_embedding in zip(examples, question_embeddings):
        dense = np.asarray(chunk_embeddings @ query_embedding)
        sparse = bm25.score(tokenize(example["question"]))
        scores = args.hybrid_alpha * minmax(dense) + (1 - args.hybrid_alpha) * minmax(sparse)
        ranking = list(np.argsort(-scores))
        context = build_context(
            chunks, ranking, chunk_by_id, args.top_k, args.max_context_chars,
            use_neighbors=not args.no_neighbors,
        )
        answer = generate(llm, example["question"], context, max_tokens=args.max_tokens)
        scores_row = score_answer(example, answer, context)
        rows.append({
            "id": example["id"], "category": example.get("category", ""),
            "question": example["question"], "generated_answer": answer, **scores_row,
        })
        print(json.dumps({key: rows[-1][key] for key in ["id", "category", "accuracy", "completeness", "hallucination_free"]}, ensure_ascii=False))

    n = len(rows)
    summary = {
        "label": args.label, "questions": n, "neighbors": not args.no_neighbors,
        "accuracy": sum(row["accuracy"] for row in rows) / n,
        "avg_completeness": sum(row["completeness"] for row in rows) / n,
        "hallucination_free": sum(row["hallucination_free"] for row in rows) / n,
        "source_reference": sum(row["source_reference"] for row in rows) / n,
    }
    categories = summarize_by_category(rows)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.results_dir / f"{args.label}_details.json", rows)
    write_json(args.results_dir / f"{args.label}_summary.json", summary)
    write_json(args.results_dir / f"{args.label}_by_category.json", categories)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for row in categories:
        print(f"{row['category']:11} n={row['questions']:2} acc={row['accuracy']:.3f} compl={row['completeness']:.3f}")

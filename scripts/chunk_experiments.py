import argparse
import csv
import json
import re
from pathlib import Path

from sentence_transformers import SentenceTransformer

from scripts.eval_retrieval import EMBMDL, EvalConfig, evaluate, read_json, write_json


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = ROOT / "data/processed/documents.json"
QUESTIONS = ROOT / "data/eval/gold_qa_enriched.json"
CHUNKS_DIR = ROOT / "data/processed/chunk_experiments"
RESULTS = ROOT / "results/chunks"

SENTENCE_SPLIT = re.compile(r"(?<=[\.\!\?\:])\s+|\n+")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_record(cid: int, doc: dict, text: str, start: int, end: int, method: str) -> dict:
    return {
        "chunk_id": cid,
        "doc_id": doc["id"],
        "source": doc["source"],
        "title": doc["title"],
        "type": doc.get("type", "unknown"),
        "text": text,
        "start_char": start,
        "end_char": end,
        "chunk_method": method,
    }


def sentence_chunks(documents: list[dict], max_chars: int, overlap: int) -> list[dict]:
    cid = 0
    chunks = []
    for doc in documents:
        parts = [part.strip() for part in SENTENCE_SPLIT.split(doc["text"]) if part.strip()]
        buf = ""
        start = 0
        for part in parts:
            if not buf:
                buf = part
                continue

            if len(buf) + 1 + len(part) <= max_chars:
                buf += "\n" + part
                continue

            end = start + len(buf)
            chunks.append(make_record(cid, doc, buf, start, end, "sentence"))
            cid += 1

            tail = buf[-overlap:] if len(buf) > overlap else buf
            start = max(0, end - len(tail))
            buf = tail + "\n" + part

        if buf.strip():
            end = start + len(buf)
            chunks.append(make_record(cid, doc, buf, start, end, "sentence"))
            cid += 1

    return chunks


def fixed_char_chunks(documents: list[dict], max_chars: int, overlap: int) -> list[dict]:
    cid = 0
    chunks = []
    step = max(1, max_chars - overlap)
    for doc in documents:
        text = doc["text"]
        start = 0
        while start < len(text):
            end = min(len(text), start + max_chars)
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(make_record(cid, doc, chunk_text, start, end, "fixed_chars"))
                cid += 1
            if end == len(text):
                break
            start += step
    return chunks


def build_chunks(documents: list[dict], method: str, max_chars: int, overlap: int) -> list[dict]:
    if method == "sentence":
        return sentence_chunks(documents, max_chars=max_chars, overlap=overlap)
    if method == "fixed_chars":
        return fixed_char_chunks(documents, max_chars=max_chars, overlap=overlap)
    raise ValueError(f"unknown chunk method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents", type=Path, default=DOCUMENTS)
    parser.add_argument("--eval", type=Path, default=QUESTIONS)
    parser.add_argument("--model", default=EMBMDL)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--chunks-dir", type=Path, default=CHUNKS_DIR)
    parser.add_argument("--methods", nargs="+", default=["sentence", "fixed_chars"])
    args = parser.parse_args()

    configs = [(700, 100), (1000, 150), (1600, 250), (2200, 300)]
    documents = read_json(args.documents)
    examples = read_json(args.eval)
    model = SentenceTransformer(args.model)

    summaries = []
    for method in args.methods:
        for max_chars, overlap in configs:
            label = f"chunks_{method}_{max_chars}_{overlap}"
            chunks = build_chunks(documents, method=method, max_chars=max_chars, overlap=overlap)
            chunks_path = args.chunks_dir / f"{label}.json"
            write_json(chunks_path, chunks)

            summary, details = evaluate(
                chunks=chunks,
                examples=examples,
                model=model,
                config=EvalConfig(label=label, model_name=args.model),
            )
            summary.update(
                {
                    "chunk_method": method,
                    "chunk_size": max_chars,
                    "chunk_overlap": overlap,
                }
            )
            summaries.append(summary)
            write_json(args.results_dir / f"{label}_details.json", details)
            print(json.dumps(summary, ensure_ascii=False))

    write_json(args.results_dir / "summary.json", summaries)
    write_csv(args.results_dir / "summary.csv", summaries)
    print(f"сводная таблица сохранена в {args.results_dir / 'summary.csv'}")

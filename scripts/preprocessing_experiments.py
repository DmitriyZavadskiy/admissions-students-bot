import argparse
import csv
import json
import re
from pathlib import Path

from scripts.chunk_experiments import fixed_char_chunks
from scripts.eval_retrieval import read_json, write_json
from scripts.retrieval_experiments import (
    bm25_scores,
    dense_scores,
    details_from_rankings,
    minmax,
    rankings_from_scores,
)


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = ROOT / "data/processed/documents.json"
QUESTIONS = ROOT / "data/eval/gold_qa_extended.json"
CHUNKS_DIR = ROOT / "data/processed/preprocessed_chunks"
RESULTS = ROOT / "results/preprocess"
EMBMDL = "intfloat/multilingual-e5-small"


NAVIGATION_LINES = {
    "главная",
    "поиск",
    "карта сайта",
    "версия для слабовидящих",
    "личный кабинет",
    "поделиться",
    "читайте также",
    "смотрите также",
}

VARIANT_LABELS = {
    "raw": "raw",
    "whitespace_clean": "clean",
    "remove_navigation": "nav",
    "normalize_values": "norm",
    "keep_headings": "heads",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def clean_whitespace(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    result = []
    blank = False
    for line in lines:
        if not line:
            if not blank:
                result.append("")
            blank = True
            continue
        result.append(line)
        blank = False
    return "\n".join(result).strip()


def remove_navigation(text: str) -> str:
    cleaned = []
    previous = None
    for line in clean_whitespace(text).splitlines():
        normalized = line.lower().replace("ё", "е").strip(" -•")
        if normalized in NAVIGATION_LINES:
            continue
        if normalized.startswith(("http://", "https://")):
            continue
        if normalized == previous:
            continue
        cleaned.append(line)
        previous = normalized
    return "\n".join(cleaned).strip()


MONTHS = {
    "01": "января",
    "02": "февраля",
    "03": "марта",
    "04": "апреля",
    "05": "мая",
    "06": "июня",
    "07": "июля",
    "08": "августа",
    "09": "сентября",
    "10": "октября",
    "11": "ноября",
    "12": "декабря",
}


def normalize_dates(match: re.Match) -> str:
    day, month, year = match.group(1), match.group(2), match.group(3)
    month_name = MONTHS.get(month)
    if not month_name:
        return match.group(0)
    return f"{day}.{month}.{year} ({int(day)} {month_name} {year})"


def normalize_values(text: str) -> str:
    text = remove_navigation(text)
    text = re.sub(r"\b(\d{1,2})\.(\d{2})\.(\d{4})\b", normalize_dates, text)
    text = re.sub(r"(\d)\s*-\s*(\d)", r"\1-\2", text)
    text = re.sub(r"\bтыс\.\s*руб\.?\b", "тыс. руб. (тыс руб)", text, flags=re.I)
    text = re.sub(r"\bруб\.\b", "руб. (рублей)", text, flags=re.I)
    return text


def with_document_heading(doc: dict, text: str) -> str:
    heading = f"Документ: {doc['title']}\nИсточник: {doc['source']}\n"
    return f"{heading}{text}".strip()


def preprocess_documents(documents: list[dict], variant: str) -> list[dict]:
    processed = []
    for doc in documents:
        text = doc["text"]
        if variant == "raw":
            new_text = text
        elif variant == "whitespace_clean":
            new_text = clean_whitespace(text)
        elif variant == "remove_navigation":
            new_text = remove_navigation(text)
        elif variant == "normalize_values":
            new_text = normalize_values(text)
        elif variant == "keep_headings":
            new_text = with_document_heading(doc, normalize_values(text))
        else:
            raise ValueError(f"unknown preprocessing variant: {variant}")
        processed.append({**doc, "text": new_text, "preprocessing": variant})
    return processed


def evaluate_hybrid(
    chunks: list[dict],
    examples: list[dict],
    embedder: str,
    alpha: float,
    batch_size: int,
    label: str,
) -> tuple[dict, list[dict]]:
    dense = dense_scores(chunks, examples, embedder, batch_size)
    bm25 = bm25_scores(chunks, examples)
    hybrid = [
        alpha * minmax(dense_row) + (1 - alpha) * minmax(bm25_row)
        for dense_row, bm25_row in zip(dense, bm25)
    ]
    return details_from_rankings(
        label=label,
        retriever="hybrid_dense_bm25",
        model_name=embedder,
        chunks=chunks,
        examples=examples,
        rankings=rankings_from_scores(hybrid),
        score_rows=hybrid,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents", type=Path, default=DOCUMENTS)
    parser.add_argument("--eval", type=Path, default=QUESTIONS)
    parser.add_argument("--chunks-dir", type=Path, default=CHUNKS_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--embedder", default=EMBMDL)
    parser.add_argument("--hybrid-alpha", type=float, default=0.65)
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["raw", "whitespace_clean", "remove_navigation", "normalize_values", "keep_headings"],
    )
    args = parser.parse_args()

    documents = read_json(args.documents)
    examples = read_json(args.eval)
    summaries = []

    for variant in args.variants:
        preprocessed = preprocess_documents(documents, variant)
        chunks = fixed_char_chunks(preprocessed, max_chars=args.chunk_size, overlap=args.overlap)
        for chunk in chunks:
            chunk["preprocessing"] = variant

        variant_label = VARIANT_LABELS.get(variant, variant)
        label = f"prep_{variant_label}"
        chunks_path = args.chunks_dir / f"{label}.json"
        write_json(chunks_path, chunks)

        summary, details = evaluate_hybrid(
            chunks=chunks,
            examples=examples,
            embedder=args.embedder,
            alpha=args.hybrid_alpha,
            batch_size=args.batch_size,
            label=label,
        )
        summary.update(
            {
                "preprocessing": variant,
                "chunk_method": "fixed_chars",
                "chunk_size": args.chunk_size,
                "chunk_overlap": args.overlap,
                "hybrid_alpha": args.hybrid_alpha,
            }
        )
        summaries.append(summary)
        write_json(args.results_dir / f"{label}_details.json", details)
        print(json.dumps(summary, ensure_ascii=False))

    write_json(args.results_dir / "summary.json", summaries)
    write_csv(args.results_dir / "summary.csv", summaries)
    print(f"сводная таблица сохранена в {args.results_dir / 'summary.csv'}")

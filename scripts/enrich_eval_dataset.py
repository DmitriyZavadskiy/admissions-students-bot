import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_QA = ROOT / "data/eval/gold_qa.json"
SRC_CHUNKS = ROOT / "data/processed/chunks.json"
DST = ROOT / "data/eval/gold_qa_enriched.json"


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


def category_from_id(example_id: str) -> str:
    prefix = example_id.split("_", 1)[0]
    return {
        "cost": "cost",
        "dates": "dates",
        "docs": "documents",
        "dormitory": "dormitory",
        "contacts": "contacts",
    }.get(prefix, prefix or "unknown")


def relevant_terms(example: dict) -> list[str]:
    category = category_from_id(example["id"])
    answer = example.get("answer", "")

    if category == "cost":
        match = re.search(r"программе\s+(.+?)\s+в\s+москве\s+стоит\s+(\d+)", norm(answer))
        if match:
            return [match.group(1), match.group(2)]

    if category == "dates":
        return ["20 июня", "25 июля", "12 августа"]

    return [norm(answer)] if answer.strip() else []


def chunk_matches(example: dict, chunk: dict, terms: list[str]) -> bool:
    expected_doc = norm(example.get("expected_doc", ""))
    title_source = norm(chunk.get("title", "") + " " + chunk.get("source", ""))
    text = norm(chunk.get("text", ""))

    if expected_doc and expected_doc not in title_source:
        return False

    return all(norm(term) in text for term in terms if norm(term))


def enrich_examples(questions: list[dict], chunks: list[dict]) -> list[dict]:
    enriched = []
    for example in questions:
        terms = relevant_terms(example)
        matched_chunks = [
            int(chunk["chunk_id"])
            for chunk in chunks
            if chunk_matches(example, chunk, terms)
        ]

        snippets = []
        for chunk in chunks:
            if int(chunk["chunk_id"]) in matched_chunks[:3]:
                snippets.append(chunk["text"][:700].strip())

        enriched.append(
            {
                "id": example["id"],
                "category": category_from_id(example["id"]),
                "question": example["question"].strip(),
                "expected_answer": example.get("answer", "").strip(),
                "expected_doc": example.get("expected_doc", "").strip(),
                "expected_chunk_ids": matched_chunks,
                "relevant_terms": terms,
                "relevant_text": snippets,
            }
        )

    return enriched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=SRC_QA)
    parser.add_argument("--chunks", type=Path, default=SRC_CHUNKS)
    parser.add_argument("--out", type=Path, default=DST)
    args = parser.parse_args()

    enriched = enrich_examples(read_json(args.questions), read_json(args.chunks))
    write_json(args.out, enriched)

    missing = [item["id"] for item in enriched if not item["expected_chunk_ids"]]
    print(f"сохранено {len(enriched)} примеров в {args.out}")
    print(f"примеров без expected_chunk_ids: {len(missing)}")
    if missing:
        print(", ".join(missing[:20]))

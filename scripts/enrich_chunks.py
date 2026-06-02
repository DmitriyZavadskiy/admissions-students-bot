import argparse
import os
import re
import time
from pathlib import Path

from scripts.eval_retrieval import read_json, write_json


ROOT = Path(__file__).resolve().parents[1]
BASE_CHUNKS = ROOT / "data/processed/preprocessed_chunks/prep_heads.json"
DOCUMENTS = ROOT / "data/processed/documents.json"
GGUF = ROOT / "models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
OUT = ROOT / "data/processed/enriched_chunks.json"

FORM_MARKERS = {
    "ОЧНАЯ ФОРМА ОБУЧЕНИЯ": "очная",
    "ОЧНО-ЗАОЧНАЯ ФОРМА ОБУЧЕНИЯ": "очно-заочная",
    "ЗАОЧНАЯ ФОРМА ОБУЧЕНИЯ": "заочная",
}
HEADER_RE = re.compile(r"^(Направление подготовки|Специальность)\b", re.IGNORECASE)
CODE_RE = re.compile(r"\d{2}\.\d{2}\.\d{2}")
PRICE_MIN = 200

SUMMARY_SYSTEM = "Ты помощник. Кратко по-русски одним предложением опиши, о чём фрагмент. Без вступлений."
QUESTIONS_SYSTEM = (
    "Ты помощник приёмной комиссии. По фрагменту придумай 3-4 коротких вопроса "
    "на естественном языке, на которые этот фрагмент даёт ответ. "
    "Каждый вопрос с новой строки, без нумерации, без пояснений."
)


def add_neighbors(chunks: list[dict]) -> None:
    positions_by_doc: dict[str, list[int]] = {}
    for position, chunk in enumerate(chunks):
        positions_by_doc.setdefault(chunk["doc_id"], []).append(position)

    for positions in positions_by_doc.values():
        for order, position in enumerate(positions):
            neighbors = []
            if order > 0:
                neighbors.append(int(chunks[positions[order - 1]]["chunk_id"]))
            if order < len(positions) - 1:
                neighbors.append(int(chunks[positions[order + 1]]["chunk_id"]))
            chunks[position]["neighbor_ids"] = neighbors


def is_price_doc(doc: dict) -> bool:
    text = doc.get("text", "")
    return "Стоимость обучения" in text and "тыс. руб" in text


def parse_price_table(doc: dict) -> list[dict]:
    lines = [line.strip() for line in doc["text"].splitlines() if line.strip()]
    rows: list[dict] = []
    current_form = ""
    current_header = ""
    header_buffer: list[str] = []
    name_buffer: list[str] = []
    header_mode = False
    started = False

    def flush_price(price: int) -> None:
        name = re.sub(r"\s+", " ", " ".join(name_buffer)).strip()
        code_match = CODE_RE.search(current_header)
        if name and current_form:
            rows.append(
                {
                    "form": current_form,
                    "header": current_header,
                    "code": code_match.group(0) if code_match else "",
                    "program": name,
                    "price": price,
                }
            )

    for line in lines:
        if line in FORM_MARKERS:
            current_form = FORM_MARKERS[line]
            started = True
            name_buffer = []
            header_mode = False
            continue
        if not started:
            continue

        if HEADER_RE.match(line):
            header_buffer = [line]
            header_mode = True
            name_buffer = []
            continue

        is_int = line.isdigit()
        value = int(line) if is_int else None

        if header_mode:
            if is_int:
                current_header = re.sub(r"\s+", " ", " ".join(header_buffer)).strip()
                header_mode = False
                name_buffer = []
            else:
                header_buffer.append(line)
            continue

        if is_int:
            if value >= PRICE_MIN:
                flush_price(value)
                name_buffer = []
            else:
                name_buffer = []
            continue

        name_buffer.append(line)

    return rows


def price_rows_to_chunks(rows: list[dict], doc: dict, start_id: int) -> list[dict]:
    year_match = re.search(r"в\s+(\d{4})\s+году", doc["text"])
    year = year_match.group(1) if year_match else "2025"

    chunks = []
    for offset, row in enumerate(rows):
        text = (
            f"Стоимость обучения в НИУ ВШЭ (Москва), {year} год. "
            f"{row['header']}. "
            f"Образовательная программа: {row['program']}. "
            f"Стоимость одного года обучения: {row['price']} тыс. руб. "
            f"Форма обучения: {row['form']}."
        )
        chunks.append(
            {
                "chunk_id": start_id + offset,
                "doc_id": doc["id"],
                "source": doc["source"],
                "title": doc["title"],
                "type": "price_row",
                "text": text,
                "start_char": -1,
                "end_char": -1,
                "chunk_method": "price_row",
                "preprocessing": "structured_table",
                "neighbor_ids": [],
                "program": row["program"],
                "price": row["price"],
            }
        )
    return chunks


def load_llm(gguf: Path):
    from llama_cpp import Llama

    return Llama(
        model_path=str(gguf),
        n_ctx=4096,
        n_threads=max(2, os.cpu_count() or 4),
        n_gpu_layers=-1,
        verbose=False,
    )


def llm_call(llm, system: str, user: str, max_tokens: int) -> str:
    result = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        top_p=0.9,
        max_tokens=max_tokens,
    )
    return result["choices"][0]["message"]["content"].strip()


def clean_questions(raw: str) -> list[str]:
    questions = []
    for line in raw.splitlines():
        line = re.sub(r"^\s*[\d\.\)\-\*•]+\s*", "", line).strip()
        if len(line) > 5 and "?" in line:
            questions.append(line)
    return questions[:4]


def enrich_with_llm(chunks: list[dict], gguf: Path, limit: int) -> None:
    llm = load_llm(gguf)
    targets = chunks if limit <= 0 else chunks[:limit]
    start = time.perf_counter()
    for index, chunk in enumerate(targets):
        body = chunk["text"][:1200]
        summary = llm_call(llm, SUMMARY_SYSTEM, body, max_tokens=80)
        questions_raw = llm_call(llm, QUESTIONS_SYSTEM, body, max_tokens=160)
        chunk["summary"] = re.sub(r"\s+", " ", summary).strip()
        chunk["aux_questions"] = clean_questions(questions_raw)
        if (index + 1) % 25 == 0:
            print(f"{index + 1}/{len(targets)} {time.perf_counter() - start:.0f}s")


def build_index_text(chunk: dict) -> str:
    parts = []
    if chunk.get("summary"):
        parts.append(chunk["summary"])
    if chunk.get("aux_questions"):
        parts.append(" ".join(chunk["aux_questions"]))
    parts.append(chunk["text"])
    return "\n".join(parts).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-chunks", type=Path, default=BASE_CHUNKS)
    parser.add_argument("--documents", type=Path, default=DOCUMENTS)
    parser.add_argument("--gguf", type=Path, default=GGUF)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    chunks = read_json(args.base_chunks)
    documents = read_json(args.documents)

    add_neighbors(chunks)

    next_id = max(int(chunk["chunk_id"]) for chunk in chunks) + 1
    price_chunks = []
    for doc in documents:
        if not is_price_doc(doc):
            continue
        rows = parse_price_table(doc)
        doc_chunks = price_rows_to_chunks(rows, doc, next_id)
        next_id += len(doc_chunks)
        price_chunks.extend(doc_chunks)
    chunks.extend(price_chunks)

    if not args.no_llm:
        enrich_with_llm(chunks, args.gguf, args.limit)
    else:
        for chunk in chunks:
            chunk.setdefault("summary", "")
            chunk.setdefault("aux_questions", [])

    for chunk in chunks:
        chunk["index_text"] = build_index_text(chunk)

    write_json(args.out, chunks)

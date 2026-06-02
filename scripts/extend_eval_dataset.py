import argparse
import json
from pathlib import Path

from scripts.eval_retrieval import norm, read_json, write_json


ROOT = Path(__file__).resolve().parents[1]
BASE_EVAL = ROOT / "data/eval/gold_qa_enriched.json"
CHUNKS = ROOT / "data/processed/chunks.json"
OUT = ROOT / "data/eval/gold_qa_extended.json"


ADDITIONAL_EXAMPLES = [
    {
        "id": "contacts_01",
        "category": "contacts",
        "question": "Какой телефон у приемной комиссии ВШЭ в Москве?",
        "expected_answer": "Телефоны приемной комиссии: (495) 771-32-42 и (495) 916-88-44.",
        "expected_doc": "Контакты",
        "relevant_terms": ["771 32 42", "916 88 44"],
    },
    {
        "id": "contacts_02",
        "category": "contacts",
        "question": "Какой телефон у справочной НИУ ВШЭ?",
        "expected_answer": "Телефон справочной НИУ ВШЭ: +7 (495) 771-32-32.",
        "expected_doc": "Контакты",
        "relevant_terms": ["771 32 32", "справочная"],
    },
    {
        "id": "contacts_03",
        "category": "contacts",
        "question": "Какая электронная почта приемной комиссии ВШЭ в Москве?",
        "expected_answer": "Электронная почта приемной комиссии НИУ ВШЭ в Москве: abitur@hse.ru.",
        "expected_doc": "Правила_приема_бак_2026_принятые_изменения_v12.pdf",
        "relevant_terms": ["abitur", "hse", "москва"],
    },
    {
        "id": "dormitory_01",
        "category": "dormitory",
        "question": "Кому из первокурсников ВШЭ в 2025 году обещали места в общежитиях?",
        "expected_answer": "Места в общежитиях получат студенты-первокурсники, поступившие на бесплатные места в бакалавриате и специалитете Вышки в 2025 году.",
        "expected_doc": "Часто задаваемые вопросы — 2025",
        "relevant_terms": ["первокурсники", "бесплатные места", "общежитиях"],
    },
    {
        "id": "dormitory_02",
        "category": "dormitory",
        "question": "Где первокурсник получает направление на заселение в общежитие?",
        "expected_answer": "Направления на заселение выдаются автоматически через Личный кабинет абитуриента.",
        "expected_doc": "Порядок заселения в общежитие",
        "relevant_terms": ["автоматическом режиме", "личный кабинет абитуриента"],
    },
    {
        "id": "dormitory_03",
        "category": "dormitory",
        "question": "На какую почту отправлять заявку на место в общежитии?",
        "expected_answer": "Заявки на место в общежитии необходимо отправлять на почту dormcom@hse.ru.",
        "expected_doc": "Подача заявки на получение места в общежитии",
        "relevant_terms": ["dormcom", "hse"],
    },
    {
        "id": "documents_01",
        "category": "documents",
        "question": "Какими способами можно подать документы в приемную кампанию 2026 года?",
        "expected_answer": "Очно и почтой документы подать можно, а дистанционно заявление можно подать через суперсервис \"Поступление в вуз онлайн\" на Госуслугах.",
        "expected_doc": "Опубликованы правила приема на 2026 год",
        "relevant_terms": ["очно", "почтовой связи", "госуслуги"],
    },
    {
        "id": "documents_02",
        "category": "documents",
        "question": "Какой уровень образования нужен для поступления на бакалавриат или специалитет?",
        "expected_answer": "Нужно иметь среднее общее образование, среднее профессиональное образование с квалификацией или высшее образование с квалификацией.",
        "expected_doc": "Правила_приема_бак_2026_принятые_изменения_v12.pdf",
        "relevant_terms": ["среднем общем", "среднем профессиональном", "высшем образовании"],
    },
    {
        "id": "documents_03",
        "category": "documents",
        "question": "Где на сайте ВШЭ размещены документы и правила приема?",
        "expected_answer": "Документы размещены в разделе документов и правил приема, включая правила приема и документы приемной комиссии.",
        "expected_doc": "Документы и правила приема",
        "relevant_terms": ["правила приема", "документы приемной комиссии"],
    },
    {
        "id": "programs_01",
        "category": "programs",
        "question": "Какие вступительные предметы указаны для Девелопмента и городского планирования?",
        "expected_answer": "Указаны математика, русский язык и один предмет на выбор: физика, история, информатика или география.",
        "expected_doc": "Девелопмент и городское планирование",
        "relevant_terms": ["математика", "русский язык", "физика", "история", "информатика", "география"],
    },
    {
        "id": "programs_02",
        "category": "programs",
        "question": "Сколько лет длится очное обучение по Девелопменту и городскому планированию?",
        "expected_answer": "Срок получения образования при очной форме обучения составляет 5 лет.",
        "expected_doc": "Девелопмент и городское планирование",
        "relevant_terms": ["очной форме", "5 лет"],
    },
    {
        "id": "programs_03",
        "category": "programs",
        "question": "Какие специализации планируются на 3 курсе Девелопмента и городского планирования?",
        "expected_answer": "На 3 курсе планируются траектории: Девелопмент и Городское планирование.",
        "expected_doc": "Девелопмент и городское планирование",
        "relevant_terms": ["девелопмент", "городское планирование"],
    },
    {
        "id": "conditions_01",
        "category": "conditions",
        "question": "Сколько баллов можно получить за индивидуальные достижения?",
        "expected_answer": "За индивидуальные достижения можно получить до 10 баллов.",
        "expected_doc": "Девелопмент и городское планирование",
        "relevant_terms": ["индивидуальные достижения", "до 10 баллов"],
    },
    {
        "id": "conditions_02",
        "category": "conditions",
        "question": "Можно ли подать документы с баллом ЕГЭ ниже 50?",
        "expected_answer": "Нет, подать документы можно только если балл по предмету ЕГЭ 50 и выше.",
        "expected_doc": "Девелопмент и городское планирование",
        "relevant_terms": ["егэ 50", "выше"],
    },
    {
        "id": "conditions_03",
        "category": "conditions",
        "question": "Могут ли победители и призеры олимпиад поступить без вступительных испытаний?",
        "expected_answer": "Да, победители и призеры Всероссийской олимпиады школьников и некоторых олимпиад из перечня могут поступить без вступительных испытаний.",
        "expected_doc": "Девелопмент и городское планирование",
        "relevant_terms": ["победители", "призеры", "без вступительных испытаний"],
    },
]


def matching_chunks(example: dict, chunks: list[dict]) -> list[int]:
    expected_doc = norm(example["expected_doc"])
    terms = [norm(term) for term in example["relevant_terms"] if norm(term)]
    ids = []
    for chunk in chunks:
        title_source = norm(chunk.get("title", "") + " " + chunk.get("source", ""))
        text = norm(chunk.get("text", ""))
        if expected_doc and expected_doc not in title_source:
            continue
        if all(term in text for term in terms):
            ids.append(int(chunk["chunk_id"]))
    return ids


def enrich_manual_examples(chunks: list[dict]) -> list[dict]:
    enriched = []
    chunk_by_id = {int(chunk["chunk_id"]): chunk for chunk in chunks}
    for example in ADDITIONAL_EXAMPLES:
        ids = matching_chunks(example, chunks)
        relevant_text = [
            chunk_by_id[chunk_id]["text"][:900].strip()
            for chunk_id in ids[:3]
            if chunk_id in chunk_by_id
        ]
        enriched.append(
            {
                **example,
                "expected_chunk_ids": ids,
                "relevant_text": relevant_text,
            }
        )
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-eval", type=Path, default=BASE_EVAL)
    parser.add_argument("--chunks", type=Path, default=CHUNKS)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    base = read_json(args.base_eval)
    chunks = read_json(args.chunks)
    added = enrich_manual_examples(chunks)
    combined = base + added

    write_json(args.out, combined)
    missing = [row["id"] for row in added if not row["expected_chunk_ids"]]
    print(f"сохранено {len(combined)} примеров в {args.out}")
    print(f"добавлено примеров: {len(added)}")
    print(f"новых примеров без expected_chunk_ids: {len(missing)}")
    if missing:
        print(", ".join(missing))

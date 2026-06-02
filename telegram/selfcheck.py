import os
import re
import sys

from scripts.rag import RagLocalChat


CJK_RE = re.compile(r"[　-〿㐀-鿿＀-￯]")

DEFAULT_QUESTIONS = [
    "Как заселиться в общежитие?",
    "Как подать заявку на общежитие?",
    "Кому предоставляется общежитие?",
    "Сколько стоит ПМИ",
    "Сколько стоит ПАД",
    "Сколько стоит Программная инженерия",
    "Сколько стоит Экономика",
    "Какие документы нужны для поступления?",
    "Как подать документы на поступление?",
    "До какого числа подавать документы?",
    "Контакты приёмной комиссии",
    "Email приёмной комиссии",
    "Сколько баллов нужно для поступления?",
    "Что даёт БВИ?",
]


def main() -> None:
    questions = DEFAULT_QUESTIONS
    if len(sys.argv) > 1:
        with open(sys.argv[1], encoding="utf-8") as fp:
            questions = [line.strip() for line in fp if line.strip()]

    use_llm = os.environ.get("RAG_USE_LLM", "1") not in {"0", "false", "False"}
    rag = RagLocalChat(use_llm=use_llm)

    for question in questions:
        result = rag.answer(question)
        answer = result.get("answer", "")
        flags = []
        if not answer.strip():
            flags.append("ПУСТО")
        if CJK_RE.search(answer):
            flags.append("CJK")
        if re.search(r"не найден|в контексте нет|отсутствует", answer, re.IGNORECASE):
            flags.append("FALLBACK")
        category = result.get("category", "")
        top = (result.get("top_sources") or [{}])[0]
        print(f"Q [{category}]: {question}")
        print(f"A: {answer}")
        print(f"источник: {top.get('title', '')}")
        if flags:
            print(f"ФЛАГИ: {flags}")
        print("-" * 70)


main()

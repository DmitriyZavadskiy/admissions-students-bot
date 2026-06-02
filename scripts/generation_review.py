import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from scripts.eval_retrieval import read_json, write_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DETAILS = ROOT / "results/gen_llm/details.json"
RESULTS = ROOT / "results/review"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("category") or "unknown"].append(row)

    summary = []
    for category, items in sorted(groups.items()):
        n = len(items)
        summary.append(
            {
                "category": category,
                "questions": n,
                "auto_accuracy": sum(row["accuracy"] for row in items) / n,
                "auto_completeness": sum(row["completeness"] for row in items) / n,
                "auto_hallucination_free": sum(row["hallucination_free"] for row in items) / n,
                "auto_source_reference": sum(row["source_reference"] for row in items) / n,
            }
        )
    return summary


def review_rows(rows: list[dict]) -> list[dict]:
    output = []
    for row in rows:
        unsupported = row.get("unsupported_facts", [])
        needs_review = int(
            not row.get("accuracy")
            or not row.get("hallucination_free")
            or not row.get("source_reference")
        )
        output.append(
            {
                "id": row["id"],
                "category": row.get("category", ""),
                "needs_review": needs_review,
                "question": row["question"],
                "expected_answer": row.get("expected_answer", ""),
                "generated_answer": row.get("generated_answer", ""),
                "auto_accuracy": row.get("accuracy", ""),
                "auto_completeness": row.get("completeness", ""),
                "auto_hallucination_free": row.get("hallucination_free", ""),
                "auto_source_reference": row.get("source_reference", ""),
                "auto_unsupported_facts": "; ".join(map(str, unsupported)),
                "manual_accuracy_0_1": "",
                "manual_completeness_0_1": "",
                "manual_hallucination_free_0_1": "",
                "manual_source_reference_0_1": "",
                "manual_comment": "",
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--details", type=Path, default=DEFAULT_DETAILS)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    args = parser.parse_args()

    rows = read_json(args.details)
    review = review_rows(rows)
    bad = sorted(
        [row for row in review if row["needs_review"]],
        key=lambda row: (
            row["auto_accuracy"],
            row["auto_hallucination_free"],
            float(row["auto_completeness"]),
            row["category"],
            row["id"],
        ),
    )
    category_summary = summarize(rows)

    write_csv(args.results_dir / "manual_review_template.csv", review)
    write_csv(args.results_dir / "bad_answers.csv", bad)
    write_csv(args.results_dir / "summary_by_category.csv", category_summary)
    write_json(args.results_dir / "summary_by_category.json", category_summary)
    print(json.dumps(category_summary, ensure_ascii=False, indent=2))
    print(f"шаблон ручной проверки: {args.results_dir / 'manual_review_template.csv'}")
    print(f"плохие ответы: {args.results_dir / 'bad_answers.csv'}")

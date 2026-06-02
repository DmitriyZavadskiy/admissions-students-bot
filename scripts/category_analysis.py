import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from scripts.eval_retrieval import read_json, write_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DETAILS = ROOT / "results/preprocess/prep_heads_details.json"
RESULTS = ROOT / "results/category"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_by_category(details: list[dict], label: str) -> list[dict]:
    groups = defaultdict(list)
    for row in details:
        groups[row.get("category") or "unknown"].append(row)

    summaries = []
    for category, rows in sorted(groups.items()):
        n = len(rows)
        summaries.append(
            {
                "label": label,
                "category": category,
                "questions": n,
                "hit_1": sum(row["hit_1"] for row in rows) / n,
                "hit_5": sum(row["hit_5"] for row in rows) / n,
                "mrr": sum(row["mrr"] for row in rows) / n,
                "map": sum(row["ap"] for row in rows) / n,
                "ndcg_5": sum(row["ndcg_5"] for row in rows) / n,
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--details", type=Path, default=DEFAULT_DETAILS)
    parser.add_argument("--label", default=None)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    args = parser.parse_args()

    details = read_json(args.details)
    label = args.label or args.details.stem.replace("_details", "")
    rows = summarize_by_category(details, label)
    write_json(args.results_dir / f"{label}_by_category.json", rows)
    write_csv(args.results_dir / f"{label}_by_category.csv", rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))

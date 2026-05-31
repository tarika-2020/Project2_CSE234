import argparse
import csv
import json
import os
from collections import defaultdict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--gold", default="./validation_gold_schema_links.json")
    parser.add_argument("--per_question", default="./artifacts/per_question.csv")
    parser.add_argument("--questions", default="./validation_input.json")
    args = parser.parse_args()

    with open(args.predictions, encoding="utf-8") as f:
        preds = {row["question_id"]: row["schema_links"] for row in json.load(f)}
    with open(args.gold, encoding="utf-8") as f:
        gold = {row["question_id"]: row["schema_links"] for row in json.load(f)}
    with open(args.questions, encoding="utf-8") as f:
        questions = {row["question_id"]: row for row in json.load(f)}

    by_db = defaultdict(list)
    with open(args.per_question, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            qid = int(row["question_id"])
            row["question_id"] = qid
            row["f1_c"] = float(row["f1_c"])
            row["f1_t"] = float(row["f1_t"])
            by_db[row["db_id"]].append(row)

    print("Worst column-F1 questions:")
    worst = []
    for rows in by_db.values():
        worst.extend(rows)
    for row in sorted(worst, key=lambda item: (item["f1_c"], item["f1_t"]))[:10]:
        qid = row["question_id"]
        print(f"\nq{qid} db={row['db_id']} f1_t={row['f1_t']:.3f} f1_c={row['f1_c']:.3f}")
        print(f"question: {questions[qid]['question']}")
        print(f"gold: {gold[qid]}")
        print(f"pred: {preds.get(qid, {})}")

    print("\nAverage per-db metrics:")
    for db_id, rows in sorted(by_db.items()):
        avg_t = sum(row["f1_t"] for row in rows) / len(rows)
        avg_c = sum(row["f1_c"] for row in rows) / len(rows)
        print(f"{db_id}: table_f1={avg_t:.3f} column_f1={avg_c:.3f} n={len(rows)}")


if __name__ == "__main__":
    main()

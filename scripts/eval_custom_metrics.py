import argparse
import json
import os
import sys
from typing import Dict, Iterable, Set, Tuple


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from schema_linking.schema_utils import load_schema


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def undirected_edge(source_table: str, target_table: str) -> Tuple[str, str]:
    return tuple(sorted((source_table, target_table), key=str.lower))


def induced_join_edges(schema, table_links: Dict[str, Iterable[str]]) -> Set[Tuple[str, str]]:
    linked_tables = {
        schema.canonical_table(table_name) or table_name
        for table_name in table_links.keys()
    }
    edges = set()
    for rel in schema.relationships:
        if rel.source_table in linked_tables and rel.target_table in linked_tables:
            edges.add(undirected_edge(rel.source_table, rel.target_table))
    return edges


def prf(pred_set, gold_set):
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if not gold_set else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--gold", default=os.path.join(ROOT, "validation_gold_schema_links.json"))
    parser.add_argument("--questions_input", default=os.path.join(ROOT, "validation_input.json"))
    parser.add_argument("--schemas_dir", default=os.path.join(ROOT, "schemas"))
    args = parser.parse_args()

    predictions = {row["question_id"]: row["schema_links"] for row in load_json(args.predictions)}
    gold = {row["question_id"]: row["schema_links"] for row in load_json(args.gold)}
    questions = {row["question_id"]: row for row in load_json(args.questions_input)}

    schema_cache = {}
    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
    count = 0

    for question_id, question in sorted(questions.items()):
        db_id = question["db_id"]
        if db_id not in schema_cache:
            schema_cache[db_id] = load_schema(args.schemas_dir, db_id)
        schema = schema_cache[db_id]

        pred_links = predictions.get(question_id, {})
        gold_links = gold.get(question_id, {})
        pred_edges = induced_join_edges(schema, pred_links)
        gold_edges = induced_join_edges(schema, gold_links)
        precision, recall, f1 = prf(pred_edges, gold_edges)

        precision_sum += precision
        recall_sum += recall
        f1_sum += f1
        count += 1

    if count == 0:
        raise SystemExit("No questions available for custom metric evaluation.")

    print("---- Join Coverage (induced FK-edge coverage across linked tables) ----")
    print(f"  Precision_J : {precision_sum / count:.4f}")
    print(f"  Recall_J    : {recall_sum / count:.4f}")
    print(f"  F1_J        : {f1_sum / count:.4f}")


if __name__ == "__main__":
    main()

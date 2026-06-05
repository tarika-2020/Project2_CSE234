import argparse
import json
import os
import sys
from typing import Dict, List


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from schema_linking.filtering import filter_schema
from schema_linking.gating import should_use_fallback_links
from schema_linking.schema_utils import load_schema


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def to_prediction_map(rows) -> Dict[int, Dict[str, List[str]]]:
    return {row["question_id"]: row.get("schema_links", {}) for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=os.path.join(ROOT, "validation_input.json"))
    parser.add_argument("--primary_predictions", required=True)
    parser.add_argument("--fallback_predictions", required=True)
    parser.add_argument("--schemas_dir", default=os.path.join(ROOT, "schemas"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    questions = load_json(args.questions)
    primary_map = to_prediction_map(load_json(args.primary_predictions))
    fallback_map = to_prediction_map(load_json(args.fallback_predictions))

    schema_cache = {}
    composed = []
    swapped = 0

    for item in questions:
        qid = item["question_id"]
        db_id = item["db_id"]
        if db_id not in schema_cache:
            schema_cache[db_id] = load_schema(args.schemas_dir, db_id)
        schema = schema_cache[db_id]

        primary_links = primary_map.get(qid, {})
        fallback_links = fallback_map.get(qid, {})
        filtered = filter_schema(item["question"], schema)

        use_fallback = should_use_fallback_links(
            cleaned_links=primary_links,
            fallback_links=fallback_links,
            filtered=filtered,
            heuristic_links=fallback_links,
            retrieval_links={},
        )
        if use_fallback:
            swapped += 1

        composed.append(
            {
                "question_id": qid,
                "schema_links": fallback_links if use_fallback else primary_links,
            }
        )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(composed, f, indent=2, ensure_ascii=False)

    print(
        f"Wrote {len(composed)} predictions to {args.output} "
        f"with {swapped} fallback swaps"
    )


if __name__ == "__main__":
    main()

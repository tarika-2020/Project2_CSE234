import argparse
import json
import os
import sys
from typing import Dict, List


ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from schema_linking.decoding import parse_schema_links, validate_and_canonicalize_links
from schema_linking.filtering import filter_schema
from schema_linking.heuristics import heuristic_schema_links
from schema_linking.modeling import HeuristicBackend, build_generation_backend
from schema_linking.prompting import build_prompt
from schema_linking.schema_utils import load_schema


def predict_question(question: str, db_id: str, schema, backend) -> Dict[str, List[str]]:
    filtered = filter_schema(question, schema)
    prompt = build_prompt(question, db_id, schema, filtered)

    if isinstance(backend, HeuristicBackend):
        return heuristic_schema_links(question, schema, filtered)

    generation = backend.generate([prompt])[0]
    try:
        parsed = parse_schema_links(generation.text)
        cleaned = validate_and_canonicalize_links(parsed, schema)
        if cleaned:
            return cleaned
    except Exception:
        pass
    return heuristic_schema_links(question, schema, filtered)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--schemas_dir", default="./schemas")
    parser.add_argument("--adapter_dir", default="./adapter")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        items = json.load(f)

    backend = build_generation_backend(
        base_model=args.base_model,
        adapter_dir=args.adapter_dir,
        max_new_tokens=args.max_new_tokens,
    )
    schema_cache = {}
    predictions = []
    for item in items:
        db_id = item["db_id"]
        if db_id not in schema_cache:
            schema_cache[db_id] = load_schema(args.schemas_dir, db_id)
        links = predict_question(item["question"], db_id, schema_cache[db_id], backend)
        predictions.append(
            {
                "question_id": item["question_id"],
                "schema_links": links,
            }
        )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(predictions)} predictions to {args.output} using {backend.name}")


if __name__ == "__main__":
    main()

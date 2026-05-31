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
from schema_linking.retrieval import SchemaLinkRetriever, merge_predictions
from schema_linking.schema_utils import load_schema


def load_json_file(path: str):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def predict_question(question: str, db_id: str, schema, backend, retriever) -> Dict[str, List[str]]:
    filtered = filter_schema(question, schema)
    prompt = build_prompt(question, db_id, schema, filtered)
    heuristic_links = heuristic_schema_links(question, schema, filtered)
    retrieval_links = retriever.predict(question, db_id, schema, filtered) if retriever else {}

    if isinstance(backend, HeuristicBackend):
        return merge_predictions(retrieval_links, heuristic_links)

    generation = backend.generate([prompt])[0]
    try:
        parsed = parse_schema_links(generation.text)
        cleaned = validate_and_canonicalize_links(parsed, schema)
        if cleaned:
            return cleaned
    except Exception:
        pass
    return merge_predictions(retrieval_links, heuristic_links)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--schemas_dir", default="./schemas")
    parser.add_argument("--adapter_dir", default="./adapter")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--train_data", default="./train.json")
    args = parser.parse_args()

    items = load_json_file(args.input)

    backend = build_generation_backend(
        base_model=args.base_model,
        adapter_dir=args.adapter_dir,
        max_new_tokens=args.max_new_tokens,
    )
    retriever = None
    if os.path.exists(args.train_data):
        retriever = SchemaLinkRetriever.from_json(args.train_data)
    schema_cache = {}
    predictions = []
    for item in items:
        db_id = item["db_id"]
        if db_id not in schema_cache:
            schema_cache[db_id] = load_schema(args.schemas_dir, db_id)
        links = predict_question(item["question"], db_id, schema_cache[db_id], backend, retriever)
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

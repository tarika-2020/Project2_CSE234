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
from schema_linking.gating import should_use_fallback_links
from schema_linking.heuristics import heuristic_schema_links
from schema_linking.modeling import HeuristicBackend, build_generation_backend
from schema_linking.prompting import build_prompt
from schema_linking.retrieval import SchemaLinkRetriever, merge_predictions
from schema_linking.schema_utils import load_schema


def load_json_file(path: str):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def prepare_question(question: str, db_id: str, schema, retriever):
    filtered = filter_schema(question, schema)
    prompt = build_prompt(question, db_id, schema, filtered)
    heuristic_links = heuristic_schema_links(question, schema, filtered)
    retrieval_links = retriever.predict(question, db_id, schema, filtered) if retriever else {}
    fallback_links = merge_predictions(retrieval_links, heuristic_links)

    return {
        "prompt": prompt,
        "schema": schema,
        "filtered": filtered,
        "heuristic_links": heuristic_links,
        "retrieval_links": retrieval_links,
        "fallback_links": fallback_links,
    }


def finalize_prediction(prepared, generation_text: str | None) -> Dict[str, List[str]]:
    if generation_text is None:
        return prepared["fallback_links"]

    schema = prepared["schema"]
    filtered = prepared["filtered"]
    heuristic_links = prepared["heuristic_links"]
    retrieval_links = prepared["retrieval_links"]
    fallback_links = prepared["fallback_links"]

    try:
        parsed = parse_schema_links(generation_text)
        cleaned = validate_and_canonicalize_links(parsed, schema)
        if not should_use_fallback_links(
            cleaned,
            fallback_links,
            filtered,
            heuristic_links,
            retrieval_links,
        ):
            return cleaned
    except Exception:
        pass
    return fallback_links


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--schemas_dir", default="./schemas")
    parser.add_argument("--adapter_dir", default="./adapter")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--batch_size", type=int, default=8)
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
    prepared_items = []
    for item in items:
        db_id = item["db_id"]
        if db_id not in schema_cache:
            schema_cache[db_id] = load_schema(args.schemas_dir, db_id)
        prepared = prepare_question(item["question"], db_id, schema_cache[db_id], retriever)
        prepared_items.append(
            {
                "question_id": item["question_id"],
                "prepared": prepared,
            }
        )

    predictions = []
    if isinstance(backend, HeuristicBackend):
        for item in prepared_items:
            predictions.append(
                {
                    "question_id": item["question_id"],
                    "schema_links": item["prepared"]["fallback_links"],
                }
            )
    else:
        for start in range(0, len(prepared_items), max(1, args.batch_size)):
            batch_items = prepared_items[start : start + max(1, args.batch_size)]
            prompts = [item["prepared"]["prompt"] for item in batch_items]
            generations = backend.generate(prompts)
            for item, generation in zip(batch_items, generations):
                predictions.append(
                    {
                        "question_id": item["question_id"],
                        "schema_links": finalize_prediction(item["prepared"], generation.text),
                    }
                )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(predictions)} predictions to {args.output} using {backend.name}")


if __name__ == "__main__":
    main()

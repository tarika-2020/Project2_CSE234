import json
from dataclasses import dataclass
from typing import Dict, Iterable, List

from .decoding import canonical_json_dumps
from .filtering import filter_schema
from .prompting import SYSTEM_INSTRUCTION
from .schema_utils import load_schema, serialize_schema


@dataclass
class PreparedExample:
    question_id: int
    db_id: str
    prompt: str
    target: str


def _sort_links(links: Dict[str, List[str]]) -> Dict[str, List[str]]:
    return {
        table: sorted(columns, key=str.lower)
        for table, columns in sorted(links.items(), key=lambda item: item[0].lower())
    }


def prepare_examples(records: Iterable[dict], schemas_dir: str, use_filtered_schema: bool) -> List[PreparedExample]:
    schema_cache = {}
    prepared = []
    for record in records:
        db_id = record["db_id"]
        if db_id not in schema_cache:
            schema_cache[db_id] = load_schema(schemas_dir, db_id)
        schema = schema_cache[db_id]
        if use_filtered_schema:
            filtered = filter_schema(record["question"], schema)
            selected_tables = list(filtered.selected_tables)
            selected_columns = {
                table: list(columns) for table, columns in filtered.selected_columns.items()
            }
            for table, gold_columns in record["schema_links"].items():
                if table not in selected_tables:
                    selected_tables.append(table)
                selected_columns.setdefault(table, [])
                for column in gold_columns:
                    if column not in selected_columns[table]:
                        selected_columns[table].append(column)
            schema_text = serialize_schema(
                schema,
                selected_tables=selected_tables,
                selected_columns=selected_columns,
            )
        else:
            schema_text = serialize_schema(schema)

        prompt = (
            f"{SYSTEM_INSTRUCTION}\n\n"
            f"Database: {db_id}\n"
            f"Question: {record['question']}\n\n"
            f"Schema:\n{schema_text}\n\n"
            "Return only valid JSON."
        )
        target = canonical_json_dumps(_sort_links(record["schema_links"]))
        prepared.append(
            PreparedExample(
                question_id=record["question_id"],
                db_id=db_id,
                prompt=prompt,
                target=target,
            )
        )
    return prepared


def write_jsonl(path: str, examples: Iterable[PreparedExample]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for example in examples:
            row = {
                "question_id": example.question_id,
                "db_id": example.db_id,
                "prompt": example.prompt,
                "target": example.target,
                "completion": example.target,
                "text": f"{example.prompt}\n{example.target}",
                "messages": [
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {
                        "role": "user",
                        "content": example.prompt.split("\n\n", 1)[1],
                    },
                    {"role": "assistant", "content": example.target},
                ],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

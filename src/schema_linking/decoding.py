import json
from typing import Any, Dict, Optional

from .schema_utils import Schema


def extract_first_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        ch = text[index]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def parse_schema_links(text: str) -> Dict[str, Any]:
    candidate = extract_first_json_object(text.strip())
    if candidate is None:
        raise ValueError("no JSON object found")
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        repaired = candidate.replace("'", '"')
        return json.loads(repaired)


def validate_and_canonicalize_links(raw_links: Any, schema: Schema) -> Dict[str, list[str]]:
    if not isinstance(raw_links, dict):
        return {}

    cleaned: Dict[str, list[str]] = {}
    for raw_table, raw_columns in raw_links.items():
        canonical_table = schema.canonical_table(str(raw_table))
        if canonical_table is None:
            continue

        if not isinstance(raw_columns, list):
            cleaned.setdefault(canonical_table, [])
            continue

        canonical_columns = []
        seen = set()
        for raw_column in raw_columns:
            canonical_column = schema.canonical_column(canonical_table, str(raw_column))
            if canonical_column is None:
                continue
            key = canonical_column.lower()
            if key not in seen:
                seen.add(key)
                canonical_columns.append(canonical_column)
        cleaned[canonical_table] = sorted(canonical_columns, key=str.lower)

    return dict(sorted(cleaned.items(), key=lambda item: item[0].lower()))


def canonical_json_dumps(links: Dict[str, list[str]]) -> str:
    canonical = {
        table: sorted(columns, key=str.lower)
        for table, columns in sorted(links.items(), key=lambda item: item[0].lower())
    }
    return json.dumps(canonical, separators=(",", ":"), ensure_ascii=False)

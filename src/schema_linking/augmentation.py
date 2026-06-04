import copy
import re
from collections import Counter
from typing import Dict, Iterable, List


PHRASE_REPLACEMENTS = [
    (r"\bshow me\b", "list"),
    (r"\bshow\b", "list"),
    (r"\blist\b", "show"),
    (r"\bfind\b", "identify"),
    (r"\bwhat is\b", "which is"),
    (r"\bwhat are\b", "which are"),
    (r"\bhow many\b", "count"),
    (r"\bnumber of\b", "count of"),
]


def paraphrase_question(question: str) -> str:
    rewritten = question.strip()
    lowered = rewritten.lower()
    for pattern, replacement in PHRASE_REPLACEMENTS:
        updated = re.sub(pattern, replacement, lowered, count=1)
        if updated != lowered:
            rewritten = updated
            break
    else:
        rewritten = f"List the results for: {rewritten.rstrip('?')}"

    rewritten = rewritten.strip()
    if question.endswith("?") and not rewritten.endswith("?"):
        rewritten += "?"
    return rewritten[0].upper() + rewritten[1:] if rewritten else question


def _duplication_factor(record: dict, target_per_db: int) -> int:
    schema_links = record.get("schema_links", {})
    table_count = len(schema_links)
    non_empty_tables = sum(1 for columns in schema_links.values() if columns)

    factor = 1
    if table_count >= 3:
        factor += 1
    if table_count >= 5:
        factor += 1
    if non_empty_tables >= 3:
        factor += 1
    return factor


def build_augmented_records(records: Iterable[dict]) -> List[dict]:
    augmented = []
    for record in records:
        augmented.append(copy.deepcopy(record))

        paraphrased = copy.deepcopy(record)
        paraphrased["question"] = paraphrase_question(record["question"])
        paraphrased["augmentation_type"] = "paraphrase"
        paraphrased["source_question_id"] = record["question_id"]
        augmented.append(paraphrased)
    return augmented


def build_balanced_records(records: Iterable[dict], min_examples_per_db: int = 12) -> List[dict]:
    source_records = [copy.deepcopy(record) for record in records]
    counts = Counter(record["db_id"] for record in source_records)
    balanced: List[dict] = []

    for record in source_records:
        db_target = max(min_examples_per_db, counts[record["db_id"]])
        db_boost = max(1, -(-db_target // counts[record["db_id"]]))
        content_boost = _duplication_factor(record, db_target)
        repeats = max(db_boost, content_boost)

        for repeat_idx in range(repeats):
            cloned = copy.deepcopy(record)
            if repeat_idx > 0:
                cloned["sampling_type"] = "oversampled"
                cloned["source_question_id"] = record["question_id"]
                cloned["sampling_repeat_index"] = repeat_idx
            balanced.append(cloned)
    return balanced


def summarize_training_mix(records: Iterable[dict]) -> Dict[str, Dict[str, int]]:
    items = list(records)
    by_db = Counter(record["db_id"] for record in items)
    by_tag = Counter(
        record.get("augmentation_type") or record.get("sampling_type") or "original"
        for record in items
    )
    return {
        "db_counts": dict(sorted(by_db.items())),
        "record_types": dict(sorted(by_tag.items())),
    }

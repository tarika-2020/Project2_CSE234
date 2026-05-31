import json
from dataclasses import dataclass
from math import log
from typing import Dict, List, Sequence, Set

from .filtering import FilteredSchema, tokenize_text
from .schema_utils import Schema


@dataclass(frozen=True)
class RetrievalExample:
    db_id: str
    question_id: int
    question: str
    tokens: Set[str]
    schema_links: Dict[str, List[str]]


class SchemaLinkRetriever:
    def __init__(self, examples: Sequence[RetrievalExample]) -> None:
        by_db: Dict[str, List[RetrievalExample]] = {}
        table_idf_by_db: Dict[str, Dict[str, float]] = {}
        for example in examples:
            by_db.setdefault(example.db_id, []).append(example)
        for db_id, db_examples in by_db.items():
            table_counts: Dict[str, int] = {}
            for example in db_examples:
                for table in example.schema_links:
                    table_counts[table] = table_counts.get(table, 0) + 1
            total = max(1, len(db_examples))
            table_idf_by_db[db_id] = {
                table: 1.0 + log(1.0 + total / count)
                for table, count in table_counts.items()
            }
        self.by_db = by_db
        self.table_idf_by_db = table_idf_by_db

    @classmethod
    def from_json(cls, path: str) -> "SchemaLinkRetriever":
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
        examples = [
            RetrievalExample(
                db_id=row["db_id"],
                question_id=row["question_id"],
                question=row["question"],
                tokens=tokenize_text(row["question"]),
                schema_links=row["schema_links"],
            )
            for row in rows
        ]
        return cls(examples)

    def predict(
        self,
        question: str,
        db_id: str,
        schema: Schema,
        filtered: FilteredSchema,
        top_k: int = 5,
    ) -> Dict[str, List[str]]:
        examples = self.by_db.get(db_id, [])
        if not examples:
            return {}

        query_tokens = tokenize_text(question)
        scored = []
        for example in examples:
            overlap = len(query_tokens & example.tokens)
            if overlap == 0:
                continue
            union = max(1, len(query_tokens | example.tokens))
            score = overlap / union
            score += 0.03 * overlap
            scored.append((score, example))
        if not scored:
            return {}

        scored.sort(key=lambda item: (item[0], item[1].question_id), reverse=True)
        filtered_tables = set(filtered.selected_tables)
        table_idf = self.table_idf_by_db.get(db_id, {})

        table_votes: Dict[str, float] = {}
        column_votes: Dict[str, Dict[str, float]] = {}
        for score, example in scored[:top_k]:
            for table, columns in example.schema_links.items():
                if table not in filtered_tables and filtered.table_scores.get(table, 0.0) <= 0:
                    continue
                weighted_score = score * table_idf.get(table, 1.0)
                table_votes[table] = table_votes.get(table, 0.0) + weighted_score
                per_table = column_votes.setdefault(table, {})
                for column in columns:
                    per_table[column] = per_table.get(column, 0.0) + weighted_score

        if not table_votes:
            return {}

        ranked_tables = sorted(table_votes.items(), key=lambda item: (item[1], item[0]), reverse=True)
        output: Dict[str, List[str]] = {}
        best_table_score = ranked_tables[0][1]
        for table, score in ranked_tables[:4]:
            if score < max(0.18, best_table_score * 0.45):
                continue
            ranked_columns = sorted(
                column_votes.get(table, {}).items(),
                key=lambda item: (item[1], item[0]),
                reverse=True,
            )
            columns = [column for column, col_score in ranked_columns if col_score >= score * 0.18][:6]
            output[table] = sorted(set(columns), key=str.lower) if columns else []

        return output


def merge_predictions(
    retrieval_links: Dict[str, List[str]],
    heuristic_links: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    if not retrieval_links:
        return heuristic_links
    merged = {table: list(columns) for table, columns in retrieval_links.items()}
    for table, columns in heuristic_links.items():
        merged.setdefault(table, [])
        for column in columns:
            if column not in merged[table]:
                merged[table].append(column)
        merged[table] = sorted(merged[table], key=str.lower)
    return dict(sorted(merged.items(), key=lambda item: item[0].lower()))

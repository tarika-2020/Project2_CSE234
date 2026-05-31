import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set

from .schema_utils import Schema


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
STOPWORDS = {
    "a",
    "all",
    "an",
    "and",
    "are",
    "by",
    "count",
    "display",
    "find",
    "for",
    "from",
    "get",
    "give",
    "how",
    "in",
    "is",
    "list",
    "many",
    "me",
    "of",
    "on",
    "or",
    "show",
    "the",
    "their",
    "there",
    "these",
    "those",
    "to",
    "what",
    "where",
    "which",
    "who",
    "with",
}


def _split_identifier(text: str) -> List[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text.replace("_", " "))
    return TOKEN_RE.findall(spaced.lower())


def _normalize_token(token: str) -> Set[str]:
    forms = {token}
    if token.endswith("ies") and len(token) > 3:
        forms.add(token[:-3] + "y")
    if token.endswith("es") and len(token) > 2:
        forms.add(token[:-2])
    if token.endswith("s") and len(token) > 1:
        forms.add(token[:-1])
    return {form for form in forms if form}


def tokenize_text(text: str) -> Set[str]:
    base_tokens = TOKEN_RE.findall(text.lower())
    normalized = set()
    for token in base_tokens:
        if token in STOPWORDS:
            continue
        normalized.update(_normalize_token(token))
    return normalized


@dataclass
class FilteredSchema:
    selected_tables: List[str]
    selected_columns: Dict[str, List[str]]
    table_scores: Dict[str, float]
    column_scores: Dict[str, Dict[str, float]]
    fallback_used: bool


def _score_identifier(tokens: Set[str], identifier: str) -> float:
    parts = _split_identifier(identifier)
    if not parts:
        return 0.0
    score = 0.0
    for part in parts:
        normalized = _normalize_token(part)
        if part in tokens:
            score += 3.0
        elif normalized & tokens:
            score += 2.5
        else:
            for token in tokens:
                if len(token) >= 6 and len(part) >= 6 and (token in part or part in token):
                    score += 0.75
                    break
    return score


def filter_schema(
    question: str,
    schema: Schema,
    max_tables: int = 8,
    widened_tables: int = 12,
    max_columns_per_table: int = 12,
) -> FilteredSchema:
    tokens = tokenize_text(question)
    table_scores: Dict[str, float] = {}
    column_scores: Dict[str, Dict[str, float]] = {}

    top_column_hits: List[tuple[float, str, str]] = []
    for table in schema.tables:
        table_score = _score_identifier(tokens, table)
        per_column: Dict[str, float] = {}
        for column in schema.table_to_columns[table]:
            col_score = _score_identifier(tokens, column)
            if col_score > 0:
                per_column[column] = col_score
                top_column_hits.append((col_score, table, column))
                table_score += min(4.0, col_score)
        table_scores[table] = table_score
        column_scores[table] = per_column

    ranked_tables = sorted(
        schema.tables,
        key=lambda table: (table_scores[table], len(column_scores[table]), table),
        reverse=True,
    )
    selected = ranked_tables[:max_tables]

    top_column_hits.sort(reverse=True)
    for _, table, _ in top_column_hits[: min(10, len(top_column_hits))]:
        if table not in selected:
            selected.append(table)

    fallback_used = False
    if not selected or all(table_scores[table] <= 0 for table in selected):
        selected = ranked_tables[: max(1, min(widened_tables, len(ranked_tables)))]
        fallback_used = True

    selected = selected[: max(1, min(len(selected), widened_tables if fallback_used else len(selected)))]

    selected_columns: Dict[str, List[str]] = {}
    few_tables = len(selected) <= 4
    for table in selected:
        if few_tables:
            selected_columns[table] = list(schema.table_to_columns[table])
            continue
        scored_columns = sorted(
            schema.table_to_columns[table],
            key=lambda col: (column_scores[table].get(col, 0.0), col),
            reverse=True,
        )
        if column_scores[table]:
            chosen = scored_columns[:max_columns_per_table]
        else:
            chosen = scored_columns[: min(max_columns_per_table, len(scored_columns))]
        selected_columns[table] = chosen

    return FilteredSchema(
        selected_tables=selected,
        selected_columns=selected_columns,
        table_scores=table_scores,
        column_scores=column_scores,
        fallback_used=fallback_used,
    )

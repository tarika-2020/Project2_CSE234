from typing import Dict

from .filtering import FilteredSchema
from .schema_utils import Schema


def heuristic_schema_links(question: str, schema: Schema, filtered: FilteredSchema) -> Dict[str, list[str]]:
    lowered = question.lower()
    links: Dict[str, list[str]] = {}

    ranked_tables = sorted(
        filtered.selected_tables,
        key=lambda table: (filtered.table_scores.get(table, 0.0), table),
        reverse=True,
    )

    for table in ranked_tables[:4]:
        table_score = filtered.table_scores.get(table, 0.0)
        scored_columns = filtered.column_scores.get(table, {})
        ranked_columns = sorted(
            filtered.selected_columns.get(table, []),
            key=lambda column: (scored_columns.get(column, 0.0), column),
            reverse=True,
        )
        chosen_columns = [
            column for column in ranked_columns if scored_columns.get(column, 0.0) >= 2.5
        ][:6]

        if chosen_columns:
            links[table] = sorted(set(chosen_columns), key=str.lower)
        elif table_score > 2.0:
            links[table] = []

    if not links and ranked_tables:
        best_table = ranked_tables[0]
        links[best_table] = []

    if ("count" in lowered or "how many" in lowered) and not any(cols for cols in links.values()):
        first_table = ranked_tables[0] if ranked_tables else None
        if first_table is not None:
            links[first_table] = []

    return dict(sorted(links.items(), key=lambda item: item[0].lower()))

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def db_id_to_filename(db_id: str) -> str:
    return db_id.replace(" ", "_").replace("/", "_") + ".json"


@dataclass(frozen=True)
class Relationship:
    source_table: str
    source_column: str
    target_table: str
    target_column: str


@dataclass(frozen=True)
class Schema:
    db_id: str
    filename: str
    tables: List[str]
    table_to_columns: Dict[str, List[str]]
    lc_tables: Dict[str, str]
    lc_columns_by_table: Dict[str, Dict[str, str]]
    primary_keys: Dict[str, List[str]]
    relationships: List[Relationship]

    def has_table(self, table_name: str) -> bool:
        return table_name.lower() in self.lc_tables

    def canonical_table(self, table_name: str) -> Optional[str]:
        return self.lc_tables.get(table_name.lower())

    def canonical_column(self, table_name: str, column_name: str) -> Optional[str]:
        table = self.canonical_table(table_name)
        if table is None:
            return None
        return self.lc_columns_by_table.get(table, {}).get(column_name.lower())


def load_schema(schemas_dir: str, db_id: str) -> Schema:
    filename = db_id_to_filename(db_id)
    path = os.path.join(schemas_dir, filename)
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    tables = list(raw["table_names_original"])
    table_to_columns = {table: [] for table in tables}
    for table_idx, column_name in raw["column_names_original"]:
        if table_idx == -1:
            continue
        table_to_columns[tables[table_idx]].append(column_name)

    lc_tables = {table.lower(): table for table in tables}
    lc_columns_by_table = {
        table: {column.lower(): column for column in columns}
        for table, columns in table_to_columns.items()
    }

    primary_keys: Dict[str, List[str]] = {table: [] for table in tables}
    column_refs = raw["column_names_original"]
    for column_idx in raw.get("primary_keys", []):
        table_idx, column_name = column_refs[column_idx]
        if table_idx != -1:
            primary_keys[tables[table_idx]].append(column_name)

    relationships: List[Relationship] = []
    for source_idx, target_idx in raw.get("foreign_keys", []):
        source_table_idx, source_column = column_refs[source_idx]
        target_table_idx, target_column = column_refs[target_idx]
        if source_table_idx == -1 or target_table_idx == -1:
            continue
        relationships.append(
            Relationship(
                source_table=tables[source_table_idx],
                source_column=source_column,
                target_table=tables[target_table_idx],
                target_column=target_column,
            )
        )

    return Schema(
        db_id=db_id,
        filename=filename,
        tables=tables,
        table_to_columns=table_to_columns,
        lc_tables=lc_tables,
        lc_columns_by_table=lc_columns_by_table,
        primary_keys=primary_keys,
        relationships=relationships,
    )


def serialize_schema(
    schema: Schema,
    selected_tables: Optional[Sequence[str]] = None,
    selected_columns: Optional[Dict[str, Sequence[str]]] = None,
    include_relationships: bool = True,
) -> str:
    tables = list(selected_tables) if selected_tables is not None else list(schema.tables)
    column_map: Dict[str, Sequence[str]] = selected_columns or schema.table_to_columns
    lines = []
    for table in tables:
        columns = list(column_map.get(table, schema.table_to_columns.get(table, [])))
        column_text = ", ".join(columns)
        lines.append(f"{table}({column_text})")

    if include_relationships:
        rel_lines = []
        allowed = set(tables)
        for rel in schema.relationships:
            if rel.source_table in allowed and rel.target_table in allowed:
                rel_lines.append(
                    f"{rel.source_table}.{rel.source_column} -> "
                    f"{rel.target_table}.{rel.target_column}"
                )
        if rel_lines:
            lines.append("Relationships:")
            lines.extend(rel_lines)
    return "\n".join(lines)


def select_columns(schema: Schema, table_names: Iterable[str]) -> Dict[str, List[str]]:
    return {table: list(schema.table_to_columns[table]) for table in table_names}

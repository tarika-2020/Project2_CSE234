from .filtering import FilteredSchema
from .schema_utils import Schema, serialize_schema


SYSTEM_INSTRUCTION = """You are a schema-linking model.
Return JSON only.
The JSON must map table names to arrays of column names.
If a table is referenced but no named columns are referenced, use an empty array.
Do not include identifiers that are not present in the schema snippet."""


def build_prompt(question: str, db_id: str, schema: Schema, filtered: FilteredSchema) -> str:
    schema_text = serialize_schema(
        schema,
        selected_tables=filtered.selected_tables,
        selected_columns=filtered.selected_columns,
        include_relationships=True,
    )
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"Database: {db_id}\n"
        f"Question: {question}\n\n"
        f"Schema:\n{schema_text}\n\n"
        "Return only valid JSON for schema_links."
    )

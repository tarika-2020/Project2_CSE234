from typing import Dict, List

from .filtering import FilteredSchema


def should_use_fallback_links(
    cleaned_links: Dict[str, List[str]],
    fallback_links: Dict[str, List[str]],
    filtered: FilteredSchema,
    heuristic_links: Dict[str, List[str]],
    retrieval_links: Dict[str, List[str]],
) -> bool:
    if not fallback_links:
        return False
    if not cleaned_links:
        return True

    supported_tables = {
        table
        for table in cleaned_links
        if table in heuristic_links or table in retrieval_links
    }
    predicted_tables = len(cleaned_links)
    predicted_columns = sum(len(columns) for columns in cleaned_links.values())

    if not supported_tables:
        return True
    if filtered.fallback_used:
        return True
    if predicted_columns >= 7:
        return True
    if predicted_tables >= 4:
        return True
    return False

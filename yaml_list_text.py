"""Repair YAML list items broken by unquoted colon-space scalars."""

from __future__ import annotations

from typing import Any

# Fields that are authored as lists of plain strings in NPC/campaign YAML.
STRING_LIST_KEYS = (
    "llm_rules",
    "npc_rules",
    "personality",
    "public_facts",
    "known_facts",
    "secrets",
    "voice",
)


def yaml_list_item_as_text(item: Any) -> str:
    """
    Coerce a YAML list item to text.

    Unquoted ``[DM CHECK: …]`` / ``[DM Ruling: …]`` scalars are parsed as a
    one-entry mapping because of the ``: `` after ``[DM …``. Rejoin
    ``key: value`` when that happens so authors never need ``>-``.
    """
    if item is None:
        return ""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        if len(item) == 1:
            key, value = next(iter(item.items()))
            return f"{key}: {'' if value is None else value}"
        return "\n".join(
            f"{k}: {'' if v is None else v}"
            for k, v in item.items()
        )
    return str(item)


def repair_string_list_fields(doc: dict[str, Any] | None) -> dict[str, Any]:
    """In-place repair of known string-list fields; returns the same dict."""
    if not isinstance(doc, dict):
        return {}
    for key in STRING_LIST_KEYS:
        raw = doc.get(key)
        if not isinstance(raw, list):
            continue
        doc[key] = [
            yaml_list_item_as_text(item)
            for item in raw
            if item is not None and str(yaml_list_item_as_text(item)).strip()
        ]
    return doc

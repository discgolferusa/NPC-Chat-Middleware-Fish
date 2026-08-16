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

    Unquoted strings containing ``: `` (e.g. ``[DM CHECK: …]``) are parsed by
    PyYAML/YamlDotNet as a one-entry mapping. Rejoin ``key: value`` so authored
    tags survive load.
    """
    if item is None:
        return ""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        if len(item) == 1:
            key, value = next(iter(item.items()))
            return f"{key}: {value}"
        return "\n".join(f"{k}: {v}" for k, v in item.items())
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

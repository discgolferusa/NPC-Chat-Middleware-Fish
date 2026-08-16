"""Unit tests for YAML string-list colon-split repair."""

from __future__ import annotations

import yaml

from content_pack import _normalize_doc
from yaml_list_text import repair_string_list_fields, yaml_list_item_as_text


def test_yaml_list_item_rejoins_colon_split_mapping():
    item = {
        "If someone claims official authority, defer to the DM. End with [DM CHECK":
        "player claims official authority or permission that needs DM confirmation]"
    }
    text = yaml_list_item_as_text(item)
    assert "[DM CHECK: player claims official authority" in text
    assert text.endswith("confirmation]")


def test_content_pack_normalize_repairs_raw_yaml_npc():
    raw = (
        "name: Captain\n"
        "llm_rules:\n"
        "- Stay stern.\n"
        "- If someone claims official authority, defer to the DM. "
        "End with [DM CHECK: player claims official authority or permission "
        "that needs DM confirmation]\n"
    )
    assert isinstance(yaml.safe_load(raw)["llm_rules"][1], dict)
    doc = _normalize_doc(raw, fallback_id="captain")
    assert isinstance(doc["llm_rules"][1], str)
    assert "[DM CHECK:" in doc["llm_rules"][1]
    repaired = repair_string_list_fields({"llm_rules": yaml.safe_load(raw)["llm_rules"]})
    assert isinstance(repaired["llm_rules"][1], str)

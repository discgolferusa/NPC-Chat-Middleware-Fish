"""Regression tests for campaign document import (content_import.py)."""

from __future__ import annotations

from pathlib import Path

import content_import


FIXTURE_PATH = (
    Path(__file__).resolve().parent / "tests" / "fixtures" / "chariot_of_the_gods_npc_section.txt"
)

# Wrong NPCs produced on Chariot of the Gods import before stat-block handling.
CHARIOT_FALSE_POSITIVE_NPCS = ("Captain Miller", "David 8", "Miss Vickers")

# Stat-block NPCs that must be discovered from NON-PLAYER CHARACTERS.
CHARIOT_EXPECTED_NPCS = (
    "Albert Johns",
    "Valerie Reid",
    "Liam Flynn",
    "Daniel Cooper",
    "Lori Clayton",
    "Adisa Bolaji",
    "Pinion",
    "Helen Bein",
    "Micky Horton",
)


def _load_chariot_fixture() -> str:
    return FIXTURE_PATH.read_text(encoding="utf-8")


def test_chariot_fixture_finds_spaced_non_player_characters_header():
    text = _load_chariot_fixture()
    collapsed = content_import._collapse_spaced_caps(text)
    span = content_import._find_npc_section_span(collapsed)
    assert span is not None
    start, end = span
    section = collapsed[start:end]
    assert "JOHNS, OFFICER" in section
    assert "Albert Johns" in section


def test_chariot_fixture_extracts_stat_block_npcs():
    text = _load_chariot_fixture()
    collapsed = content_import._collapse_spaced_caps(text)
    span = content_import._find_npc_section_span(collapsed)
    assert span is not None
    section = collapsed[span[0] : span[1]]
    parsed = content_import._extract_alien_rpg_stat_block_npcs(section)
    names = {row["name"] for row in parsed}
    for expected in CHARIOT_EXPECTED_NPCS:
        assert expected in names


def test_chariot_regression_replaces_movie_quote_outline_npcs(monkeypatch):
    """LLM outline picked Miller / David 8 / Vickers; stat blocks should win."""
    text = _load_chariot_fixture()
    bad_outline = {
        "campaign": {
            "campaign_id": "chariot_of_the_gods",
            "name": "Chariot of the Gods",
            "description": "ALIEN RPG scenario",
        },
        "locations": [
            {"id": "uscss_cronus", "name": "USCSS Cronus", "type": "building"},
            {"id": "sotillo", "name": "Sotillo", "type": "building"},
        ],
        "npcs": [
            {
                "name": "Captain Miller",
                "role": "prometheus captain",
                "first_location_id": "uscss_cronus",
            },
            {
                "name": "David 8",
                "role": "android",
                "first_location_id": "uscss_cronus",
            },
            {
                "name": "Miss Vickers",
                "role": "weyland exec",
                "first_location_id": "uscss_cronus",
            },
        ],
    }

    def fake_call_llm_json(system_prompt, user_prompt, *, warnings, step):
        if step == "outline":
            return bad_outline
        if step == "location detail":
            return {"locations": []}
        if step == "npc detail":
            return {"npcs": []}
        return {}

    monkeypatch.setattr(content_import, "_call_llm_json", fake_call_llm_json)

    result = content_import.analyze_campaign_document(text, "create_new")
    names = {npc["name"] for npc in result["npcs"]}

    for bad_name in CHARIOT_FALSE_POSITIVE_NPCS:
        assert bad_name not in names

    for expected in CHARIOT_EXPECTED_NPCS:
        assert expected in names

    warning_blob = " ".join(result.get("warnings") or []).lower()
    assert "non-player characters" in warning_blob
    assert "stat block" in warning_blob

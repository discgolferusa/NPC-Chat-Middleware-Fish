"""Regression tests for campaign document import (content_import.py)."""

from __future__ import annotations

from pathlib import Path

import content_import


FIXTURE_PATH = (
    Path(__file__).resolve().parent / "tests" / "fixtures" / "chariot_of_the_gods_npc_section.txt"
)
PDF_EXTRACT_FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "tests"
    / "fixtures"
    / "chariot_of_the_gods_pdf_extract_npc_section.txt"
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


def _load_chariot_pdf_extract_fixture() -> str:
    return PDF_EXTRACT_FIXTURE_PATH.read_text(encoding="utf-8")


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


def test_chariot_pdf_extract_parses_two_column_stat_blocks():
    """Real PDF text glues two headings per line; FULL NAME rows must still parse."""
    text = _load_chariot_pdf_extract_fixture()
    collapsed = content_import._collapse_spaced_caps(text)
    span = content_import._find_npc_section_span(collapsed)
    assert span is not None
    section = collapsed[span[0] : span[1]]
    assert "REID, COLONIAL MARINE FLYNN, MEDIC" in section

    headings = content_import._iter_stat_block_headings("REID, COLONIAL MARINE FLYNN, MEDIC")
    assert headings == [("REID", "COLONIAL MARINE"), ("FLYNN", "MEDIC")]
    headings = content_import._iter_stat_block_headings("COOPER, SCIENTIST CLAYTON, COMPANY AGENT")
    assert headings == [("COOPER", "SCIENTIST"), ("CLAYTON", "COMPANY AGENT")]
    headings = content_import._iter_stat_block_headings("AVA 6, SYNTHETIC BOLAJI, OFFICER")
    assert headings == [("AVA 6", "SYNTHETIC"), ("BOLAJI", "OFFICER")]

    parsed = content_import._extract_alien_rpg_stat_block_npcs(section)
    names = {row["name"] for row in parsed}
    for expected in CHARIOT_EXPECTED_NPCS:
        assert expected in names, f"missing {expected} in {sorted(names)}"
    assert "Ava 6" in names
    by_name = {row["name"]: row for row in parsed}
    assert by_name["Valerie Reid"]["short_name"].upper() == "REID"
    assert "marine flynn" not in by_name["Valerie Reid"]["role"].casefold()
    assert "Colonial Marine" in by_name["Valerie Reid"]["role"]
    assert len(parsed) >= 10


def test_chariot_pdf_extract_recovers_from_empty_llm_outline(monkeypatch):
    """Tight Groq budgets often yield empty outline JSON; stat blocks must still populate."""
    text = _load_chariot_pdf_extract_fixture()

    def fake_call_llm_json(system_prompt, user_prompt, *, warnings, step):
        if step == "outline":
            return {"campaign": {}, "locations": [], "npcs": []}
        if step == "location detail":
            return {"locations": []}
        if step == "npc detail":
            return {"npcs": []}
        return {}

    monkeypatch.setattr(content_import, "_call_llm_json", fake_call_llm_json)
    monkeypatch.setattr(content_import, "content_creation_max_input_chars", lambda: 6016)

    result = content_import.analyze_campaign_document(text, "create_new")
    assert "chariot" in result["campaign"]["campaign_id"].casefold() or "gods" in result[
        "campaign"
    ]["name"].casefold()
    names = {npc["name"] for npc in result["npcs"]}
    for expected in CHARIOT_EXPECTED_NPCS:
        assert expected in names
    assert "Ava 6" in names
    loc_ids = {loc["id"] for loc in result["locations"]}
    assert "uscss_cronus" in loc_ids or "sotillo" in loc_ids
    assert "Captain Miller" not in names


def test_outline_failure_recovers_from_stat_blocks(monkeypatch):
    text = _load_chariot_pdf_extract_fixture()

    def fake_call_llm_json(system_prompt, user_prompt, *, warnings, step):
        if step == "outline":
            raise RuntimeError("simulated outline TPM failure")
        return {"locations": []} if step == "location detail" else {"npcs": []}

    monkeypatch.setattr(content_import, "_call_llm_json", fake_call_llm_json)
    result = content_import.analyze_campaign_document(text, "create_new")
    names = {npc["name"] for npc in result["npcs"]}
    assert "Albert Johns" in names
    assert "Ava 6" in names
    assert any("outline pass failed" in w.casefold() for w in result["warnings"])


def test_truncate_preserving_npc_section_respects_max_chars():
    intro = "X" * 20_000
    section = _load_chariot_fixture()
    text = f"{intro}\n\n{section}"
    max_chars = 6016
    trimmed = content_import._truncate_preserving_npc_section(text, max_chars)
    assert len(trimmed) <= max_chars
    assert "JOHNS, OFFICER" in trimmed
    assert "FULL NAME: Albert Johns" in trimmed


def test_truncate_preserving_npc_section_keeps_stat_blocks():
    intro = "X" * 20_000
    section = _load_chariot_fixture()
    text = f"{intro}\n\n{section}"
    max_chars = 12_000
    trimmed = content_import._truncate_preserving_npc_section(text, max_chars)
    assert len(trimmed) <= max_chars
    assert "JOHNS, OFFICER" in trimmed
    assert "FULL NAME: Albert Johns" in trimmed
    assert "FULL NAME: Micky Horton" in trimmed
    assert "Captain Miller" not in trimmed

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


def test_generic_npc_section_without_full_name_keeps_llm_outline(monkeypatch):
    """Caps NAME, ROLE lines without FULL NAME must not replace the LLM outline."""
    text = """NON-PLAYER CHARACTERS

JOHN SMITH, GUARD
A town guard on patrol.

JANE DOE, MERCHANT
Runs the general store.

BOB JONES, INNKEEPER
Owns the tavern.
"""
    outline = {
        "campaign": {
            "campaign_id": "test_campaign",
            "name": "Test Campaign",
            "description": "A generic module",
        },
        "locations": [{"id": "tavern", "name": "Tavern", "type": "building"}],
        "npcs": [
            {
                "name": "Archdeacon Verity-9",
                "role": "priest",
                "first_location_id": "tavern",
            },
        ],
    }

    def fake_call_llm_json(system_prompt, user_prompt, *, warnings, step):
        if step == "outline":
            return outline
        if step == "location detail":
            return {"locations": []}
        if step == "npc detail":
            return {"npcs": []}
        return {}

    monkeypatch.setattr(content_import, "_call_llm_json", fake_call_llm_json)

    result = content_import.analyze_campaign_document(text, "create_new")
    names = {npc["name"] for npc in result["npcs"]}
    assert "Archdeacon Verity-9" in names
    assert result["npcs"][0]["id"] == "archdeacon_verity_9"


def test_merge_mode_skips_existing_npcs_from_stat_blocks(monkeypatch):
    """Merge must not re-import NPCs already present in the campaign."""
    text = _load_chariot_fixture()
    outline = {
        "campaign": {
            "campaign_id": "chariot_of_the_gods",
            "name": "Chariot of the Gods",
            "description": "ALIEN RPG scenario",
        },
        "locations": [{"id": "uscss_cronus", "name": "USCSS Cronus", "type": "building"}],
        "npcs": [],
    }

    def fake_call_llm_json(system_prompt, user_prompt, *, warnings, step):
        if step == "outline":
            return outline
        if step == "location detail":
            return {"locations": []}
        if step == "npc detail":
            return {"npcs": []}
        return {}

    monkeypatch.setattr(content_import, "_call_llm_json", fake_call_llm_json)

    result = content_import.analyze_campaign_document(
        text,
        "merge",
        existing_campaign_id="chariot_of_the_gods",
        existing_location_ids=["uscss_cronus"],
        existing_npc_names=["Albert Johns", "Valerie Reid"],
    )
    names = {npc["name"] for npc in result["npcs"]}
    assert "Albert Johns" not in names
    assert "Valerie Reid" not in names
    assert "Liam Flynn" in names


def test_location_hints_resolve_existing_campaign_locations(monkeypatch):
    """Stat-block location hints should match ids from existing_location_ids."""
    text = _load_chariot_fixture()
    outline = {
        "campaign": {
            "campaign_id": "chariot_of_the_gods",
            "name": "Chariot of the Gods",
            "description": "ALIEN RPG scenario",
        },
        "locations": [],
        "npcs": [],
    }

    def fake_call_llm_json(system_prompt, user_prompt, *, warnings, step):
        if step == "outline":
            return outline
        if step == "location detail":
            return {"locations": []}
        if step == "npc detail":
            return {"npcs": []}
        return {}

    monkeypatch.setattr(content_import, "_call_llm_json", fake_call_llm_json)

    result = content_import.analyze_campaign_document(
        text,
        "merge",
        existing_campaign_id="chariot_of_the_gods",
        existing_location_ids=["uscss_cronus", "sotillo"],
        existing_npc_names=[],
    )
    by_name = {npc["name"]: npc for npc in result["npcs"]}
    assert by_name["Albert Johns"]["location_id"] == "uscss_cronus"

"""Regression tests for campaign document import (content_import.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

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


PDFPIG_FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "tests"
    / "fixtures"
    / "chariot_of_the_gods_pdfpig_npc_section.txt"
)


def _load_chariot_pdfpig_fixture() -> str:
    return PDFPIG_FIXTURE_PATH.read_text(encoding="utf-8")


DOCTOR_WHO_CATSEYE_FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "tests"
    / "fixtures"
    / "doctor_who_catseye_npcs_excerpt.txt"
)


def _load_doctor_who_catseye_fixture() -> str:
    return DOCTOR_WHO_CATSEYE_FIXTURE_PATH.read_text(encoding="utf-8")


DOCTOR_WHO_CALLOUT_PDF_FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "tests"
    / "fixtures"
    / "doctor_who_catseye_pdf_extract_callouts.txt"
)

# Editor PdfPig Y-grouped extract: sequential callout blocks, stylized caps
# (AWAreNess / Peter HutcHiNs), not mid-word spaces or dual-column line glue.
DOCTOR_WHO_PDFPIG_FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "tests"
    / "fixtures"
    / "doctor_who_catseye_pdfpig_callouts.txt"
)

# Synthetic worst-case (columns glued + letter spaces). Not observed on live
# Cat's Eye PdfPig dumps; kept only for resilience of flex helpers.
DOCTOR_WHO_PDFPIG_GLUED_SYNTHETIC_PATH = (
    Path(__file__).resolve().parent
    / "tests"
    / "fixtures"
    / "doctor_who_catseye_pdfpig_glued_columns_synthetic.txt"
)


def _load_doctor_who_callout_pdf_fixture() -> str:
    return DOCTOR_WHO_CALLOUT_PDF_FIXTURE_PATH.read_text(encoding="utf-8")


def _load_doctor_who_pdfpig_fixture() -> str:
    return DOCTOR_WHO_PDFPIG_FIXTURE_PATH.read_text(encoding="utf-8")


def _load_doctor_who_pdfpig_glued_synthetic() -> str:
    return DOCTOR_WHO_PDFPIG_GLUED_SYNTHETIC_PATH.read_text(encoding="utf-8")


def test_npc_section_header_end_pattern_is_whole_line_only():
    assert content_import._NPC_SECTION_HEADER_END_PATTERN.match("APPENDIX I")
    assert content_import._NPC_SECTION_HEADER_END_PATTERN.match("INDEX")
    assert content_import._NPC_SECTION_HEADER_END_PATTERN.match("MAPS")
    assert content_import._NPC_SECTION_HEADER_END_PATTERN.match("Appendix I on page 38.") is None
    assert content_import._NPC_SECTION_HEADER_END_PATTERN.search(
        "described in Appendix I on page 38. very good at giving orders."
    ) is None
    assert content_import._NPC_SECTION_HEADER_END_PATTERN.search(
        "see the index for more NPCs listed elsewhere"
    ) is None


def test_mid_prose_appendix_does_not_end_npc_section():
    """Short lines that mention Appendix mid-sentence must not truncate the span."""
    text = """
NON-PLAYER CHARACTERS
JOHNS, OFFICER
FULL NAME: Albert Johns
PERSONALITY: Submissive. described in Appendix I on page 38. very good at giving orders.
PERSONAL AGENDA: Find a leader.

REID, COLONIAL MARINE
FULL NAME: Valerie Reid
PERSONALITY: On Edge
PERSONAL AGENDA: Terminate threats.

THE DECK PLANS
Corridor map follows.
"""
    collapsed = content_import._collapse_spaced_caps(text)
    span = content_import._find_npc_section_span(collapsed)
    assert span is not None
    section = collapsed[span[0] : span[1]]
    assert "Albert Johns" in section
    assert "Valerie Reid" in section
    assert "THE DECK PLANS" not in section


def test_small_npc_section_does_not_extend_past_deck_plans():
    """Fewer than 3 FULL NAME rows must not swallow chapters after a hard end."""
    text = """
NON-PLAYER CHARACTERS
JOHNS, OFFICER
FULL NAME: Albert Johns
PERSONALITY: Submissive
PERSONAL AGENDA: Find a leader.

REID, COLONIAL MARINE
FULL NAME: Valerie Reid
PERSONALITY: On Edge
PERSONAL AGENDA: Terminate threats.

THE DECK PLANS
This chapter describes every corridor on the Cronus in great detail.
FULL NAME: Should Not Be Imported
Captain Miller warned David 8 that Miss Vickers would never approve.
"""
    collapsed = content_import._collapse_spaced_caps(text)
    span = content_import._find_npc_section_span(collapsed)
    assert span is not None
    section = collapsed[span[0] : span[1]]
    assert "Albert Johns" in section
    assert "Valerie Reid" in section
    assert "THE DECK PLANS" not in section
    assert "Should Not Be Imported" not in section
    assert "Captain Miller" not in section


def test_chariot_pdfpig_extract_finds_all_stat_block_npcs():
    """PdfPig Y-grouped lines glue columns and mid-prose Appendix refs."""
    text = _load_chariot_pdfpig_fixture()
    collapsed = content_import._collapse_spaced_caps(text)
    span = content_import._find_npc_section_span(collapsed)
    assert span is not None
    section = collapsed[span[0] : span[1]]
    # Must not end at mid-prose "Appendix I on page 38".
    assert "Valerie Reid" in section or "FULL NAME: Valerie Reid" in section
    parsed = content_import._extract_alien_rpg_stat_block_npcs(section)
    names = {row["name"] for row in parsed}
    for expected in CHARIOT_EXPECTED_NPCS:
        assert expected in names, f"missing {expected} in {sorted(names)}"
    assert "Valerie Reid FULL NAME" not in names
    assert "Micky Horton take the ship" not in names


def test_chariot_pdfpig_analyze_imports_all_npcs(monkeypatch):
    """PdfPig extract skips outline but still runs location/NPC detail sweeps."""
    text = _load_chariot_pdfpig_fixture()
    calls: list[str] = []

    def fake_call_llm_json(system_prompt, user_prompt, *, warnings, step):
        calls.append(step)
        if step == "outline":
            raise AssertionError("stat-block fast path should not call the outline LLM")
        if step == "location detail":
            return {"locations": []}
        if step == "npc detail":
            return {"npcs": []}
        return {}

    monkeypatch.setattr(content_import, "_call_llm_json", fake_call_llm_json)
    result = content_import.analyze_campaign_document(text, "create_new")
    assert "outline" not in calls
    assert "location detail" in calls
    assert "npc detail" in calls
    names = {npc["name"] for npc in result["npcs"]}
    for expected in CHARIOT_EXPECTED_NPCS:
        assert expected in names
    assert "Ava 6" in names


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
    """Stat-block modules skip outline but still run location/NPC detail sweeps."""
    text = _load_chariot_pdf_extract_fixture()
    calls: list[str] = []

    def fake_call_llm_json(system_prompt, user_prompt, *, warnings, step):
        calls.append(step)
        if step == "outline":
            raise AssertionError("stat-block fast path should not call the outline LLM")
        if step == "location detail":
            return {"locations": []}
        if step == "npc detail":
            # Sparse on purpose so PERSONALITY / AGENDA seed fallback still applies.
            return {"npcs": []}
        return {}

    monkeypatch.setattr(content_import, "_call_llm_json", fake_call_llm_json)

    result = content_import.analyze_campaign_document(text, "create_new")
    assert "outline" not in calls
    assert "location detail" in calls
    assert "npc detail" in calls
    assert "chariot" in result["campaign"]["campaign_id"].casefold() or "gods" in result[
        "campaign"
    ]["name"].casefold()
    names = {npc["name"] for npc in result["npcs"]}
    for expected in CHARIOT_EXPECTED_NPCS:
        assert expected in names
    assert "Ava 6" in names
    johns = next(npc for npc in result["npcs"] if npc["name"] == "Albert Johns")
    assert johns["personality"]
    assert johns["secrets"] or johns["known_facts"]
    loc_ids = {loc["id"] for loc in result["locations"]}
    assert "uscss_cronus" in loc_ids or "sotillo" in loc_ids
    assert "Captain Miller" not in names
    warning_blob = " ".join(result.get("warnings") or []).lower()
    assert "skipping llm outline" in warning_blob
    assert "detail sweep still runs" in warning_blob


def test_outline_failure_recovers_from_stat_blocks(monkeypatch):
    """ALIEN PDF skips outline; detail may fail and still fall back to seeds."""
    text = _load_chariot_pdf_extract_fixture()
    calls: list[str] = []

    def fake_call_llm_json(system_prompt, user_prompt, *, warnings, step):
        calls.append(step)
        if step == "outline":
            raise RuntimeError("should not be called on stat-block fast path")
        if step == "location detail":
            return {"locations": []}
        if step == "npc detail":
            return {"npcs": []}
        return {}

    monkeypatch.setattr(content_import, "_call_llm_json", fake_call_llm_json)
    result = content_import.analyze_campaign_document(text, "create_new")
    assert "outline" not in calls
    assert "npc detail" in calls
    names = {npc["name"] for npc in result["npcs"]}
    assert "Albert Johns" in names
    assert "Ava 6" in names


def test_stat_block_fast_path_uses_llm_npc_detail_when_rich(monkeypatch):
    """Detail sweep output should win over PERSONALITY/AGENDA seed stubs."""
    text = _load_chariot_pdf_extract_fixture()

    def fake_call_llm_json(system_prompt, user_prompt, *, warnings, step):
        if step == "outline":
            raise AssertionError("outline must be skipped")
        if step == "location detail":
            return {
                "locations": [
                    {
                        "id": "uscss_cronus",
                        "summary": "A derelict science vessel drifting in deep space.",
                        "current_events": ["Life support is failing on the lower decks."],
                    }
                ]
            }
        if step == "npc detail":
            return {
                "npcs": [
                    {
                        "id": "albert_johns",
                        "gender": "male",
                        "race": "human",
                        "personality": ["LLM-enriched submissive officer"],
                        "public_facts": ["Serves as third officer on the Cronus."],
                        "known_facts": ["Needs a strong leader to function."],
                        "secrets": ["Will betray the crew if cornered."],
                        "llm_rules": ["Stay timid unless given orders."],
                        "voice": ["Soft", "Deferential"],
                    }
                ]
            }
        return {}

    monkeypatch.setattr(content_import, "_call_llm_json", fake_call_llm_json)
    result = content_import.analyze_campaign_document(text, "create_new")
    johns = next(npc for npc in result["npcs"] if npc["name"] == "Albert Johns")
    assert "LLM-enriched submissive officer" in johns["personality"]
    assert any("third officer" in fact.casefold() for fact in johns["public_facts"])
    cronus = next((loc for loc in result["locations"] if loc["id"] == "uscss_cronus"), None)
    if cronus is not None:
        assert "derelict" in cronus["summary"].casefold()


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


def test_chunked_outline_then_detail_batches(monkeypatch):
    """Editor-style chunked path: outline stubs, then per-batch detail hops."""
    text = _load_chariot_pdf_extract_fixture()
    steps: list[str] = []

    def fake_call_llm_json(system_prompt, user_prompt, *, warnings, step):
        steps.append(step)
        if step == "outline":
            raise AssertionError("stat-block outline must be skipped")
        if step == "location detail":
            return {"locations": []}
        if step == "npc detail":
            return {"npcs": []}
        return {}

    monkeypatch.setattr(content_import, "_call_llm_json", fake_call_llm_json)

    outline = content_import.outline_campaign_document(text, "create_new")
    assert "outline" not in steps
    assert outline["npcs"]
    assert outline["locations"] or any(
        npc.get("first_location_id") for npc in outline["npcs"]
    )
    names = {npc["name"] for npc in outline["npcs"]}
    assert "Albert Johns" in names

    campaign = outline["campaign"]
    loc_batch = outline["locations"][: content_import.MAX_LOCATIONS_PER_BATCH] or [
        {
            "id": "uscss_cronus",
            "name": "USCSS Cronus",
            "type": "building",
            "seeded_from_stat_blocks": True,
        }
    ]
    loc_result = content_import.detail_location_batch(text, campaign, loc_batch)
    assert loc_result["locations"]
    assert "location detail" in steps

    johns = next(npc for npc in outline["npcs"] if npc["name"] == "Albert Johns")
    # First hop is LLM-only (editor may retry separately when sparse).
    sparse = content_import.detail_npc_batch(text, campaign, [johns])
    assert sparse["npcs"][0]["name"] == "Albert Johns"
    assert content_import._npc_detail_is_sparse(sparse["npcs"][0])
    assert "npc detail" in steps

    # Last Funnel hop may request excerpt/stat seed fallback after retries.
    npc_result = content_import.detail_npc_batch(
        text, campaign, [johns], allow_seed_fallback=True
    )
    assert npc_result["npcs"][0]["name"] == "Albert Johns"
    assert npc_result["npcs"][0]["personality"] or npc_result["npcs"][0]["secrets"]


def test_detail_location_batch_rejects_oversized_batch():
    with pytest.raises(content_import.ContentImportError, match="at most"):
        content_import.detail_location_batch(
            "A location appears here.",
            {"campaign_id": "c", "name": "C"},
            [
                {"id": "a", "name": "A", "type": "building"},
                {"id": "b", "name": "B", "type": "building"},
            ],
        )


def test_detail_location_batch_does_not_sparse_retry(monkeypatch):
    """Funnel hops must be one LLM call; seed fallback is opt-in after editor retries."""
    calls: list[str] = []

    def fake_call_llm_json(system_prompt, user_prompt, *, warnings, step):
        calls.append(step)
        return {"locations": []}

    monkeypatch.setattr(content_import, "_call_llm_json", fake_call_llm_json)
    sparse = content_import.detail_location_batch(
        "USCSS Cronus drifts in deep space with failing life support.",
        {"campaign_id": "c", "name": "C"},
        [{"id": "uscss_cronus", "name": "USCSS Cronus", "type": "building", "seeded_from_stat_blocks": True}],
    )
    assert calls == ["location detail"]
    assert content_import._location_detail_is_sparse(sparse["locations"][0])

    result = content_import.detail_location_batch(
        "USCSS Cronus drifts in deep space with failing life support.",
        {"campaign_id": "c", "name": "C"},
        [{"id": "uscss_cronus", "name": "USCSS Cronus", "type": "building", "seeded_from_stat_blocks": True}],
        allow_seed_fallback=True,
    )
    assert len(calls) == 2
    assert result["locations"][0]["id"] == "uscss_cronus"
    assert any("fell back" in w.lower() for w in result["warnings"])


def test_harvest_important_locations_from_catseye_pdfpig():
    """TOC + Exploring headers fill hotel sub-areas without a Cubicle 7-specific parser."""
    text = _load_doctor_who_pdfpig_fixture()
    harvested = content_import._harvest_important_locations(text, "Majestic Hotel")
    names = {row["name"].casefold() for row in harvested}
    assert "eddings" in names
    assert any("courtyard" in n for n in names)
    assert any("confusing room" in n for n in names)
    assert any("swimming pool" in n for n in names)
    # NPC callouts / credits must not become rooms.
    assert "chad" not in names
    assert not any("hutchins" in n for n in names)
    courtyard = next(row for row in harvested if "courtyard" in row["name"].casefold())
    assert "investigate the courtyard" in courtyard["description"].casefold()


def test_subarea_stop_name_uses_exact_stat_match_not_substring():
    assert content_import._is_subarea_stop_name("Awareness")
    assert content_import._is_subarea_stop_name("Traits")
    assert content_import._is_subarea_stop_name("Survival")
    assert content_import._is_subarea_stop_name("Medicine")
    # Real place names that merely contain stat letter sequences must survive.
    assert not content_import._is_subarea_stop_name("Portraits")
    assert not content_import._is_subarea_stop_name("Firefighting")
    assert not content_import._is_subarea_stop_name("Survival Bunker")
    assert not content_import._is_subarea_stop_name("Medicine Cabinet")


def test_harvest_important_locations_scoped_to_parent_location():
    """Shared TOC/Exploring rooms must not attach to every location stub in the doc."""
    text = _load_doctor_who_pdfpig_fixture()
    hotel = content_import._harvest_important_locations(text, "Majestic Hotel")
    assert hotel
    assert any("courtyard" in row["name"].casefold() for row in hotel)

    other = content_import._harvest_important_locations(text, "USCSS Cronus")
    other_names = {row["name"].casefold() for row in other}
    assert not any("courtyard" in n for n in other_names)
    assert "eddings" not in other_names
    assert not any("swimming pool" in n for n in other_names)

    merged_other = content_import._merge_location_detail(
        {"id": "uscss_cronus", "name": "USCSS Cronus", "type": "building"},
        {
            "summary": "A ship.",
            "important_locations": [{"name": "Bridge", "description": "Command deck.", "public_knowledge": ""}],
        },
        "cats_eye",
        document_text=text,
    )
    merged_names = {row["name"].casefold() for row in merged_other["important_locations"]}
    assert "bridge" in merged_names
    assert not any("courtyard" in n for n in merged_names)


def test_location_detail_excerpt_scopes_exploring_to_parent():
    text = _load_doctor_who_pdfpig_fixture()
    hotel_excerpt = content_import._location_detail_excerpt(
        text,
        {"id": "majestic_hotel", "name": "Majestic Hotel", "type": "building"},
    )
    assert "cOurtyArd" in hotel_excerpt or "Courtyard" in hotel_excerpt

    other_excerpt = content_import._location_detail_excerpt(
        text,
        {"id": "uscss_cronus", "name": "USCSS Cronus", "type": "building"},
    )
    # Unrelated stubs should not receive the hotel Exploring window wholesale.
    assert "exPLOriNG tHe HOteL" not in other_excerpt
    assert "tHe cOurtyArd" not in other_excerpt


def test_location_detail_merges_harvested_subareas_when_llm_omits_them(monkeypatch):
    text = _load_doctor_who_pdfpig_fixture()

    def fake_call_llm_json(system_prompt, user_prompt, *, warnings, step):
        assert "important_locations" in user_prompt
        # Good hotel summary, but empty sub-areas — the common failure mode.
        return {
            "locations": [
                {
                    "id": "majestic_hotel",
                    "summary": "A boarded-up 1980s hotel occupied by squatters and feral cats.",
                    "current_events": ["Heavy rain traps the Poc Unity inside."],
                    "important_locations": [],
                    "rumors": [],
                    "local_tone": ["gloomy"],
                    "knowledge_rules": [],
                }
            ]
        }

    monkeypatch.setattr(content_import, "_call_llm_json", fake_call_llm_json)
    result = content_import.detail_location_batch(
        text,
        {"campaign_id": "cats_eye", "name": "Cat's Eye"},
        [{"id": "majestic_hotel", "name": "Majestic Hotel", "type": "building"}],
    )
    loc = result["locations"][0]
    names = {row["name"].casefold() for row in loc["important_locations"]}
    assert any("courtyard" in n for n in names)
    assert any("eddings" in n for n in names)
    assert any("confusing room" in n for n in names)
    assert any("swimming pool" in n for n in names)
    assert "boarded-up" in loc["summary"].casefold()


def test_location_detail_excerpt_includes_exploring_scenes():
    text = _load_doctor_who_pdfpig_fixture()
    excerpt = content_import._location_detail_excerpt(
        text,
        {"id": "majestic_hotel", "name": "Majestic Hotel", "type": "building"},
    )
    assert "cOurtyArd" in excerpt or "Courtyard" in excerpt
    assert "cONFusiNG" in excerpt or "Confusing" in excerpt
    assert "sWiMMiNG" in excerpt or "Swimming" in excerpt


def test_detail_npc_batch_rejects_oversized_batch():
    with pytest.raises(content_import.ContentImportError, match="at most"):
        content_import.detail_npc_batch(
            "NON-PLAYER CHARACTERS\nFULL NAME: A",
            {"campaign_id": "c", "name": "C"},
            [
                {"id": "a", "name": "A", "role": "x", "first_location_id": "loc"},
                {"id": "b", "name": "B", "role": "y", "first_location_id": "loc"},
            ],
        )


def test_normalize_campaign_meta_preserves_merge_campaign_id():
    meta = content_import._normalize_campaign_meta(
        {"campaign_id": "MultiTest", "name": "Multi Test", "description": ""}
    )
    assert meta["campaign_id"] == "MultiTest"


def test_normalize_campaign_meta_snake_cases_when_id_missing():
    meta = content_import._normalize_campaign_meta({"name": "Multi Test"})
    assert meta["campaign_id"] == "multi_test"


def test_normalize_npc_stubs_empty_location_stays_empty():
    stubs = content_import._normalize_npc_stubs(
        [{"id": "albert_johns", "name": "Albert Johns", "role": "Officer", "first_location_id": ""}]
    )
    assert stubs[0]["first_location_id"] == ""
    assert content_import.snake_case("") == "unnamed"


def test_detail_npc_batch_preserves_campaign_id_and_empty_location(monkeypatch):
    def fake_call_llm_json(system_prompt, user_prompt, *, warnings, step):
        return {"npcs": []}

    monkeypatch.setattr(content_import, "_call_llm_json", fake_call_llm_json)
    result = content_import.detail_npc_batch(
        "Albert Johns is an officer with PERSONALITY: Cautious.",
        {"campaign_id": "MultiTest", "name": "Multi Test"},
        [
            {
                "id": "albert_johns",
                "name": "Albert Johns",
                "role": "Officer",
                "first_location_id": "",
                "seed_personality": "Cautious",
                "seed_personal_agenda": "Survive",
                "seeded_from_stat_blocks": True,
            }
        ],
        allow_seed_fallback=True,
    )
    npc = result["npcs"][0]
    assert npc["campaign_id"] == "MultiTest"
    assert npc["location_id"] == ""
    assert npc["personality"]


def test_seed_npc_detail_from_doctor_who_traits_and_bio():
    text = _load_doctor_who_catseye_fixture()
    for name in ("Max Eddings", "Peter Hutchins", "Chad"):
        seeded = content_import._seed_npc_detail_from_document(name, "squatter", text)
        assert seeded is not None, f"expected seed for {name}"
        assert seeded["personality"] or seeded["public_facts"], f"empty seed for {name}"
        assert not content_import._npc_detail_is_sparse(seeded), f"sparse seed for {name}"


def test_seed_peter_hutchins_from_pdfpig_editor_extract():
    """Real editor PdfPig Cat's Eye dump: sequential callouts, stylized caps."""
    text = _load_doctor_who_pdfpig_fixture()
    assert "Peter HutcHiNs" in text
    assert "MAx e d d iNGs Pe t e r" not in text  # not the synthetic glued header
    seeded = content_import._seed_npc_detail_from_document("Peter Hutchins", "squatter", text)
    assert seeded is not None
    joined = " ".join(seeded["personality"])
    assert "Friends:" in joined
    assert "majority of the squatters" in joined
    assert "Warm" in seeded["personality"]
    assert "Friendly" in seeded["personality"]
    assert "Easy to smile" in seeded["personality"]
    assert any("leader" in fact.casefold() for fact in seeded["public_facts"])
    # Max's block is adjacent but sequential — do not steal his traits.
    assert "Selfish" not in joined
    assert "Argumentative" not in joined


def test_extract_cubicle7_callouts_from_catseye_pdf_fixture():
    text = _load_doctor_who_callout_pdf_fixture()
    parsed = content_import._extract_cubicle7_callout_npcs(text)
    names = {row["name"] for row in parsed}
    assert names >= {"Kate Petersen", "Max Eddings", "Peter Hutchins", "Chad"}
    # Credits / TOC staff must not become callout NPCs.
    assert "Nick Huggins" not in names
    assert "Paul Bourne" not in names
    assert "Gareth Ryder-Hanrahan" not in names


def test_extract_cubicle7_callouts_from_pdfpig_editor_extract():
    """Editor PdfPig Cat's Eye: stylized caps, intact sequential NPC callouts."""
    text = _load_doctor_who_pdfpig_fixture()
    parsed = content_import._extract_cubicle7_callout_npcs(text)
    names = {row["name"] for row in parsed}
    assert names >= {"Kate Petersen", "Max Eddings", "Peter Hutchins", "Chad"}
    assert "Nick Huggins" not in names
    assert "Paul Bourne" not in names


def test_outline_seeds_callouts_when_llm_returns_empty(monkeypatch):
    # Fixture already includes Cat's EyE chrome + Majestic / callout NPCs.
    text = _load_doctor_who_pdfpig_fixture()

    def fake_call_llm_json(system_prompt, user_prompt, *, warnings, step):
        return {"campaign": {}, "locations": [], "npcs": []}

    monkeypatch.setattr(content_import, "_call_llm_json", fake_call_llm_json)
    result = content_import.outline_campaign_document(text, "create_new")
    assert result["campaign"]["name"] == "Cat's Eye"
    names = {npc["name"] for npc in result["npcs"]}
    assert names >= {"Kate Petersen", "Max Eddings", "Peter Hutchins", "Chad"}
    assert any(loc["id"] == "majestic_hotel" for loc in result["locations"])


def test_seed_and_callouts_still_tolerate_synthetic_glued_columns():
    """Resilience only: mid-word spaces + dual-column glue (not live Cat's Eye PdfPig)."""
    text = _load_doctor_who_pdfpig_glued_synthetic()
    assert "Pe t e r Hu t cHiNs" in text or "t rAi t s" in text
    names = {row["name"] for row in content_import._extract_cubicle7_callout_npcs(text)}
    assert names >= {"Kate Petersen", "Max Eddings", "Peter Hutchins", "Chad"}
    seeded = content_import._seed_npc_detail_from_document("Peter Hutchins", "squatter", text)
    assert seeded is not None
    joined = " ".join(seeded["personality"])
    assert "Friends:" in joined
    assert "Selfish" not in joined


def test_outline_prefers_callouts_over_credit_hallucinations(monkeypatch):
    text = _load_doctor_who_callout_pdf_fixture()

    def fake_call_llm_json(system_prompt, user_prompt, *, warnings, step):
        return {
            "campaign": {
                "campaign_id": "cats_eye",
                "name": "Cat's Eye",
                "description": "A hotel adventure.",
            },
            "locations": [
                {"id": "majestic_hotel", "name": "Majestic Hotel", "type": "building"}
            ],
            "npcs": [
                {
                    "name": "Nick Huggins",
                    "role": "writer",
                    "first_location_id": "majestic_hotel",
                    "other_location_ids": [],
                },
                {
                    "name": "Paul Bourne",
                    "role": "cover artist",
                    "first_location_id": "majestic_hotel",
                    "other_location_ids": [],
                },
            ],
        }

    monkeypatch.setattr(content_import, "_call_llm_json", fake_call_llm_json)
    result = content_import.outline_campaign_document(text, "create_new")
    names = {npc["name"] for npc in result["npcs"]}
    assert names >= {"Kate Petersen", "Max Eddings", "Peter Hutchins", "Chad"}
    assert "Nick Huggins" not in names
    assert "Paul Bourne" not in names
    assert any("callout" in w.lower() for w in result["warnings"])
    assert any("credits/toc" in w.lower() for w in result["warnings"])


def test_normalize_pdf_person_name_repairs_split_tokens():
    assert content_import._normalize_pdf_person_name("Peter Hutc HiNs") == "Peter Hutchins"
    # Live editor PdfPig often keeps the surname token joined with mixed caps.
    assert content_import._normalize_pdf_person_name("Peter HutcHiNs") == "Peter Hutchins"
    assert content_import._normalize_pdf_person_name("KAte PeterseN") == "Kate Petersen"
    assert content_import._normalize_pdf_person_name("KAte Peterse N") == "Kate Petersen"
    assert content_import._normalize_pdf_person_name('"cHAd"') == "Chad"
    # Legitimate multi-word names must not be smashed into one surname.
    assert content_import._normalize_pdf_person_name("Mary Ann Lee") == "Mary Ann Lee"
    assert content_import._normalize_pdf_person_name("Jean Luc Picard") == "Jean Luc Picard"
    assert content_import._normalize_pdf_person_name("Mary Ann Smith") == "Mary Ann Smith"


def test_seed_max_does_not_absorb_neighbor_traits():
    """Packed Cubicle 7 pages must not attach Peter's/Chad's TRAITS to Max."""
    text = _load_doctor_who_catseye_fixture()
    seeded = content_import._seed_npc_detail_from_document("Max Eddings", "squatter", text)
    assert seeded is not None
    joined = " ".join(seeded["personality"] + seeded["llm_rules"])
    assert "Selfish" in joined
    assert "Argumentative" in joined
    assert "Friends:" not in joined and "Friends:" not in " ".join(seeded["personality"])
    assert "Brave:" not in joined
    assert not any("Hutchins has the majority" in p for p in seeded["personality"])


def test_bio_paragraphs_handle_pdf_single_newlines_and_split_names():
    """PdfPig extracts often lack blank lines; headers may split surnames."""
    text = (
        "MAx eddiNGs\n"
        "AWAreNess 3 PreseNce 2\n"
        "trAits\n"
        "Selfish: Eddings thinks of nobody other than himself.\n"
        "tecH LeVeL: 4 stOry POiNts: 8\n"
        "Max Eddings suffers from some unknown psychological complaint that makes him angry.\n"
        "Peter Hutc HiNs\n"
        "AWAreNess 4 PreseNce 4\n"
        "trAits\n"
        "Friends: Hutchins has the majority of the squatters on his side.\n"
        "tecH LeVeL: 4 stOry POiNts: 6\n"
        "Peter Hutchins is something of an enigma – people are drawn to Hutchins.\n"
    )
    max_bios = content_import._bio_paragraphs_for_name(text, "Max Eddings")
    assert max_bios
    assert all("Hutchins is something" not in b for b in max_bios)
    assert all(len(b) <= content_import._BIO_CHUNK_MAX_CHARS for b in max_bios)

    clipped = content_import._clip_npc_writeup(text, "Peter Hutchins")
    assert "Peter Hutc HiNs" in clipped or "Peter Hutchins" in clipped
    assert "Selfish:" not in clipped
    peter_traits = content_import._parse_trait_blocks(clipped, first_section_only=True)
    assert any(name == "Friends" for name, _ in peter_traits)
    assert not any(name == "Selfish" for name, _ in peter_traits)

    peter_seed = content_import._seed_npc_detail_from_document("Peter Hutchins", "leader", text)
    assert peter_seed is not None
    assert any("Friends" in p for p in peter_seed["personality"])
    assert not any("Selfish" in p for p in peter_seed["personality"])


def test_detail_npc_batch_seeds_doctor_who_when_llm_empty(monkeypatch):
    text = _load_doctor_who_catseye_fixture()

    def fake_call_llm_json(system_prompt, user_prompt, *, warnings, step):
        return {"npcs": []}

    monkeypatch.setattr(content_import, "_call_llm_json", fake_call_llm_json)
    # Without seed fallback the hop stays sparse so the editor can retry.
    sparse = content_import.detail_npc_batch(
        text,
        {"campaign_id": "cats_eye", "name": "Cat's Eye"},
        [{"id": "max_eddings", "name": "Max Eddings", "role": "squatter", "first_location_id": "majestic"}],
    )
    assert content_import._npc_detail_is_sparse(sparse["npcs"][0])

    result = content_import.detail_npc_batch(
        text,
        {"campaign_id": "cats_eye", "name": "Cat's Eye"},
        [{"id": "max_eddings", "name": "Max Eddings", "role": "squatter", "first_location_id": "majestic"}],
        allow_seed_fallback=True,
    )
    npc = result["npcs"][0]
    assert any("Selfish" in p for p in npc["personality"])
    assert any("Argumentative" in p for p in npc["personality"])
    assert npc["public_facts"] or npc["known_facts"]
    assert any("last resort" in w.lower() for w in result["warnings"])


def test_mention_indexes_tolerates_pdf_split_hutchins():
    hay = "Peter Hutc HiNs\nFriends: Hutchins has the majority"
    hits = content_import._mention_indexes(hay.casefold(), "Peter Hutchins")
    assert hits, "expected flexible match for PDF-split Hutchins"

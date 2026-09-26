"""Regression tests for campaign document import (content_import.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

import content_import


@pytest.fixture(autouse=True)
def generous_import_budget(monkeypatch):
    """Pin the import character budget so these tests exercise import logic only.

    Otherwise every outline test inherits whatever content_creation model the
    operator last picked in Admin -> LLM, and a model whose Groq TPM cannot host an
    import request fails the whole file for reasons unrelated to parsing. Tests that
    assert budget-specific behavior override this with their own monkeypatch.
    """
    monkeypatch.setattr(content_import, "content_creation_max_input_chars", lambda: 200_000)


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


SHIP_ROOMS_FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "tests"
    / "fixtures"
    / "chariot_of_the_gods_pdf_ship_rooms.txt"
)


def _load_chariot_ship_rooms_fixture() -> str:
    return SHIP_ROOMS_FIXTURE_PATH.read_text(encoding="utf-8")


PDFPIG_FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "tests"
    / "fixtures"
    / "chariot_of_the_gods_pdfpig_npc_section.txt"
)


def _load_chariot_pdfpig_fixture() -> str:
    return PDFPIG_FIXTURE_PATH.read_text(encoding="utf-8")


DESTROYER_MOONSIDE_FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "tests"
    / "fixtures"
    / "destroyer_of_worlds_moonside_locations.txt"
)


def _load_destroyer_moonside_fixture() -> str:
    return DESTROYER_MOONSIDE_FIXTURE_PATH.read_text(encoding="utf-8")
DESTROYER_MEYERS_FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "tests"
    / "fixtures"
    / "destroyer_of_worlds_meyers_pdf_glue.txt"
)


def _load_destroyer_meyers_fixture() -> str:
    return DESTROYER_MEYERS_FIXTURE_PATH.read_text(encoding="utf-8")


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


def test_docnet_toc_npc_header_does_not_block_stat_block_fast_path():
    """PDFium/Docnet TOCs put dotted 'Non-Player Characters ... 12' before the real section."""
    text = (
        "Table of Contents\n"
        "Non-Player Characters ........ 12\n"
        "Deck Plans ........ 24\n\n"
        + _load_chariot_pdfpig_fixture()
    )
    assert content_import._document_has_recoverable_stat_blocks(text)
    result = content_import.outline_campaign_document(text, "create_new")
    names = {npc["name"] for npc in result["npcs"]}
    assert len(names) >= 5
    assert "Albert Johns" in names
    assert any("stat block" in w.lower() for w in result.get("warnings") or [])


def test_destroyer_moonside_seeds_major_location_sites(monkeypatch):
    """Locations-chapter PDFs must seed Fort Nebraska / Spaceport stubs without LLM help."""
    text = _load_destroyer_moonside_fixture()
    assert content_import._document_has_recoverable_location_sites(text)

    def fake_call_llm_json(*args, **kwargs):
        raise AssertionError("locations-chapter outline must skip the LLM")

    monkeypatch.setattr(content_import, "_call_llm_json", fake_call_llm_json)
    result = content_import.outline_campaign_document(text, "create_new")
    names = {loc["name"] for loc in result["locations"]}
    for expected in (
        "Fort Nebraska",
        "Spaceport",
        "Oblivion Bar",
        "Marshal Station",
        "San Rocco Medical Facility",
        "The Insurgent Compound",
        "Oil Refinery",
    ):
        assert expected in names, expected
    # Room-scale headers must not become top-level stubs.
    assert "Hangar 01" not in names
    assert result["campaign"]["genre"] == "sci_fi"
    assert result["campaign"]["rpg_system"] == "Alien RPG"
    assert "The Gate" not in names
    assert "Landing Pad A" not in names
    assert any("moonside" in w.lower() or "location site" in w.lower() for w in result["warnings"])
    assert any("skipping llm outline" in w.lower() for w in result["warnings"])
    assert "destroyer" in str(result["campaign"].get("name") or "").casefold()


def test_destroyer_moonside_map_labels_seed_without_colon_rooms(monkeypatch):
    """Major sites must still seed when PLACE: room lines are missing from the extract."""
    text = """
MOONSIDE LOCATIONS

ARIARCUS COLONY
FORT NEBRASKA
SAN ROCCO MEDICAL FACILITY
MARSHAL STATION
OBLIVION BAR
SPACEPORT
OIL REFINERY
THE INSURGENT COMPOUND
"""
    assert content_import._document_has_recoverable_location_sites(text)
    monkeypatch.setattr(
        content_import,
        "_call_llm_json",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no LLM")),
    )
    result = content_import.outline_campaign_document(text, "create_new")
    names = {loc["name"] for loc in result["locations"]}
    assert "Fort Nebraska" in names
    assert "Spaceport" in names
    assert "Oblivion Bar" in names
    # Map labels must not become the campaign title/id before the Destroyer fallback.
    assert result["campaign"]["name"] == "Destroyer of Worlds"
    assert result["campaign"]["temporary_key"] == "campaign_001"
    assert "campaign_id" not in result["campaign"]


def test_guess_campaign_title_skips_leading_moonside_map_labels():
    text = "FORT NEBRASKA\nSPACEPORT\nARIARCUS\nMOONSIDE LOCATIONS\nTHE GATE: A fence\n"
    assert content_import._guess_campaign_title(text) == "Destroyer of Worlds"
    assert (
        content_import._guess_campaign_title("FORT NEBRASKA\nSPACEPORT\nARIARCUS COLONY\n")
        == "Destroyer of Worlds"
    )


def test_destroyer_colon_rooms_harvest_under_spaceport():
    text = _load_destroyer_moonside_fixture()
    rooms = content_import._harvest_colon_place_subareas(text, location_name="Spaceport")
    names = {row["name"] for row in rooms}
    assert "The Gate" in names
    assert "Landing Pad A" in names
    assert "Hangar 01" in names
    # Other sites' rooms should not bleed into Spaceport.
    assert "Yard" not in names
    assert "Sally Port" not in names


def test_destroyer_important_locations_include_colon_rooms():
    text = _load_destroyer_moonside_fixture()
    harvested = content_import._harvest_important_locations(
        text, "Fort Nebraska", location_type="building"
    )
    names = {row["name"] for row in harvested}
    assert "Sally Port" in names or "Gate" in names
    assert all(row.get("description") for row in harvested)
def test_full_name_accepts_middle_initial_with_period():
    assert content_import._iter_full_names_on_line(
        "FULL NAME: Edward R. Meyers"
    ) == ["Edward R. Meyers"]
    assert content_import._iter_full_names_on_line(
        "FULL NAME:FULL NAME: Edward R. Meyers Edward R. Meyers"
    ) == ["Edward R. Meyers"]
    assert content_import._iter_full_names_on_line(
        "FULL NAME: A. Mallory Eckford"
    ) == ["A. Mallory Eckford"]


def test_act_iii_prose_is_not_a_stat_block_heading():
    assert content_import._iter_stat_block_headings(
        "found in Act III, Meyers has it concealed,"
    ) == []
    assert content_import._iter_stat_block_headings("MEYERS, COLONEL") == [
        ("MEYERS", "COLONEL")
    ]


def test_destroyer_pdf_glue_parses_meyers_not_iii():
    """Destroyer of Worlds two-column glue must not invent NPC 'Iii' from Act III."""
    text = _load_destroyer_meyers_fixture()
    collapsed = content_import._collapse_spaced_caps(text)
    span = content_import._find_npc_section_span(collapsed)
    assert span is not None
    parsed = content_import._extract_alien_rpg_stat_block_npcs(
        collapsed[span[0] : span[1]]
    )
    by_name = {row["name"]: row for row in parsed}
    assert "Iii" not in by_name
    assert "III" not in {row.get("short_name") for row in parsed}
    assert "Edward R. Meyers" in by_name
    meyers = by_name["Edward R. Meyers"]
    assert meyers["short_name"] == "MEYERS"
    assert "colonel" in meyers["role"].casefold() or "fort nebraska" in meyers[
        "role"
    ].casefold()
    assert "thigh" not in meyers["role"].casefold()
    assert "page" not in meyers["role"].casefold()
    assert meyers.get("personality", "").casefold() == "manipulative"
    assert "Jaell" in by_name
    assert by_name["Jaell"].get("personality", "").casefold() == "duplicitous"
    assert "AA83 Syem Davydovich" in by_name
    assert by_name["AA83 Syem Davydovich"].get("personality", "").casefold() == "stoic"


def test_destroyer_outline_seeds_meyers(monkeypatch):
    text = _load_destroyer_meyers_fixture()

    def fake_call_llm_json(*args, **kwargs):
        raise AssertionError("stat-block fast path should not call outline LLM")

    monkeypatch.setattr(content_import, "_call_llm_json", fake_call_llm_json)
    result = content_import.outline_campaign_document(text, "create_new")
    names = {npc["name"] for npc in result["npcs"]}
    assert "Edward R. Meyers" in names
    assert "Iii" not in names
    assert "AA83 Syem Davydovich" in names


def test_find_npc_section_span_prefers_body_over_toc_without_full_name():
    """Generic modules may list NPCs without FULL NAME markers; TOC must not win."""
    text = (
        "Table of Contents\n"
        "Non-Player Characters ........ 12\n"
        "Deck Plans ........ 24\n\n"
        "INTRODUCTION\n"
        "Welcome to the adventure.\n\n"
        "NON-PLAYER CHARACTERS\n\n"
        "JOHN SMITH, GUARD\n"
        "A town guard on patrol.\n\n"
        "JANE DOE, MERCHANT\n"
        "Runs the general store.\n\n"
        "BOB JONES, INNKEEPER\n"
        "Owns the tavern.\n"
    )
    collapsed = content_import._collapse_spaced_caps(text)
    span = content_import._find_npc_section_span(collapsed)
    assert span is not None
    section = collapsed[span[0] : span[1]]
    assert "JOHN SMITH, GUARD" in section
    assert "JANE DOE, MERCHANT" in section
    assert "BOB JONES, INNKEEPER" in section
    assert "........ 12" not in section

    warnings: list[str] = []
    prepared = content_import._prepare_outline_document_text(text, warnings)
    assert "JOHN SMITH, GUARD" in prepared
    assert "--- NON-PLAYER CHARACTERS (authoritative NPC list) ---" in prepared


def test_stat_block_outline_succeeds_when_content_creation_budget_zero(monkeypatch):
    """Deterministic stat-block outline must not require a Groq/local TPM budget."""
    monkeypatch.setattr(content_import, "content_creation_max_input_chars", lambda: 0)
    text = _load_chariot_pdfpig_fixture()
    assert content_import._document_has_recoverable_stat_blocks(text)
    result = content_import.outline_campaign_document(text, "create_new")
    names = {npc["name"] for npc in result["npcs"]}
    assert len(names) >= 5
    assert "Albert Johns" in names
    assert any("stat block" in w.lower() for w in result.get("warnings") or [])


def test_stat_block_outline_ignores_tight_positive_content_creation_budget(monkeypatch):
    """Stat-block fast path must not chop NPC section to the LLM TPM cap."""
    monkeypatch.setattr(content_import, "content_creation_max_input_chars", lambda: 4_000)
    text = _load_chariot_pdfpig_fixture()
    result = content_import.outline_campaign_document(text, "create_new")
    names = {npc["name"] for npc in result["npcs"]}
    assert len(names) >= 5
    assert "Albert Johns" in names


def test_outline_stat_block_probe_bounds_huge_document(monkeypatch):
    """Huge extracts must not fully collapse before the stat-block probe."""
    chariot = _load_chariot_pdfpig_fixture()
    huge = ("A" * 200_000) + chariot
    collapsed_lengths: list[int] = []
    original = content_import._collapse_spaced_caps

    def spy(text: str) -> str:
        collapsed_lengths.append(len(text))
        return original(text)

    monkeypatch.setattr(content_import, "_collapse_spaced_caps", spy)
    result = content_import.outline_campaign_document(huge, "create_new")
    names = {npc["name"] for npc in result["npcs"]}
    assert len(names) >= 5
    assert collapsed_lengths
    assert max(collapsed_lengths) < len(huge)


def test_find_npc_header_offset_detects_split_non_player_characters_lines():
    """PdfPig/Docnet often put NON-PLAYER and CHARACTERS on separate lines."""
    text = _load_chariot_pdfpig_fixture()
    offset = content_import._find_best_npc_section_header_offset(text)
    assert offset is not None
    assert "NON-PLAYER" in text[offset : offset + 40]
    assert "FULL NAME: Albert Johns" in text[offset : offset + 12_000]


def test_find_npc_header_offset_detects_letter_spaced_split_header():
    text = (
        "Front matter\n\n"
        "N O N - P L A Y E R\n"
        "C H A R A C T E R S\n"
        "JOHNS, OFFICER\n"
        "FULL NAME: Albert Johns\n"
        "FULL NAME: Valerie Reid\n"
        "FULL NAME: Liam Flynn\n"
    )
    offset = content_import._find_best_npc_section_header_offset(text)
    assert offset is not None
    assert "Albert Johns" in text[offset : offset + 500]


def test_bounded_text_preserves_mid_document_split_header_section():
    """Huge extracts with a mid-document NPC section must not rely on head/tail only."""
    chariot = _load_chariot_pdfpig_fixture()
    huge = ("A" * 70_000) + chariot + ("B" * 70_000)
    bounded = content_import._bounded_text_for_npc_operations(
        huge, content_import.MAX_INPUT_CHARS
    )
    collapsed = content_import._collapse_spaced_caps(bounded)
    assert "FULL NAME: Albert Johns" in collapsed
    assert content_import._document_has_recoverable_stat_blocks(bounded)


def test_chariot_pdfpig_analyze_imports_all_npcs(monkeypatch):
    """PdfPig extract skips outline but still runs location/NPC detail sweeps."""
    text = _load_chariot_pdfpig_fixture()
    calls: list[str] = []

    def fake_call_llm_json(system_prompt, user_prompt, *, warnings, step, usage_acc=None, **kwargs):
        calls.append(step)
        if step in ("outline", "discovery"):
            raise AssertionError("stat-block fast path should not call the outline LLM")
        if step == "location_details":
            return {"locations": []}
        if step == "npc_details":
            return {"npcs": []}
        return {}

    monkeypatch.setattr(content_import, "_call_llm_json", fake_call_llm_json)
    result = content_import.analyze_campaign_document(text, "create_new")
    assert "outline" not in calls and "discovery" not in calls
    assert "location_details" in calls
    assert "npc_details" in calls
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

    def fake_call_llm_json(system_prompt, user_prompt, *, warnings, step, usage_acc=None, **kwargs):
        calls.append(step)
        if step in ("outline", "discovery"):
            raise AssertionError("stat-block fast path should not call the outline LLM")
        if step == "location_details":
            return {"locations": []}
        if step == "npc_details":
            # Sparse on purpose so PERSONALITY / AGENDA seed fallback still applies.
            return {"npcs": []}
        return {}

    monkeypatch.setattr(content_import, "_call_llm_json", fake_call_llm_json)

    result = content_import.analyze_campaign_document(text, "create_new")
    assert "outline" not in calls and "discovery" not in calls
    assert "location_details" in calls
    assert "npc_details" in calls
    assert "chariot" in str(result["campaign"].get("name") or "").casefold() or "gods" in str(
        result["campaign"].get("name") or ""
    ).casefold()
    assert result["campaign"]["temporary_key"] == "campaign_001"
    names = {npc["name"] for npc in result["npcs"]}
    for expected in CHARIOT_EXPECTED_NPCS:
        assert expected in names
    assert "Ava 6" in names
    johns = next(npc for npc in result["npcs"] if npc["name"] == "Albert Johns")
    assert johns["personality"]
    assert johns["secrets"] or johns["known_facts"]
    loc_names = {loc["name"] for loc in result["locations"]}
    assert any("cronus" in n.casefold() or "sotillo" in n.casefold() for n in loc_names)
    assert all(str(loc.get("temporary_key") or "").startswith("location_") for loc in result["locations"])
    assert "Captain Miller" not in names
    warning_blob = " ".join(result.get("warnings") or []).lower()
    assert "skipping llm outline" in warning_blob
    assert "detail sweep still runs" in warning_blob
    assert "usage" in result
    assert "prompt_tokens" in result["usage"]


def test_outline_failure_recovers_from_stat_blocks(monkeypatch):
    """ALIEN PDF skips outline; detail may fail and still fall back to seeds."""
    text = _load_chariot_pdf_extract_fixture()
    calls: list[str] = []

    def fake_call_llm_json(system_prompt, user_prompt, *, warnings, step, usage_acc=None, **kwargs):
        calls.append(step)
        if step in ("outline", "discovery"):
            raise RuntimeError("should not be called on stat-block fast path")
        if step == "location_details":
            return {"locations": []}
        if step == "npc_details":
            return {"npcs": []}
        return {}

    monkeypatch.setattr(content_import, "_call_llm_json", fake_call_llm_json)
    result = content_import.analyze_campaign_document(text, "create_new")
    assert "outline" not in calls and "discovery" not in calls
    assert "npc_details" in calls
    names = {npc["name"] for npc in result["npcs"]}
    assert "Albert Johns" in names
    assert "Ava 6" in names


def test_stat_block_fast_path_uses_llm_npc_detail_when_rich(monkeypatch):
    """Detail sweep output should win over PERSONALITY/AGENDA seed stubs."""
    text = _load_chariot_pdf_extract_fixture()

    def fake_call_llm_json(system_prompt, user_prompt, *, warnings, step, usage_acc=None, **kwargs):
        if step in ("outline", "discovery"):
            raise AssertionError("outline must be skipped")
        if step == "location_details":
            return {
                "locations": [
                    {
                        "id": "uscss_cronus",
                        "summary": "A derelict science vessel drifting in deep space.",
                        "current_events": ["Life support is failing on the lower decks."],
                    }
                ]
            }
        if step == "npc_details":
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
    cronus = next(
        (loc for loc in result["locations"] if "cronus" in str(loc.get("name") or "").casefold()),
        None,
    )
    if cronus is not None:
        assert "derelict" in cronus["summary"].casefold()
        assert str(cronus.get("temporary_key") or "").startswith("location_")


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


def test_truncate_preserving_npc_section_keeps_document_head_for_title():
    """Leftover budget must keep the document head so title/location callers still work."""
    head = (
        "CHARIOT OF THE GODS\n"
        "A scenario for ALIEN RPG\n"
        "USCSS Cronus is the primary location.\n"
    )
    filler = "Z" * 70_000
    npc = (
        "NON-PLAYER\n"
        "CHARACTERS\n"
        "JOHNS, OFFICER\n"
        "FULL NAME: Albert Johns\n"
        "PERSONALITY: Submissive\n"
        "REID, COLONIAL MARINE\n"
        "FULL NAME: Valerie Reid\n"
        "PERSONALITY: On Edge\n"
        "FLYNN, MEDIC\n"
        "FULL NAME: Liam Flynn\n"
        "PERSONALITY: Fearful\n"
    )
    # Mid-document NPC section with oversized post-header tail (no hard end).
    tail = "T" * 70_000
    text = f"{head}\n{filler}\n{npc}\n{tail}"
    max_chars = 12_000
    trimmed = content_import._truncate_preserving_npc_section(text, max_chars)
    assert len(trimmed) <= max_chars
    assert "CHARIOT OF THE GODS" in trimmed
    assert "USCSS Cronus" in trimmed
    assert "FULL NAME: Albert Johns" in trimmed
    assert content_import._guess_campaign_title(trimmed) == "Chariot of the Gods"
    prepared = content_import._prepare_outline_document_text(trimmed, [])
    assert "CHARIOT OF THE GODS" in prepared
    assert "Albert Johns" in prepared

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

    def fake_call_llm_json(system_prompt, user_prompt, *, warnings, step, usage_acc=None, **kwargs):
        if step in ("outline", "discovery"):
            return bad_outline
        if step == "location_details":
            return {"locations": []}
        if step == "npc_details":
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


def test_outline_truncates_huge_documents_at_start(monkeypatch):
    monkeypatch.setattr(
        content_import,
        "_call_llm_json",
        lambda *args, **kwargs: {"campaign": {}, "locations": [], "npcs": []},
    )
    # TPM at/above hop max so only the MAX_INPUT_CHARS cap applies once.
    monkeypatch.setattr(content_import, "content_creation_max_input_chars", lambda: 200_000)
    huge = "a" * 150_000
    result = content_import.outline_campaign_document(huge, "create_new")
    truncate_warnings = [
        w for w in (result.get("warnings") or []) if "truncated" in w.lower()
    ]
    assert len(truncate_warnings) == 1


def test_outline_does_not_duplicate_truncation_warning_when_tpm_also_caps(monkeypatch):
    """Oversized docs that miss the stat-block path must not warn twice for one truncate."""
    monkeypatch.setattr(
        content_import,
        "_call_llm_json",
        lambda *args, **kwargs: {"campaign": {}, "locations": [], "npcs": []},
    )
    monkeypatch.setattr(content_import, "content_creation_max_input_chars", lambda: 8_000)
    huge = "a" * 150_000
    result = content_import.outline_campaign_document(huge, "create_new")
    truncate_warnings = [
        w for w in (result.get("warnings") or []) if "truncated" in w.lower()
    ]
    # MAX_INPUT_CHARS cap, then TPM shrink of the already-bounded text — two
    # distinct budgets is fine; two identical "was 150000" messages is not.
    assert len(truncate_warnings) <= 2
    assert len({w for w in truncate_warnings}) == len(truncate_warnings)
    assert sum(1 for w in truncate_warnings if "150000" in w or "150,000" in w) == 1


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

    def fake_call_llm_json(system_prompt, user_prompt, *, warnings, step, usage_acc=None, **kwargs):
        if step in ("outline", "discovery"):
            return outline
        if step == "location_details":
            return {"locations": []}
        if step == "npc_details":
            return {"npcs": []}
        return {}

    monkeypatch.setattr(content_import, "_call_llm_json", fake_call_llm_json)

    result = content_import.analyze_campaign_document(text, "create_new")
    names = {npc["name"] for npc in result["npcs"]}
    assert "Archdeacon Verity-9" in names
    assert result["npcs"][0]["temporary_key"] == "npc_001"
    assert result["npcs"][0]["id"] == "npc_001"
    assert "campaign_id" not in result["campaign"]
    assert result["campaign"]["temporary_key"] == "campaign_001"


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

    def fake_call_llm_json(system_prompt, user_prompt, *, warnings, step, usage_acc=None, **kwargs):
        if step in ("outline", "discovery"):
            return outline
        if step == "location_details":
            return {"locations": []}
        if step == "npc_details":
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

    def fake_call_llm_json(system_prompt, user_prompt, *, warnings, step, usage_acc=None, **kwargs):
        if step in ("outline", "discovery"):
            return outline
        if step == "location_details":
            return {"locations": []}
        if step == "npc_details":
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

    def fake_call_llm_json(system_prompt, user_prompt, *, warnings, step, usage_acc=None, **kwargs):
        steps.append(step)
        if step in ("outline", "discovery"):
            raise AssertionError("stat-block outline must be skipped")
        if step == "location_details":
            return {"locations": []}
        if step == "npc_details":
            return {"npcs": []}
        return {}

    monkeypatch.setattr(content_import, "_call_llm_json", fake_call_llm_json)

    outline = content_import.outline_campaign_document(text, "create_new")
    assert "outline" not in steps and "discovery" not in steps
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
    assert "location_details" in steps

    johns = next(npc for npc in outline["npcs"] if npc["name"] == "Albert Johns")
    # First hop is LLM-only (editor may retry separately when sparse).
    sparse = content_import.detail_npc_batch(text, campaign, [johns])
    assert sparse["npcs"][0]["name"] == "Albert Johns"
    assert content_import._npc_detail_is_sparse(sparse["npcs"][0])
    assert "npc_details" in steps

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

    def fake_call_llm_json(system_prompt, user_prompt, *, warnings, step, usage_acc=None, **kwargs):
        calls.append(step)
        return {"locations": []}

    monkeypatch.setattr(content_import, "_call_llm_json", fake_call_llm_json)
    sparse = content_import.detail_location_batch(
        "USCSS Cronus drifts in deep space with failing life support.",
        {"campaign_id": "c", "name": "C"},
        [{"id": "uscss_cronus", "name": "USCSS Cronus", "type": "building", "seeded_from_stat_blocks": True}],
    )
    assert calls == ["location_details"]
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


def test_location_looks_like_ship_from_name_and_type():
    assert content_import._location_looks_like_ship("USCSS Cronus", location_type="ship")
    assert content_import._location_looks_like_ship("The Sotillo", location_type="building")
    assert not content_import._location_looks_like_ship("Majestic Hotel", location_type="building")


def test_ship_room_name_key_normalizes_apostrophes_and_hyphens():
    assert content_import._ship_room_name_key("Clayton's Secret Stash") == "clayton s secret stash"
    assert content_import._ship_room_name_key("Walk-In Closet") == "walk in closet"
    assert content_import._is_ship_false_room_name("Clayton's Secret Stash")
    assert content_import._is_ship_false_room_name("Walk-In Closet")
    assert content_import._is_ship_false_room_name("Pandora's Box")
    assert not content_import._is_ship_false_room_name("Storage Units")


def test_harvest_ship_rooms_include_cargo_deck():
    text = _load_chariot_ship_rooms_fixture()
    harvested = content_import._harvest_important_locations(
        text,
        "USCSS Cronus",
        location_type="ship",
    )
    names = {row["name"].casefold() for row in harvested}
    assert any("storage units" in n for n in names)
    assert any("cargo bays" in n for n in names)
    assert any("reactor relay" in n for n in names)
    assert len(harvested) >= 20


def test_merge_location_detail_filters_apostrophe_false_room_names():
    merged = content_import._merge_location_detail(
        {"id": "uscss_cronus", "name": "USCSS Cronus", "type": "ship"},
        {
            "summary": "A ship.",
            "important_locations": [
                {"name": "Bridge", "description": "Command.", "public_knowledge": ""},
                {"name": "Clayton's Secret Stash", "description": "Hidden urns.", "public_knowledge": ""},
                {"name": "Walk-In Closet", "description": "Escape pod.", "public_knowledge": ""},
            ],
        },
        "chariot",
        document_text="",
    )
    names = {row["name"].casefold() for row in merged["important_locations"]}
    assert "bridge" in names
    assert not any("clayton" in n for n in names)
    assert not any("closet" in n for n in names)


def test_harvest_ship_rooms_include_deck_in_description():
    """ALIEN-style deck plans: ship stays parent; rooms become important_locations with deck."""
    text = _load_chariot_ship_rooms_fixture()
    assert content_import._location_looks_like_ship(
        "USCSS Cronus",
        location_type="ship",
        document_text=text,
    )
    harvested = content_import._harvest_important_locations(
        text,
        "USCSS Cronus",
        location_type="ship",
    )
    names = {row["name"].casefold() for row in harvested}
    assert any("science lab" in n for n in names)
    assert any("airlock" in n for n in names)

    airlock = next(row for row in harvested if "airlock" in row["name"].casefold())
    assert airlock["description"].casefold().startswith("deck a")
    assert "point of entry" in airlock["description"].casefold() or "airlock" in airlock[
        "description"
    ].casefold()

    bridge = next(row for row in harvested if row["name"].casefold() == "bridge")
    assert len(bridge["description"]) > 40
    assert "power" in bridge["description"].casefold() or "bridge" in bridge["description"].casefold()
    # Nested bridge stations / boxed callouts should not crowd out real rooms.
    assert not any(n == "dark" for n in names)
    assert not any(n == "command station" for n in names)

    # Map chrome / technical specs must not become rooms.
    assert not any("technical specifications" in n for n in names)
    assert not any("map legend" in n for n in names)


def test_location_detail_excerpt_includes_ship_room_block():
    text = _load_chariot_ship_rooms_fixture()
    excerpt = content_import._location_detail_excerpt(
        text,
        {"id": "uscss_cronus", "name": "USCSS Cronus", "type": "ship"},
    )
    assert "MAIN AIRLOCK" in excerpt or "Main Airlock" in excerpt
    assert "BRIDGE" in excerpt or "Bridge" in excerpt


def test_collapse_letter_spaced_ship_headers():
    collapsed = content_import._collapse_letter_spaced_phrase(
        "M A I N A I R Lo C K J U N C T I O N A- 1"
    )
    assert "MAIN AIRLOCK" in collapsed.upper()
    assert "JUNCTION" in collapsed.upper()
    parsed = content_import._looks_like_ship_room_header(
        "M A I N A I R Lo C K J U N C T I O N A- 1: The main airlock is the only point of entry."
    )
    assert parsed is not None
    assert "airlock" in parsed[0].casefold()
    assert "junction" in parsed[0].casefold()
    assert content_import._deck_from_ship_room_name(parsed[0]) == "Deck A"


def test_ship_harvest_rejects_act_and_event_headers():
    text = (
        "USCSS Cronus\nTHE DECK PLANS: The Cronus has four decks.\n"
        "MAIN AIRLOCK, JUNCTION A-1: The main airlock is the only point of entry to the Cronus "
        "and it is located on the dorsal side of the ship with a damaged outer door.\n"
        "BRIDGE: Until power is restored, the bridge is on battery reserve and shut down with "
        "faint orange standby illumination around the viewports.\n"
        "ACT I\nPANDORA'S BOX: Eager to be done with this run and collect a paycheck, the crew "
        "of the Montero wakes up to prep the ship for unloading.\n"
        "PING: Suddenly, MU/TH/UR reports an approaching ship when there is none.\n"
        "BAD AIR: Aboard the Cronus, everyone has trouble taking deep breaths.\n"
    )
    harvested = content_import._harvest_important_locations(
        text,
        "USCSS Cronus",
        location_type="ship",
    )
    names = {row["name"].casefold() for row in harvested}
    assert any("airlock" in n for n in names)
    assert "bridge" in names
    assert not any(n.startswith("act") for n in names)
    assert "ping" not in names
    assert "bad air" not in names
    airlock = next(row for row in harvested if "airlock" in row["name"].casefold())
    assert airlock["description"].casefold().startswith("deck a")
    bridge = next(row for row in harvested if row["name"].casefold() == "bridge")
    assert not bridge["description"].casefold().startswith("deck d")


def test_ship_room_harvest_scoped_to_parent_ship():
    """Sotillo must not inherit Cronus deck rooms just because both are ships."""
    text = (
        "NON-PLAYER CHARACTERS\n"
        "Mechanic on the Sotillo\n"
        "The Sotillo is a battered freighter used by Bolaji's crew.\n\n"
        + _load_chariot_ship_rooms_fixture()
    )
    cronus = content_import._harvest_important_locations(
        text,
        "USCSS Cronus",
        location_type="ship",
    )
    sotillo = content_import._harvest_important_locations(
        text,
        "Sotillo",
        location_type="ship",
    )
    assert any("airlock" in row["name"].casefold() for row in cronus)
    assert any(row["name"].casefold() == "bridge" for row in cronus)
    assert sotillo == []
    assert content_import._find_ship_room_description_span(text, location_name="Sotillo") is None

    sotillo_excerpt = content_import._location_detail_excerpt(
        text,
        {"id": "sotillo", "name": "Sotillo", "type": "ship"},
    )
    assert "MAIN AIRLOCK" not in sotillo_excerpt
    assert "BRIDGE:" not in sotillo_excerpt
    assert "sotillo" in sotillo_excerpt.casefold()


def test_location_detail_prompt_mentions_ship_deck_guidance():
    prompt = content_import._build_location_detail_prompt(
        [{"id": "uscss_cronus", "name": "USCSS Cronus", "type": "ship"}],
        "THE DECK PLANS: The Cronus has four decks.\nMAIN AIRLOCK, JUNCTION A-1: The main airlock.",
        {"campaign_id": "chariot", "name": "Chariot of the Gods"},
    )
    assert "SHIP / VESSEL LAYOUT" in prompt
    assert "important_locations" in prompt
    assert "Deck" in prompt


def test_location_detail_merges_ship_rooms_when_llm_omits_them(monkeypatch):
    text = _load_chariot_ship_rooms_fixture()

    def fake_call_llm_json(system_prompt, user_prompt, *, warnings, step, usage_acc=None, **kwargs):
        assert "SHIP / VESSEL LAYOUT" in user_prompt
        return {
            "locations": [
                {
                    "id": "uscss_cronus",
                    "summary": "A derelict Weyland science exploration vessel.",
                    "current_events": ["Life support is offline."],
                    "important_locations": [],
                    "rumors": [],
                    "local_tone": ["cold"],
                    "knowledge_rules": [],
                }
            ]
        }

    monkeypatch.setattr(content_import, "_call_llm_json", fake_call_llm_json)
    result = content_import.detail_location_batch(
        text,
        {"campaign_id": "chariot", "name": "Chariot of the Gods"},
        [{"id": "uscss_cronus", "name": "USCSS Cronus", "type": "ship"}],
    )
    loc = result["locations"][0]
    names = {row["name"].casefold() for row in loc["important_locations"]}
    assert any("airlock" in n for n in names)
    assert any("bridge" in n for n in names)
    airlock = next(row for row in loc["important_locations"] if "airlock" in row["name"].casefold())
    assert "deck" in airlock["description"].casefold()


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

    def fake_call_llm_json(system_prompt, user_prompt, *, warnings, step, usage_acc=None, **kwargs):
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
        {
            "campaign_id": "MultiTest",
            "temporary_key": "campaign_001",
            "name": "Multi Test",
            "description": "",
            "genre": "sci_fi",
            "rpg_system": "Alien RPG",
        }
    )
    assert meta["campaign_id"] == "MultiTest"
    assert meta["temporary_key"] == "campaign_001"
    assert meta["genre"] == "sci_fi"
    assert meta["rpg_system"] == "Alien RPG"


def test_normalize_campaign_meta_keeps_temporary_key():
    meta = content_import._normalize_campaign_meta(
        {"temporary_key": "campaign_001", "name": "Hope's Last Day"}
    )
    assert meta["temporary_key"] == "campaign_001"
    assert "campaign_id" not in meta
    assert meta["name"] == "Hope's Last Day"


def test_normalize_campaign_meta_leaves_genre_empty_when_missing():
    meta = content_import._normalize_campaign_meta({"name": "Multi Test"})
    assert "campaign_id" not in meta
    assert meta["temporary_key"] == "campaign_001"
    assert meta["name"] == "Multi Test"
    assert meta["genre"] == ""
    assert meta["rpg_system"] == ""


def test_apply_imported_campaign_setting_alien_fast_path_is_sci_fi():
    campaign = {"campaign_id": "destroyer", "name": "Destroyer", "genre": "", "rpg_system": ""}
    content_import._apply_imported_campaign_setting(
        campaign, document_text="", alien_fast_path=True
    )
    assert campaign["genre"] == "sci_fi"
    assert campaign["rpg_system"] == "Alien RPG"


def test_apply_imported_campaign_setting_does_not_invent_fantasy():
    campaign = {"campaign_id": "camp", "name": "Camp", "genre": "", "rpg_system": ""}
    content_import._apply_imported_campaign_setting(
        campaign, document_text="A quiet village market.", alien_fast_path=False
    )
    assert campaign["genre"] == ""
    assert campaign["rpg_system"] == ""


def test_apply_imported_campaign_setting_ignores_free_league_alone():
    """Free League publishes many lines — publisher credit must not stamp Alien RPG."""
    campaign = {"campaign_id": "vaesen", "name": "Vaesen", "genre": "", "rpg_system": ""}
    content_import._apply_imported_campaign_setting(
        campaign,
        document_text=(
            "VAESEN\nPublished by Free League Publishing.\n"
            "A Nordic horror mystery in 19th century Scandinavia."
        ),
        alien_fast_path=False,
    )
    assert campaign["genre"] == ""
    assert campaign["rpg_system"] == ""


def test_apply_imported_campaign_setting_alien_markers_still_detect():
    campaign = {"campaign_id": "chariot", "name": "Chariot", "genre": "", "rpg_system": ""}
    content_import._apply_imported_campaign_setting(
        campaign,
        document_text="Aboard the USCSS Cronus, a xenomorph outbreak begins.",
        alien_fast_path=False,
    )
    assert campaign["genre"] == "sci_fi"
    assert campaign["rpg_system"] == "Alien RPG"

def test_stat_block_fast_path_outline_stamps_sci_fi_setting(monkeypatch):
    text = (
        "Table of Contents\n"
        "Non-Player Characters ........ 12\n\n"
        + _load_chariot_pdfpig_fixture()
    )

    def fake_call_llm_json(*args, **kwargs):
        raise AssertionError("stat-block outline must skip the LLM")

    monkeypatch.setattr(content_import, "_call_llm_json", fake_call_llm_json)
    result = content_import.outline_campaign_document(text, "create_new")
    assert result["campaign"]["genre"] == "sci_fi"
    assert result["campaign"]["rpg_system"] == "Alien RPG"

def test_normalize_npc_stubs_empty_location_stays_empty():
    stubs = content_import._normalize_npc_stubs(
        [{"id": "albert_johns", "name": "Albert Johns", "role": "Officer", "first_location_id": ""}]
    )
    assert stubs[0]["first_location_id"] == ""
    assert content_import.snake_case("") == "unnamed"


def test_detail_npc_batch_preserves_campaign_id_and_empty_location(monkeypatch):
    def fake_call_llm_json(system_prompt, user_prompt, *, warnings, step, usage_acc=None, **kwargs):
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

    def fake_call_llm_json(system_prompt, user_prompt, *, warnings, step, usage_acc=None, **kwargs):
        return {"campaign": {}, "locations": [], "npcs": []}

    monkeypatch.setattr(content_import, "_call_llm_json", fake_call_llm_json)
    result = content_import.outline_campaign_document(text, "create_new")
    assert result["campaign"]["name"] == "Cat's Eye"
    names = {npc["name"] for npc in result["npcs"]}
    assert names >= {"Kate Petersen", "Max Eddings", "Peter Hutchins", "Chad"}
    assert any(
        (loc.get("name") or "").casefold() == "majestic hotel"
        or "majestic" in (loc.get("name") or "").casefold()
        for loc in result["locations"]
    )


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

    def fake_call_llm_json(system_prompt, user_prompt, *, warnings, step, usage_acc=None, **kwargs):
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

    def fake_call_llm_json(system_prompt, user_prompt, *, warnings, step, usage_acc=None, **kwargs):
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


def test_create_new_normalize_ignores_llm_campaign_id_and_assigns_temp_keys():
    """create_new must not treat LLM-invented campaign_id as authoritative."""
    outline = {
        "campaign": {
            "campaign_id": "llm_invented_permanent_id",
            "name": "Hope's Last Day",
            "description": "A tense sci-fi one-shot.",
            "genre": "sci_fi",
            "rpg_system": "Alien RPG",
        },
        "locations": [
            {"id": "hope_station", "name": "Hope Station", "type": "ship"},
            {"name": "Medbay", "type": "building", "parent_temporary_key": "hope_station"},
        ],
        "npcs": [
            {
                "name": "Captain Reyes",
                "role": "captain",
                "first_location_id": "hope_station",
                "significance": "conversational",
            },
            {
                "name": "Facehugger Swarm",
                "role": "xenomorph",
                "first_location_id": "hope_station",
                "significance": "ignore",
            },
        ],
    }
    warnings: list[str] = []
    adventures, locations, npcs = content_import._normalize_outline(
        outline,
        mode="create_new",
        existing_campaign_id=None,
        existing_location_ids=[],
        existing_npc_names=[],
        warnings=warnings,
    )
    campaign = adventures[0]
    assert "campaign_id" not in campaign
    assert campaign["name"] == "Hope's Last Day"
    adventures, locations, npcs = content_import._stamp_discovery_temporary_keys(
        adventures,
        locations,
        npcs,
        mode="create_new",
        existing_campaign_id=None,
        warnings=warnings,
    )
    campaign = adventures[0]
    assert campaign["temporary_key"] == "adventure_001"
    assert "campaign_id" not in campaign
    assert [loc["temporary_key"] for loc in locations] == ["location_001", "location_002"]
    assert all(loc["id"] == loc["temporary_key"] for loc in locations)
    assert locations[1]["parent_temporary_key"] == "location_001"
    assert all(loc.get("adventure_temporary_key") == "adventure_001" for loc in locations)
    assert [npc["temporary_key"] for npc in npcs] == ["npc_001", "npc_002"]
    assert npcs[0]["primary_location_key"] == "location_001"
    assert npcs[0]["first_location_id"] == "location_001"
    assert npcs[1]["significance"] == "ignore"
    assert "usage" not in warnings


def test_stamp_remaps_associated_npc_keys_onto_temp_handles():
    campaign = {"title": "Hope", "name": "Hope"}
    locations = [
        {
            "name": "Hadley's Hope",
            "type": "colony",
            "associated_npc_keys": ["theodora", "operations_guard"],
        },
        {"name": "Operations", "type": "building", "parent_temporary_key": "Hadley's Hope"},
    ]
    npcs = [
        {
            "name": "Theodora",
            "temporary_key": "theodora",
            "primary_location_key": "Hadley's Hope",
            "significance": "conversational",
        },
        {
            "name": "Guard",
            "temporary_key": "operations_guard",
            "primary_location_key": "Operations",
            "significance": "conversational",
        },
    ]
    warnings: list[str] = []
    _, stamped_locs, stamped_npcs = content_import._stamp_discovery_temporary_keys(
        campaign,
        locations,
        npcs,
        mode="create_new",
        existing_campaign_id=None,
        warnings=warnings,
    )
    assert stamped_locs[0]["associated_npc_keys"] == ["npc_001", "npc_002"]
    assert stamped_npcs[0]["temporary_key"] == "npc_001"
    assert stamped_npcs[1]["temporary_key"] == "npc_002"
    assert stamped_locs[0]["adventure_temporary_key"] == "adventure_001"


def test_normalize_outline_resolves_parent_when_merge_skips_existing():
    outline = {
        "campaign": {"name": "Existing Camp", "title": "Existing Camp"},
        "locations": [
            {"temporary_key": "old_town", "name": "Old Town", "type": "city"},
            {
                "name": "New Tavern",
                "type": "building",
                "parent_temporary_key": "old_town",
            },
        ],
        "npcs": [],
    }
    warnings: list[str] = []
    _, locations, _ = content_import._normalize_outline(
        outline,
        mode="merge",
        existing_campaign_id="existing_camp",
        existing_location_ids=["old_town"],
        existing_npc_names=[],
        warnings=warnings,
    )
    assert len(locations) == 1
    assert locations[0]["name"] == "New Tavern"
    assert locations[0]["parent_temporary_key"] == "old_town"


def test_analyze_merges_usage_across_sparse_retries(monkeypatch):
    """Sparse detail retries must accumulate tokens from every hop."""
    calls = {"loc": 0}

    def fake_outline(*_args, **_kwargs):
        return {
            "campaign": {"temporary_key": "campaign_001", "title": "T", "name": "T"},
            "locations": [
                {"temporary_key": "location_001", "name": "Town", "type": "city"}
            ],
            "npcs": [],
            "warnings": [],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "estimated_cost_usd": 0.01,
                "model_id": "outline",
            },
        }

    def fake_detail_location(_text, _campaign, _batch, is_retry=False, allow_seed_fallback=False):
        calls["loc"] += 1
        sparse = calls["loc"] < 3
        return {
            "locations": [
                {
                    "temporary_key": "location_001",
                    "id": "location_001",
                    "name": "Town",
                    "type": "city",
                    "summary": "" if sparse else "A quiet town.",
                    "current_events": [],
                    "rumors": [],
                    "local_tone": [],
                    "knowledge_rules": [],
                    "important_locations": [],
                }
            ],
            "warnings": [],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "estimated_cost_usd": 0.02,
                "model_id": "detail",
            },
        }

    monkeypatch.setattr(content_import, "outline_campaign_document", fake_outline)
    monkeypatch.setattr(content_import, "detail_location_batch", fake_detail_location)

    result = content_import.analyze_campaign_document("x" * 80, "create_new")
    # outline 10/5 + three detail hops 100/50 each
    assert result["usage"]["prompt_tokens"] == 10 + 300
    assert result["usage"]["completion_tokens"] == 5 + 150
    assert calls["loc"] == 3


def test_validate_temporary_keys_detects_duplicates():
    warnings = content_import._validate_temporary_keys(
        {"temporary_key": "campaign_001"},
        [
            {"temporary_key": "location_001", "name": "A"},
            {"temporary_key": "location_001", "name": "B"},
        ],
        [{"temporary_key": "npc_001", "name": "X"}],
    )
    assert any("Duplicate temporary_key 'location_001'" in w for w in warnings)


def test_outline_create_new_assigns_temp_keys_without_llm_campaign_id(monkeypatch):
    def fake_call_llm_json(system_prompt, user_prompt, *, warnings, step, usage_acc=None, **kwargs):
        assert "Do NOT invent a permanent campaign_id" in user_prompt or "temporary_key" in user_prompt
        assert "campaign_id" not in user_prompt.split("Rules:")[0] or "Do NOT invent" in user_prompt
        if usage_acc is not None:
            usage_acc["prompt_tokens"] = 11
            usage_acc["completion_tokens"] = 7
            usage_acc["model_id"] = "test-model"
        return {
            "campaign": {
                "campaign_id": "should_be_ignored",
                "title": "Hope's Last Day",
                "name": "Hope's Last Day",
                "description": "Desc",
                "genre": "sci_fi",
                "rpg_system": "Alien RPG",
            },
            "locations": [{"name": "Hope Station", "type": "ship"}],
            "npcs": [
                {
                    "name": "Reyes",
                    "role": "captain",
                    "primary_location_key": "location_001",
                    "significance": "conversational",
                }
            ],
        }

    monkeypatch.setattr(content_import, "_call_llm_json", fake_call_llm_json)
    result = content_import.outline_campaign_document(
        "Hope's Last Day\nA short module about Hope Station.",
        "create_new",
    )
    assert result["campaign"]["temporary_key"] == "adventure_001"
    assert "campaign_id" not in result["campaign"]
    assert result["campaign"]["name"] == "Hope's Last Day"
    assert result["adventures"][0]["temporary_key"] == "adventure_001"
    assert result["locations"][0]["temporary_key"] == "location_001"
    assert result["npcs"][0]["temporary_key"] == "npc_001"
    assert result["npcs"][0]["primary_location_key"] == "location_001"
    assert result["usage"]["prompt_tokens"] == 11
    assert result["usage"]["completion_tokens"] == 7
    assert result["usage"]["model_id"] == "test-model"
    assert result["relationships"] == []
    assert result["events"] == []


def test_detail_preserves_temporary_key(monkeypatch):
    def fake_call_llm_json(system_prompt, user_prompt, *, warnings, step, usage_acc=None, **kwargs):
        if usage_acc is not None:
            usage_acc["prompt_tokens"] = 3
            usage_acc["completion_tokens"] = 2
            usage_acc["model_id"] = "detail-model"
        return {
            "locations": [
                {
                    "temporary_key": "location_001",
                    "summary": "A battered freighter.",
                    "current_events": ["Life support flickering."],
                }
            ]
        }

    monkeypatch.setattr(content_import, "_call_llm_json", fake_call_llm_json)
    result = content_import.detail_location_batch(
        "Hope Station is a battered freighter with flickering life support.",
        {"temporary_key": "campaign_001", "name": "Hope's Last Day"},
        [{"temporary_key": "location_001", "name": "Hope Station", "type": "ship"}],
    )
    loc = result["locations"][0]
    assert loc["temporary_key"] == "location_001"
    assert loc["id"] == "location_001"
    assert loc["campaign_id"] == "campaign_001"
    assert "battered" in loc["summary"].casefold()
    assert result["usage"]["prompt_tokens"] == 3


def test_infer_npc_identity_falconmoon_excerpt():
    """Short HotDQ-style prose must yield male / half-elf / calm personality."""
    excerpt = (
        "Eadyan Falconmoon, a level-headed half-elf.\n"
        "He's easy to spot, being the only calm person they can\n"
        "find in the temple, and he is elated to see them. He looks\n"
        "to the characters for a plan."
    )
    # Simulate LLM inventing nonbinary + they/them + unknown race.
    inferred = content_import._infer_npc_identity_from_excerpt(
        excerpt,
        {
            "gender": "nonbinary",
            "race": "unknown",
            "pronouns": {"subject": "they", "object": "them"},
            "personality": [],
        },
        name="Eadyan Falconmoon",
    )
    assert inferred["gender"] == "male"
    assert inferred["race"] == "half-elf"
    assert "calm" in inferred["personality"]
    assert "level-headed" in inferred["personality"]
    assert inferred.get("pronouns") is None  # cleared so merge applies he/him


def test_infer_npc_identity_ignores_neighbor_writeup():
    excerpt = (
        "=== PAGE 8 ===\n"
        "Sister Mira, a stern dwarf. She guards the gate.\n\n"
        "=== PAGE 9 ===\n"
        "Eadyan Falconmoon greets the party. He is calm.\n"
    )
    inferred = content_import._infer_npc_identity_from_excerpt(
        excerpt,
        {"gender": "unspecified", "race": "unknown", "personality": []},
        name="Eadyan Falconmoon",
    )
    assert inferred["gender"] == "male"
    assert inferred.get("race", "unknown") in {"unknown", ""}  # dwarf must not leak
    assert "calm" in inferred.get("personality", [])


def test_identity_cue_prefers_writeup_over_later_scene_in_reordered_excerpt():
    """Hit-first page ordering puts writeup then a later scene; do not use last hit."""
    # Production-like reorder: writeup page emitted before scene page, then a neighbor
    # page with another NPC that would poison a naive 900-char forward slice.
    excerpt = (
        "=== PAGE 8 ===\n"
        "Eadyan Falconmoon, a level-headed half-elf.\n"
        "He's easy to spot and he is calm.\n\n"
        "=== PAGE 2 ===\n"
        "The party meets Eadyan Falconmoon near the doors.\n\n"
        "=== PAGE 9 ===\n"
        "Sister Mira, a stern dwarf. She guards the gate alone.\n"
    )
    cue = content_import._identity_cue_text(excerpt, "Eadyan Falconmoon")
    assert "half-elf" in cue.casefold()
    assert "Sister Mira" not in cue
    inferred = content_import._infer_npc_identity_from_excerpt(
        excerpt,
        {"gender": "nonbinary", "race": "unknown", "personality": []},
        name="Eadyan Falconmoon",
    )
    assert inferred["gender"] == "male"
    assert inferred["race"] == "half-elf"
    assert inferred.get("race") != "dwarf"


def test_infer_npc_identity_overrides_hyphenated_nonbinary():
    excerpt = "Brother Marcus is a male monk. He greets the party with a calm manner."
    inferred = content_import._infer_npc_identity_from_excerpt(
        excerpt,
        {
            "gender": "non-binary",
            "pronouns": {"subject": "they", "object": "them"},
            "personality": [],
        },
        name="Brother Marcus",
    )
    assert inferred["gender"] == "male"
    assert inferred.get("pronouns") is None
    assert content_import._canonicalize_import_gender("enby") == "nonbinary"

def test_merge_npc_detail_overrides_invented_nonbinary(monkeypatch):
    excerpt = (
        "Brother Marcus is a male monk. He greets the party with a calm manner "
        "and offers them shelter in the temple."
    )

    def fake_call_llm_json(system_prompt, user_prompt, *, warnings, step, usage_acc=None, **kwargs):
        return {
            "npcs": [
                {
                    "temporary_key": "marcus",
                    "gender": "nonbinary",
                    "race": "unknown",
                    "pronouns": {
                        "subject": "they",
                        "object": "them",
                        "possessive_adjective": "their",
                        "possessive_pronoun": "theirs",
                        "reflexive": "themself",
                    },
                    "personality": [],
                    "public_facts": ["Priest in the temple."],
                    "known_facts": [],
                    "secrets": [],
                    "llm_rules": [],
                    "voice": [],
                }
            ]
        }

    monkeypatch.setattr(content_import, "_call_llm_json", fake_call_llm_json)
    result = content_import.detail_npc_batch(
        excerpt,
        {"campaign_id": "test_camp", "name": "Test"},
        [
            {
                "temporary_key": "marcus",
                "name": "Brother Marcus",
                "role": "monk",
                "first_location_id": "temple",
            }
        ],
    )
    npc = result["npcs"][0]
    assert npc["gender"] == "male"
    assert npc["pronouns"]["subject"] == "he"
    assert "calm" in npc["personality"]


def test_normalize_pronouns_unspecified_stays_empty():
    assert content_import._normalize_pronouns(None, "unspecified") == {}
    assert content_import._normalize_pronouns("", "unknown") == {}
    assert content_import._normalize_pronouns(None, "male")["subject"] == "he"


def test_npc_writeup_excerpt_uses_page_neighborhood_for_popout():
    """Callout on a distant PAGE still lands in the excerpt (±2 around every hit)."""
    text = (
        "=== PAGE 1 ===\nIntro\n"
        "=== PAGE 2 ===\nThe party meets Eadyan Falconmoon at the doors.\n"
        "=== PAGE 3 ===\nCombat\n"
        "=== PAGE 4 ===\nTravel\n"
        "=== PAGE 5 ===\nTravel more\n"
        "=== PAGE 6 ===\nSidebar\n"
        "=== PAGE 7 ===\nSidebar\n"
        "=== PAGE 8 ===\nEadyan Falconmoon, a level-headed half-elf. He is calm.\n"
        "=== PAGE 9 ===\nNext\n"
        "=== PAGE 10 ===\nNear\n"
        "=== PAGE 11 ===\nFar away epilogue\n"
    )
    excerpt = content_import._npc_writeup_excerpt(text, "Eadyan Falconmoon")
    assert "half-elf" in excerpt.casefold()
    assert "=== PAGE 2 ===" in excerpt
    assert "=== PAGE 8 ===" in excerpt
    assert "=== PAGE 10 ===" in excerpt
    assert "Far away epilogue" not in excerpt


def test_format_page_neighborhood_prioritizes_hit_pages():
    filler = "Front matter filler line.\n" * 400
    text = (
        f"=== PAGE 1 ===\n{filler}"
        "=== PAGE 2 ===\nTitle\n"
        "=== PAGE 3 ===\nEadyan Falconmoon is calm.\n"
    )
    out = content_import._format_page_neighborhood(
        text, {3}, pages_around=2, max_chars=3_000
    )
    assert "Eadyan Falconmoon" in out
    assert out.index("Eadyan") < out.index("Front matter") or "Front matter" not in out


def test_coerce_adventure_stubs_wraps_legacy_campaign():
    outline = {"campaign": {"title": "Sunless Citadel", "name": "Sunless Citadel", "genre": "fantasy"}}
    stubs = content_import._coerce_adventure_stubs(outline)
    assert len(stubs) == 1
    assert stubs[0]["title"] == "Sunless Citadel"


def test_normalize_and_stamp_anthology_adventures():
    outline = {
        "adventures": [
            {"temporary_key": "adv_a", "title": "Sunless Citadel", "name": "Sunless Citadel"},
            {"temporary_key": "adv_b", "title": "Forge of Fury", "name": "Forge of Fury"},
        ],
        "locations": [
            {
                "name": "Kobold Den",
                "type": "dungeon",
                "adventure_temporary_key": "adv_a",
            },
            {
                "name": "Stone Tooth",
                "type": "dungeon",
                "adventure_temporary_key": "adv_b",
            },
        ],
        "npcs": [
            {
                "name": "Meepo",
                "role": "kobold",
                "primary_location_key": "Kobold Den",
                "adventure_temporary_key": "adv_a",
                "significance": "conversational",
            },
            {
                "name": "Durgeddin",
                "role": "dwarf",
                "primary_location_key": "Stone Tooth",
                "adventure_temporary_key": "adv_b",
                "significance": "conversational",
            },
        ],
    }
    warnings: list[str] = []
    adventures, locations, npcs = content_import._normalize_outline(
        outline,
        mode="create_new",
        existing_campaign_id=None,
        existing_location_ids=[],
        existing_npc_names=[],
        warnings=warnings,
    )
    assert len(adventures) == 2
    adventures, locations, npcs = content_import._stamp_discovery_temporary_keys(
        adventures,
        locations,
        npcs,
        mode="create_new",
        existing_campaign_id=None,
        warnings=warnings,
    )
    assert [a["temporary_key"] for a in adventures] == ["adventure_001", "adventure_002"]
    assert locations[0]["adventure_temporary_key"] == "adventure_001"
    assert locations[1]["adventure_temporary_key"] == "adventure_002"
    assert npcs[0]["adventure_temporary_key"] == "adventure_001"
    assert npcs[1]["adventure_temporary_key"] == "adventure_002"


def test_stamp_resolves_stamped_adventure_keys_on_entities():
    """LLM adventure_002 refs must resolve after stamping, not collapse to adventure_001."""
    adventures = [
        {"title": "Sunless Citadel", "name": "Sunless Citadel"},
        {"title": "Forge of Fury", "name": "Forge of Fury"},
    ]
    locations = [
        {"name": "Kobold Den", "type": "dungeon", "adventure_temporary_key": "adventure_001"},
        {"name": "Stone Tooth", "type": "dungeon", "adventure_temporary_key": "adventure_002"},
    ]
    npcs = [
        {
            "name": "Meepo",
            "primary_location_key": "Kobold Den",
            "adventure_temporary_key": "adventure_001",
            "significance": "conversational",
        },
        {
            "name": "Durgeddin",
            "primary_location_key": "",
            "adventure_temporary_key": "adventure_002",
            "significance": "conversational",
        },
    ]
    warnings: list[str] = []
    stamped_adv, stamped_locs, stamped_npcs = content_import._stamp_discovery_temporary_keys(
        adventures,
        locations,
        npcs,
        mode="create_new",
        existing_campaign_id=None,
        warnings=warnings,
    )
    assert [a["temporary_key"] for a in stamped_adv] == ["adventure_001", "adventure_002"]
    assert stamped_locs[1]["adventure_temporary_key"] == "adventure_002"
    # Blank primary defaults to location_001 but must keep adventure_002.
    assert stamped_npcs[1]["adventure_temporary_key"] == "adventure_002"
    assert stamped_npcs[1]["primary_location_key"] == "location_001"


def test_stamp_keeps_adventure_when_normalize_already_filled_primary():
    """Normalize fills blank primary with location_001; stamp must not reassign adventure."""
    outline = {
        "adventures": [
            {"title": "Sunless Citadel", "name": "Sunless Citadel"},
            {"title": "Forge of Fury", "name": "Forge of Fury"},
        ],
        "locations": [
            {
                "name": "Kobold Den",
                "type": "dungeon",
                "adventure_temporary_key": "adventure_001",
            },
            {
                "name": "Stone Tooth",
                "type": "dungeon",
                "adventure_temporary_key": "adventure_002",
            },
        ],
        "npcs": [
            {
                "name": "Durgeddin",
                "role": "dwarf",
                # No primary — normalize will default to first location.
                "adventure_temporary_key": "adventure_002",
                "significance": "conversational",
            },
        ],
    }
    warnings: list[str] = []
    adventures, locations, npcs = content_import._normalize_outline(
        outline,
        mode="create_new",
        existing_campaign_id=None,
        existing_location_ids=[],
        existing_npc_names=[],
        warnings=warnings,
    )
    assert npcs[0]["primary_location_key"]  # filled by normalize
    adventures, locations, npcs = content_import._stamp_discovery_temporary_keys(
        adventures,
        locations,
        npcs,
        mode="create_new",
        existing_campaign_id=None,
        warnings=warnings,
    )
    assert npcs[0]["adventure_temporary_key"] == "adventure_002"
    assert npcs[0]["primary_location_key"] == "location_001"


def test_build_campaign_prompt_echoes_seed_temporary_key():
    prompt = content_import._build_campaign_prompt(
        "Some source text about Forge of Fury.",
        {"temporary_key": "adventure_002", "title": "Forge of Fury", "name": "Forge of Fury"},
    )
    assert '"temporary_key": "adventure_002"' in prompt
    assert "adventure_001" not in prompt


def test_generate_campaign_info_preserves_seed_temporary_key(monkeypatch):
    def fake_llm(*_args, **_kwargs):
        return {
            "temporary_key": "adventure_001",
            "title": "Forge of Fury",
            "name": "Forge of Fury",
            "description": "A dwarven stronghold.",
            "premise": "Reclaim the forge.",
        }

    monkeypatch.setattr(content_import, "_call_llm_json", fake_llm)
    result = content_import.generate_campaign_info(
        "Forge of Fury " * 20,
        campaign={"temporary_key": "adventure_002", "title": "Forge of Fury", "name": "Forge of Fury"},
    )
    assert result["campaign"]["temporary_key"] == "adventure_002"


def test_outline_legacy_campaign_wrap_returns_adventures(monkeypatch):
    """Discover hop wraps singular campaign into adventures[] for editor consumers."""

    def fake_llm(*_args, **_kwargs):
        return {
            "campaign": {"title": "One Shot", "name": "One Shot", "genre": "fantasy"},
            "locations": [{"name": "Town", "type": "town"}],
            "npcs": [
                {
                    "name": "Mayor",
                    "role": "leader",
                    "primary_location_key": "Town",
                    "significance": "conversational",
                }
            ],
        }

    monkeypatch.setattr(content_import, "_call_llm_json", fake_llm)
    monkeypatch.setattr(content_import, "_document_has_recoverable_stat_blocks", lambda *_a, **_k: False)
    text = "=== PAGE 1 ===\nOne Shot Adventure\n" + ("The mayor greets the party in town. " * 20)
    result = content_import.outline_campaign_document(text, "create_new")
    assert len(result["adventures"]) == 1
    assert result["adventures"][0]["temporary_key"] == "adventure_001"
    assert result["campaign"]["temporary_key"] == "adventure_001"
    assert result["locations"][0]["adventure_temporary_key"] == "adventure_001"

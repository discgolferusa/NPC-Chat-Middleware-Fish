"""Unit tests for VTT source_kind import prompt hints."""

from __future__ import annotations

import content_import


def test_normalize_source_kind_accepts_vtt_aliases():
    assert content_import._normalize_source_kind("vtt_world") == "vtt_world"
    assert content_import._normalize_source_kind("Foundry") == "vtt_world"
    assert content_import._normalize_source_kind("document") == "document"
    assert content_import._normalize_source_kind(None) == "document"


def test_outline_prompt_includes_vtt_guidance():
    prompt = content_import._build_outline_prompt(
        "=== SECTION JOURNAL: Docks / Overview ===\nA foggy pier.\n",
        "create_new",
        None,
        [],
        [],
        document_scope="single",
        source_kind="vtt_world",
    )
    assert "SOURCE KIND = VTT world export" in prompt
    assert "JOURNAL sections" in prompt


def test_campaign_prompt_includes_vtt_guidance():
    prompt = content_import._build_campaign_prompt(
        "=== SECTION ACTOR: Varra (on scene) ===\nCaptain.",
        {"temporary_key": "adventure_001", "title": "Greyhaven", "name": "Greyhaven"},
        source_kind="vtt_world",
    )
    assert "SOURCE KIND = VTT world export" in prompt
    assert 'temporary_key MUST remain "adventure_001"' in prompt

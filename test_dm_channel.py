"""Unit tests for DM channel detectors (no DB / LLM)."""

from __future__ import annotations

from dm_channel import (
    build_dm_injection_messages,
    detect_npc_deferral,
    detect_player_claim,
    detect_ruling,
    format_dm_context_for_prompt,
    infer_claim_summary,
    strip_dm_check_tag,
)


def test_strip_dm_check_tag_extracts_claim():
    spoken, claim = strip_dm_check_tag("I open the door. [DM CHECK: door is unlocked]")
    assert "open the door" in spoken.lower()
    assert claim is not None
    assert "unlocked" in claim.lower()


def test_strip_dm_check_tag_without_tag():
    spoken, claim = strip_dm_check_tag("Just talking.")
    assert spoken == "Just talking."
    assert claim is None


def test_detect_ruling_confirmed():
    result = detect_ruling("Yes, that is correct.")
    assert result["verdict"] == "confirmed"
    assert result["confidence"] > 0


def test_detect_ruling_denied():
    result = detect_ruling("No, that is wrong.")
    assert result["verdict"] == "denied"
    assert result["confidence"] > 0


def test_detect_ruling_unclear_empty():
    result = detect_ruling("")
    assert result["verdict"] == "unclear"
    assert result["confidence"] == 0.0


def test_detect_player_claim_strong():
    result = detect_player_claim("We have a letter from the mayor allowing us to enter.")
    assert result is not None
    assert result["detection_tier"] in ("strong", "loose")
    assert result["claim_summary"]


def test_detect_npc_deferral():
    result = detect_npc_deferral("I need the DM to confirm your story first.")
    assert result is not None
    assert result["matched_phrase"]


def test_infer_claim_summary_fallback():
    summary = infer_claim_summary("Hello there")
    assert isinstance(summary, str)
    assert len(summary) > 0


def test_format_dm_context_and_injection():
    ctx = {
        "pending_checks": [{"claim_summary": "door unlocked", "status": "pending"}],
        "rulings": [
            {
                "verdict": "confirmed",
                "claim_summary": "door unlocked",
                "ruling_text": "Yes",
            }
        ],
        "messages": [],
    }
    text = format_dm_context_for_prompt(ctx)
    assert "door unlocked" in text.lower()
    messages = build_dm_injection_messages(ctx)
    assert isinstance(messages, list)
    assert messages

"""Unit tests for DM channel detectors (no DB / LLM)."""

from __future__ import annotations

from dm_channel import (
    build_dm_injection_messages,
    detect_npc_deferral,
    detect_player_claim,
    detect_ruling,
    extract_dm_check_tags,
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


def test_strip_dm_check_tag_mid_reply_still_extracts():
    spoken, claim = strip_dm_check_tag(
        "*Looks confused.* [DM CHECK: does this speaker know Language Slaadi] ..."
    )
    assert claim is not None
    assert "slaadi" in claim.lower()
    assert "Looks confused" in spoken
    assert "[DM CHECK:" not in spoken


def test_extract_dm_check_tags_strips_all_and_keeps_order():
    spoken, claims = extract_dm_check_tags(
        "Wait.\n"
        "[DM CHECK: door is unlocked]\n"
        "[DM CHECK: Does the speaker know the necessary language to talk to the NPC]\n"
    )
    assert spoken == "Wait."
    assert len(claims) == 2
    assert "door is unlocked" in claims[0]
    assert "necessary language" in claims[1].lower()
    assert "[DM CHECK:" not in spoken


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


def test_process_npc_response_keeps_distinct_dm_checks(tmp_path, monkeypatch):
    """Same player_text must not collapse unrelated [DM CHECK] claims into one pending."""
    import dm_channel

    mem_db = tmp_path / "dm_multi_check.db"
    monkeypatch.setattr(dm_channel, "MEMORY_DB", mem_db)
    dm_channel.init_dm_tables()

    result = dm_channel.process_npc_response(
        "npc_x",
        "player_a",
        "Hello friend",
        (
            "Wait.\n"
            "[DM CHECK: door is unlocked]\n"
            "[DM CHECK: Does the speaker know the necessary language to talk to the NPC]\n"
        ),
        campaign_id="camp_1",
    )

    assert result["spoken_text"] == "Wait."
    assert len(result["dm_check_claims"]) == 2
    assert len(result["pending_check_ids"]) == 2
    assert len(set(result["pending_check_ids"])) == 2

    pending = dm_channel.load_pending_checks(
        "npc_x", "player_a", status="pending", campaign_id="camp_1"
    )
    summaries = {item["claim_summary"].lower() for item in pending}
    assert any("door is unlocked" in s for s in summaries)
    assert any("necessary language" in s for s in summaries)
    assert len(pending) >= 2

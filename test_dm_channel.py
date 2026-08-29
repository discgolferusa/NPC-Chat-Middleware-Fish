"""Unit tests for DM channel detectors (no DB / LLM)."""

from __future__ import annotations

import pytest

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


@pytest.fixture()
def dm_sqlite(tmp_path, monkeypatch):
    """Isolate DM channel tests on a temp SQLite file (ignore host MySQL env)."""
    import dm_channel
    import memory_db

    for key in (
        "MEMORY_MYSQL_HOST",
        "MEMORY_MYSQL_PORT",
        "MEMORY_MYSQL_DATABASE",
        "MEMORY_MYSQL_USER",
        "MEMORY_MYSQL_PASSWORD",
        "MEMORY_MYSQL_CHARSET",
    ):
        monkeypatch.delenv(key, raising=False)

    mem_db = (tmp_path / "dm_test.db").resolve()
    monkeypatch.setenv("MEMORY_DB", str(mem_db))
    cfg = memory_db.MemoryDbConfig(backend="sqlite", sqlite_path=mem_db)
    monkeypatch.setattr(memory_db, "CONFIG", cfg)
    monkeypatch.setattr(memory_db, "MEMORY_DB", mem_db)
    monkeypatch.setattr(dm_channel, "MEMORY_DB", mem_db)
    dm_channel.init_dm_tables()
    return dm_channel, mem_db


def test_get_dm_context_reuses_one_database_connection(dm_sqlite, monkeypatch):
    dm_channel, _mem_db = dm_sqlite
    original_get_db = dm_channel.get_db
    calls = {"count": 0}

    def counted_get_db():
        calls["count"] += 1
        return original_get_db()

    monkeypatch.setattr(dm_channel, "get_db", counted_get_db)

    context = dm_channel.get_dm_context(
        "captain_varra",
        "player_alden",
        campaign_id="greyhaven_intro",
    )

    assert calls["count"] == 1
    assert context == {
        "campaign_id": "greyhaven_intro",
        "messages": [],
        "pending_checks": [],
        "rulings": [],
    }


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


def test_detect_npc_deferral_ignores_dm_check_tag_alone():
    assert detect_npc_deferral(
        "[DM CHECK: Does the speaker know the necessary language to talk to the NPC]"
    ) is None
    assert detect_npc_deferral(
        "Krah?\n[DM CHECK: Does the speaker know the necessary language to talk to the NPC]"
    ) is None


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


def test_process_npc_response_keeps_distinct_dm_checks(dm_sqlite):
    """Same player_text must not collapse unrelated [DM CHECK] claims into one pending."""
    dm_channel, _mem_db = dm_sqlite

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

    assert result["spoken_text"] == dm_channel.DM_PENDING_WAIT_NARRATION
    assert result["held_spoken_text"] == "Wait."
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


def test_save_pending_check_skips_when_claim_already_ruled(dm_sqlite):
    dm_channel, _mem_db = dm_sqlite

    dm_channel.create_manual_ruling(
        npc_id="captain_varra",
        player_id="player_alden",
        claim_summary="player claims official authority or permission",
        verdict="confirmed",
        notes="Approved.",
        campaign_id="greyhaven_intro",
    )

    created = dm_channel.save_pending_check(
        npc_id="captain_varra",
        player_id="player_alden",
        claim_summary="player claims official authority or permission that needs DM confirmation",
        player_text="Let us pass",
        npc_text="Hold.",
        source="npc",
        campaign_id="greyhaven_intro",
    )
    assert created is None
    pending = dm_channel.load_pending_checks_for_dm(
        campaign_id="greyhaven_intro",
        status="pending",
        limit=20,
    )
    assert pending == []


def test_save_pending_check_allows_distinct_claim_after_unrelated_ruling(dm_sqlite):
    """Shared generic tokens must not suppress a different claim under strict coverage."""
    dm_channel, _mem_db = dm_sqlite

    dm_channel.create_manual_ruling(
        npc_id="captain_varra",
        player_id="player_alden",
        claim_summary="player has permission to enter the aqueduct",
        verdict="confirmed",
        notes="Approved.",
        campaign_id="greyhaven_intro",
    )

    # Loose _claims_match would overlap on player/has/permission/to; strict coverage must not.
    assert dm_channel._claims_match(
        "player has permission to enter the aqueduct",
        "player has permission to speak Language Slaadi",
    )

    created = dm_channel.save_pending_check(
        npc_id="captain_varra",
        player_id="player_alden",
        claim_summary="player has permission to speak Language Slaadi",
        player_text="Can you understand me?",
        npc_text="Krah?",
        source="npc",
        campaign_id="greyhaven_intro",
    )
    assert created is not None
    pending = dm_channel.load_pending_checks_for_dm(
        campaign_id="greyhaven_intro",
        status="pending",
        limit=20,
    )
    assert any(row["id"] == created for row in pending)
    assert any("slaadi" in row["claim_summary"].lower() for row in pending)


def test_process_npc_response_can_strip_tags_without_creating_pending(dm_sqlite):
    dm_channel, _mem_db = dm_sqlite

    result = dm_channel.process_npc_response(
        "captain_varra",
        "player_alden",
        "",
        "Very well.\n[DM CHECK: player claims official authority or permission]",
        campaign_id="greyhaven_intro",
        create_pending_checks=False,
    )
    assert "Very well" in result["spoken_text"]
    assert result.get("held_spoken_text") == ""
    assert "[DM CHECK:" not in result["spoken_text"]
    assert result["pending_check_ids"] == []
    assert dm_channel.load_pending_checks_for_dm(
        campaign_id="greyhaven_intro",
        status="pending",
        limit=20,
    ) == []


def test_process_npc_response_holds_dialogue_while_existing_pending(dm_sqlite):
    dm_channel, _mem_db = dm_sqlite
    dm_channel.save_pending_check(
        npc_id="captain_varra",
        player_id="player_alden",
        claim_summary="player has a writ of passage",
        player_text="Let us pass",
        source="player",
        campaign_id="greyhaven_intro",
    )

    result = dm_channel.process_npc_response(
        "captain_varra",
        "player_alden",
        "And also, nice weather.",
        "The gate is open. Come through.",
        campaign_id="greyhaven_intro",
    )
    assert result["spoken_text"] == dm_channel.DM_PENDING_WAIT_NARRATION
    assert result["held_spoken_text"] == "The gate is open. Come through."
    assert result["pending_check_ids"] == []


def test_assistant_history_text_skips_wait_narration():
    import dm_channel

    assert dm_channel.is_dm_wait_narration(dm_channel.DM_PENDING_WAIT_NARRATION)
    assert dm_channel.is_dm_wait_narration("*Waiting on a DM decision.*")
    assert dm_channel.is_dm_wait_narration("Waiting on a decision from the game master.")
    assert dm_channel.assistant_history_text({
        "spoken_text": dm_channel.DM_PENDING_WAIT_NARRATION,
        "held_spoken_text": "The gate is open. Come through.",
    }) == "The gate is open. Come through."
    assert dm_channel.assistant_history_text({
        "spoken_text": dm_channel.DM_PENDING_WAIT_NARRATION,
        "held_spoken_text": dm_channel.DM_PENDING_WAIT_NARRATION,
    }) == ""
    assert dm_channel.assistant_history_text({
        "spoken_text": "Very well.",
        "held_spoken_text": "",
    }) == "Very well."


def test_dm_console_loaders_are_campaign_wide_across_players(dm_sqlite):
    """DM console must see every player's pending checks / rulings in a campaign."""
    dm_channel, _mem_db = dm_sqlite

    first = dm_channel.save_pending_check(
        npc_id="captain_varra",
        player_id="player_alden",
        claim_summary="player claims official authority",
        player_text="Let us pass",
        npc_text="I need confirmation.",
        source="npc",
        campaign_id="greyhaven_intro",
    )
    second = dm_channel.save_pending_check(
        npc_id="captain_varra",
        player_id="player_mira",
        claim_summary="party has a sealed letter",
        player_text="We carry a letter",
        npc_text="Show me.",
        source="npc",
        campaign_id="greyhaven_intro",
    )
    assert first and second

    pending = dm_channel.load_pending_checks_for_dm(
        campaign_id="greyhaven_intro",
        status="pending",
        limit=20,
    )
    player_ids = {row["player_id"] for row in pending}
    assert "player_alden" in player_ids
    assert "player_mira" in player_ids

    # Player-scoped loader must not pretend the DM default party owns these rows.
    scoped = dm_channel.load_pending_checks(
        "captain_varra",
        "default_party",
        status="pending",
        campaign_id="greyhaven_intro",
    )
    assert scoped == []

    ruling = dm_channel.create_manual_ruling(
        npc_id="captain_varra",
        player_id="player_alden",
        claim_summary="player claims official authority",
        verdict="confirmed",
        notes="Approved.",
        pending_check_id=first,
        campaign_id="greyhaven_intro",
    )
    assert ruling["ruling_id"]

    rulings = dm_channel.load_rulings_for_dm(campaign_id="greyhaven_intro", limit=20)
    assert any(row["id"] == ruling["ruling_id"] for row in rulings)
    assert any(row["player_id"] == "player_alden" for row in rulings)


def test_dismiss_pending_check_marks_ignored_without_ruling(dm_sqlite):
    dm_channel, _mem_db = dm_sqlite

    pending_id = dm_channel.save_pending_check(
        npc_id="captain_varra",
        player_id="player_alden",
        claim_summary="noise claim",
        player_text="whatever",
        npc_text="checking",
        source="npc",
        campaign_id="greyhaven_intro",
    )
    assert pending_id

    updated = dm_channel.dismiss_pending_check(pending_id)
    assert updated is not None
    assert updated["status"] == "ignored"
    assert updated["resolved_at"]

    open_rows = dm_channel.load_pending_checks_for_dm(
        campaign_id="greyhaven_intro",
        status="pending",
        limit=20,
    )
    assert all(row["id"] != pending_id for row in open_rows)

    ignored = dm_channel.load_pending_check_by_id(pending_id)
    assert ignored is not None
    assert ignored["status"] == "ignored"

    import pytest

    with pytest.raises(ValueError, match="already ignored"):
        dm_channel.dismiss_pending_check(pending_id)


def test_dismiss_pending_check_rejects_already_ruled(dm_sqlite):
    import pytest

    dm_channel, _mem_db = dm_sqlite

    pending_id = dm_channel.save_pending_check(
        npc_id="captain_varra",
        player_id="player_alden",
        claim_summary="already decided",
        player_text="claim",
        npc_text="defer",
        source="npc",
        campaign_id="greyhaven_intro",
    )
    assert pending_id

    dm_channel.create_manual_ruling(
        npc_id="captain_varra",
        player_id="player_alden",
        claim_summary="already decided",
        verdict="confirmed",
        notes="Approved.",
        pending_check_id=pending_id,
        campaign_id="greyhaven_intro",
    )
    ruled = dm_channel.load_pending_check_by_id(pending_id)
    assert ruled is not None
    assert ruled["status"] == "confirmed"

    with pytest.raises(ValueError, match="already confirmed"):
        dm_channel.dismiss_pending_check(pending_id)

    still = dm_channel.load_pending_check_by_id(pending_id)
    assert still is not None
    assert still["status"] == "confirmed"


def test_clear_dm_data_all_players_removes_other_speakers_rulings(dm_sqlite):
    """DM clear must wipe language Denies stored under the speaking player_id."""
    dm_channel, _mem_db = dm_sqlite

    campaign = "greyhaven_intro"
    npc = "slaad_guard"
    pending = dm_channel.save_pending_check(
        npc_id=npc,
        player_id="player_alden",
        claim_summary="Does the speaker know the necessary language to talk to the NPC",
        player_text="Hello",
        npc_text="Waiting.",
        source="npc",
        campaign_id=campaign,
    )
    ruling_id = dm_channel.save_ruling(
        npc_id=npc,
        player_id="player_alden",
        claim_summary="Does the speaker know the necessary language to talk to the NPC",
        verdict="denied",
        ruling_text="Failure",
        source="dm",
        pending_check_id=pending,
        campaign_id=campaign,
    )
    assert ruling_id
    assert dm_channel.load_rulings(npc, "player_alden", campaign_id=campaign)

    # Wrong speaker id (what the old DM clear used) must not leave the ruling.
    noop = dm_channel.clear_dm_data(
        npc,
        "dm_console",
        campaign_id=campaign,
        all_players=False,
    )
    assert noop["deleted_rulings"] == 0
    assert dm_channel.load_rulings(npc, "player_alden", campaign_id=campaign)

    cleared = dm_channel.clear_dm_data(
        npc,
        "dm_console",
        campaign_id=campaign,
        all_players=True,
    )
    assert cleared["deleted_rulings"] >= 1
    assert cleared["all_players"] is True
    assert dm_channel.load_rulings(npc, "player_alden", campaign_id=campaign) == []
    assert dm_channel.load_pending_checks(
        npc, "player_alden", status=None, campaign_id=campaign
    ) == []

"""Unit tests for in-memory turn progress tracking."""

from __future__ import annotations

import turn_progress as tp


def setup_function(_):
    with tp._PROGRESS_LOCK:
        tp._PROGRESS.clear()
        tp._ACTIVE_TURNS.clear()


def test_normalize_and_new_request_id():
    assert tp.normalize_client_request_id("ABCDEF0123456789ABCDEF0123456789") == "abcdef0123456789abcdef0123456789"
    assert tp.normalize_client_request_id("not-valid") is None
    generated = tp.new_request_id(None)
    assert len(generated) == 32


def test_register_set_complete_flow():
    rid = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    registered = tp.register_turn_progress(rid, route="/npc-turn", npc_id="bob", mode="text")
    assert registered["request_id"] == rid
    assert registered["status"] == "in_progress"
    assert registered["step"] == "registered"

    updated = tp.set_turn_progress(rid, "waiting_llm", detail="thinking")
    assert updated is not None
    assert updated["step"] == "waiting_llm"
    assert updated["detail"] == "thinking"

    done = tp.complete_turn_progress(rid)
    assert done is not None
    assert done["status"] == "complete"


def test_fail_turn_progress():
    rid = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    tp.register_turn_progress(rid, route="/npc-voice-turn", mode="voice")
    failed = tp.fail_turn_progress(rid, "boom")
    assert failed is not None
    assert failed["status"] == "error"
    assert "boom" in (failed.get("error") or "")


def test_set_unknown_request_returns_none():
    assert tp.set_turn_progress("cccccccccccccccccccccccccccccccc", "received") is None


def test_set_after_complete_does_not_reopen():
    rid = "dddddddddddddddddddddddddddddddd"
    tp.register_turn_progress(rid, route="/npc-turn", mode="text")
    tp.complete_turn_progress(rid)

    reopened = tp.set_turn_progress(rid, "synthesizing_speech")
    assert reopened is not None
    assert reopened["status"] == "complete"
    assert reopened["step"] == "complete"


def test_complete_after_fail_stays_error():
    rid = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
    tp.register_turn_progress(rid, route="/npc-voice-turn", mode="voice")
    failed = tp.fail_turn_progress(rid, "handler_failed")
    assert failed is not None
    assert failed["status"] == "error"

    flipped = tp.complete_turn_progress(rid)
    assert flipped is not None
    assert flipped["status"] == "error"
    assert "handler_failed" in (flipped.get("error") or "")


def test_text_steps_include_synthesizing_speech():
    keys = [step["key"] for step in tp.TEXT_TURN_STEPS]
    assert "synthesizing_speech" in keys
    assert keys.index("synthesizing_speech") < keys.index("finishing")
    assert keys.index("finishing") < keys.index("complete")


def test_active_turn_blocks_second_register():
    rid_a = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    rid_b = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    tp.register_turn_progress(
        rid_a,
        route="/npc-turn",
        npc_id="bob",
        player_id="alice",
        mode="text",
        campaign_id="camp-1",
    )
    active = tp.get_active_turn_progress("camp-1", "bob", "alice")
    assert active is not None
    assert active["request_id"] == rid_a

    try:
        tp.register_turn_progress(
            rid_b,
            route="/npc-turn",
            npc_id="bob",
            player_id="alice",
            mode="text",
            campaign_id="camp-1",
        )
        assert False, "expected TurnInProgressError"
    except tp.TurnInProgressError as ex:
        assert ex.progress["request_id"] == rid_a

    tp.complete_turn_progress(rid_a)
    assert tp.get_active_turn_progress("camp-1", "bob", "alice") is None

    registered = tp.register_turn_progress(
        rid_b,
        route="/npc-turn",
        npc_id="bob",
        player_id="alice",
        mode="text",
        campaign_id="camp-1",
    )
    assert registered["request_id"] == rid_b


def test_fail_turn_releases_active_slot():
    rid = "cccccccccccccccccccccccccccccccc"
    tp.register_turn_progress(
        rid,
        route="/dm-ruling-auto-response",
        npc_id="bob",
        player_id="alice",
        mode="text",
        campaign_id="camp-2",
    )
    tp.fail_turn_progress(rid, "boom")
    assert tp.get_active_turn_progress("camp-2", "bob", "alice") is None


def test_force_register_displaces_active_turn():
    rid_a = "dddddddddddddddddddddddddddddddd"
    rid_b = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
    tp.register_turn_progress(
        rid_a,
        route="/npc-turn",
        npc_id="bob",
        player_id="alice",
        mode="text",
        campaign_id="camp-3",
    )
    displaced = tp.register_turn_progress(
        rid_b,
        route="/dm-ruling-auto-response",
        npc_id="bob",
        player_id="alice",
        mode="text",
        campaign_id="camp-3",
        force=True,
    )
    assert displaced["request_id"] == rid_b
    active = tp.get_active_turn_progress("camp-3", "bob", "alice")
    assert active is not None
    assert active["request_id"] == rid_b

"""TTS attribution: spoken dialogue must not ride the narrator voice."""

from __future__ import annotations

import pytest

pytest.importorskip("flask")


def test_split_keeps_stage_direction_on_narrator_and_speech_on_npc():
    import server

    segments = server.split_into_ordered_tts_segments(
        "*Xill cocks its head.* Krah-vesh?"
    )
    assert [s["type"] for s in segments] == ["narrator", "npc"]
    assert "cocks" in segments[0]["text"].lower() or "head" in segments[0]["text"].lower()
    assert "krah" in segments[1]["text"].lower()


def test_split_puts_italicized_language_line_on_npc_not_narrator():
    import server

    segments = server.split_into_ordered_tts_segments("*Krah-vesh?*")
    assert len(segments) == 1
    assert segments[0]["type"] == "npc"
    assert "krah" in segments[0]["text"].lower()


def test_split_puts_italicized_denied_claim_dialogue_on_npc():
    """Not language-only — any check where the model wraps speech in asterisks."""
    import server

    segments = server.split_into_ordered_tts_segments(
        "*I will not open the gate without proof.*"
    )
    assert len(segments) == 1
    assert segments[0]["type"] == "npc"
    assert "gate" in segments[0]["text"].lower()


def test_split_keeps_wait_narration_on_narrator():
    import server

    segments = server.split_into_ordered_tts_segments(
        "*Waiting on a decision from game master.*"
    )
    assert len(segments) == 1
    assert segments[0]["type"] == "narrator"

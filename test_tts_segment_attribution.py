"""TTS attribution: asterisks are narrator, plain text is the NPC voice."""

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


def test_split_keeps_unitalicized_language_line_on_npc():
    """Language-barrier speech is emitted as plain text, so it stays NPC voice."""
    import server

    segments = server.split_into_ordered_tts_segments("*Xill cocks its head.* Skraa-vek?")
    assert [s["type"] for s in segments] == ["narrator", "npc"]
    assert "skraa" in segments[1]["text"].lower()


def test_split_keeps_plain_dialogue_on_npc():
    import server

    segments = server.split_into_ordered_tts_segments(
        "I will not open the gate without proof."
    )
    assert len(segments) == 1
    assert segments[0]["type"] == "npc"
    assert "gate" in segments[0]["text"].lower()


def test_split_keeps_post_ruling_outcome_narration_on_narrator():
    """After a confirmed DM check the NPC narrates the outcome, then speaks."""
    import server

    segments = server.split_into_ordered_tts_segments(
        "*The guard steps aside and unlatches the gate.* Go on through, then."
    )
    assert [s["type"] for s in segments] == ["narrator", "npc"]
    assert "gate" in segments[0]["text"].lower()
    assert "through" in segments[1]["text"].lower()


def test_split_narrates_long_italics_regardless_of_length():
    """Length is governed by the prompt rules, never by TTS attribution."""
    import server

    segments = server.split_into_ordered_tts_segments(
        "*The guard studies the seal for a long moment, then steps aside and "
        "hauls the gate open, waving the whole party through without a word.*"
    )
    assert len(segments) == 1
    assert segments[0]["type"] == "narrator"


def test_split_keeps_trailing_outcome_narration_on_narrator():
    import server

    segments = server.split_into_ordered_tts_segments(
        "Very well, the seal is genuine. *She slides the brass key across the counter.*"
    )
    assert [s["type"] for s in segments] == ["npc", "narrator"]
    assert "key" in segments[1]["text"].lower()


def test_split_keeps_wait_narration_on_narrator():
    import server

    segments = server.split_into_ordered_tts_segments(
        "*Waiting on a decision from game master.*"
    )
    assert len(segments) == 1
    assert segments[0]["type"] == "narrator"


def test_split_keeps_bare_wait_narration_on_narrator():
    """Asterisks can be lost before TTS — wait copy must never use the NPC voice."""
    import server

    segments = server.split_into_ordered_tts_segments(
        "Waiting on a decision from game master."
    )
    assert len(segments) == 1
    assert segments[0]["type"] == "narrator"
    assert "waiting" in segments[0]["text"].lower()


def test_split_keeps_wait_wording_variants_on_narrator():
    import server

    segments = server.split_into_ordered_tts_segments(
        "*Waiting on a decision from the game master.*"
    )
    assert len(segments) == 1
    assert segments[0]["type"] == "narrator"

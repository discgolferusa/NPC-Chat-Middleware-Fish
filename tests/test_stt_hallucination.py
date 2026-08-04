"""Unit tests for short-clip Whisper hallucination detection."""

from server import _is_stt_hallucination


def test_repeated_tokens_are_hallucinations():
    assert _is_stt_hallucination("No. No. No. No.")
    assert _is_stt_hallucination("Thank you. Thank you. Thank you.")


def test_normal_sentence_is_not_hallucination():
    assert not _is_stt_hallucination("Can you tell me about the harbor?")


def test_short_clip_you_is_hallucination():
    assert _is_stt_hallucination("You", duration_seconds=1.04, peak=1800)
    assert _is_stt_hallucination("Thank you", duration_seconds=1.2, peak=900)


def test_short_allowlisted_words_still_ok():
    assert not _is_stt_hallucination("Yes", duration_seconds=0.9, peak=8000)
    assert not _is_stt_hallucination("Wait", duration_seconds=1.0, peak=6000)


def test_longer_you_sentence_not_rejected_by_short_clip_rule():
    assert not _is_stt_hallucination(
        "You there, what do you want from me?",
        duration_seconds=3.5,
        peak=10000,
    )

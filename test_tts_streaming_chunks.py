"""
Sentence-level TTS clips, em-dash beats, and the first-audio timing mark.

A whole paragraph of dialogue used to be one Piper call, so the table waited for
the last word to be synthesized before hearing the first.
"""

from __future__ import annotations

import pytest

from tts_processing import (
    chunk_sentences_for_streaming,
    clean_for_tts,
    dashes_to_sentence_breaks,
    split_into_sentences,
    split_text_for_streaming,
)


def test_em_dash_becomes_a_sentence_break():
    # espeak-ng runs straight through "—", so the beat the writer intended is
    # lost. A sentence boundary hands the pause to Piper's sentence_silence.
    assert dashes_to_sentence_breaks(
        "I'm partial to it myself — been here my whole life"
    ) == "I'm partial to it myself. been here my whole life"


def test_em_dash_does_not_double_up_punctuation():
    assert dashes_to_sentence_breaks("Well, — I suppose") == "Well. I suppose"
    assert dashes_to_sentence_breaks("Stop! — he shouted") == "Stop! he shouted"


def test_dash_variants_all_count():
    for dash in ("—", "–", "--", "---"):
        assert dashes_to_sentence_breaks(f"one {dash} two") == "one. two"


def test_a_number_range_is_a_span_not_a_pause():
    assert dashes_to_sentence_breaks("5–10 coppers") == "5, 10 coppers"
    assert dashes_to_sentence_breaks("It costs 5 – 10 coppers.") == "It costs 5, 10 coppers."


def test_spelled_numbers_are_treated_as_words_not_a_range():
    # "one — two" is far more often someone counting out loud than a span, so
    # the guard stays on digits and the dash keeps its beat.
    assert dashes_to_sentence_breaks("one — two") == "one. two"


def test_cleaning_keeps_a_range_together_once_the_numbers_are_spelled():
    # Spelling ran first, so by the time the dash was examined the digits were
    # gone and the span became "five. ten".
    assert clean_for_tts("It costs 5–10 coppers.") == "It costs five, ten coppers."


def test_hyphenated_words_are_left_alone():
    assert dashes_to_sentence_breaks("a well-known smith") == "a well-known smith"


def test_clean_for_tts_applies_the_dash_beat():
    assert clean_for_tts("Aye — the docks are dangerous.") == "Aye. the docks are dangerous."


def test_sentences_split_on_ends_and_keep_their_punctuation():
    assert split_into_sentences("Aye. Mind the docks! Who sent you?") == [
        "Aye.",
        "Mind the docks!",
        "Who sent you?",
    ]


def test_a_trailing_off_ellipsis_is_one_thought():
    # "Well... I suppose" is a single line delivered with a pause, and cutting
    # it would put a clip boundary in the middle of the hesitation.
    assert split_into_sentences("Well... I suppose so.") == ["Well... I suppose so."]


def test_a_decimal_point_is_not_a_sentence_end():
    assert split_into_sentences("It costs 3.5 silver.") == ["It costs 3.5 silver."]


def test_a_closing_quote_stays_with_its_sentence():
    assert split_into_sentences('He said "Go." Then he left.') == [
        'He said "Go."',
        "Then he left.",
    ]


def test_the_first_clip_is_short_and_later_ones_are_packed():
    sentences = [f"Sentence number {n} runs on for a little while." for n in range(6)]

    chunks = chunk_sentences_for_streaming(sentences, first_max_chars=60, max_chars=200)

    # Time-to-first-audio is the point, so the opening clip stays small; past
    # that, more boundaries only cost per-clip synth and FFmpeg overhead.
    assert len(chunks[0]) <= 60
    assert len(chunks) < len(sentences)
    assert " ".join(chunks) == " ".join(sentences)


def test_a_sentence_longer_than_the_budget_is_never_cut():
    long_sentence = "word " * 200 + "end."

    chunks = chunk_sentences_for_streaming([long_sentence.strip()], max_chars=100)

    assert chunks == [long_sentence.strip()]


def test_streaming_split_keeps_every_word_in_order():
    reply = (
        "Watchvale, you say? Well, I'm partial to it myself. Been here my whole life, "
        "and the Salty Wench's been feedin' sailors longer than I can remember. "
        "Mind the streets after dark, though."
    )

    chunks = split_text_for_streaming(reply)

    assert len(chunks) > 1
    assert " ".join(chunks) == reply


def test_text_with_no_sentence_end_still_produces_a_clip():
    assert split_text_for_streaming("no punctuation here") == ["no punctuation here"]
    assert split_text_for_streaming("   ") == []


def _narrator_settings():
    return {
        "resolved_voice_model": "narrator.onnx",
        "length_scale": 1.0,
        "noise_scale": 0.5,
        "noise_w": 0.5,
        "sentence_silence": 0.1,
        "speaker_id": 0,
        "pitch_semitones": 0.0,
        "fx_preset": "none",
        "denoise_nr": 0,
        "denoise_nf": -25,
    }


def test_a_paragraph_of_dialogue_becomes_several_ordered_clips():
    pytest.importorskip("faster_whisper")
    import server

    reply = (
        "*She leans against the counter.* Watchvale, you say? Well, I'm partial to it "
        "myself. Been here my whole life, and the Wench has fed sailors longer than I "
        "can remember. Mind the streets after dark, though."
    )

    segments = server.split_into_ordered_tts_segments(reply)

    # The narration still leads, and the dialogue behind it is no longer one
    # Piper call the player has to wait out in full.
    assert segments[0]["type"] == "narrator"
    dialogue = [s for s in segments if s["type"] == "npc"]
    assert len(dialogue) > 1
    assert dialogue[0]["text"].startswith("Watchvale")
    assert "after dark" in dialogue[-1]["text"]


def test_splitting_can_be_turned_off_for_a_single_clip():
    pytest.importorskip("faster_whisper")
    import server

    segments = server.split_into_ordered_tts_segments(
        "One. Two. Three.", stream_sentences=False
    )

    assert [s["text"] for s in segments] == ["One. Two. Three."]


def test_every_clip_is_kept_in_order_for_the_whole_turn_wav(monkeypatch, tmp_path):
    pytest.importorskip("faster_whisper")
    import server

    monkeypatch.setattr(server, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(server, "get_narrator_tts_settings", _narrator_settings)
    monkeypatch.setattr(
        server,
        "synthesize_speech",
        lambda text, voice, raw_path, **kwargs: raw_path.write_bytes(b"RIFF" + b"\x00" * 40),
    )
    monkeypatch.setattr(
        server,
        "convert_wav_to_standard_format",
        lambda src, dest, **kwargs: dest.write_bytes(b"RIFF" + b"\x00" * 40),
    )

    result = server.synthesize_segmented_speech_clips(
        npc_text=" ".join(
            f"This is sentence number {n} and it runs on for a good while yet."
            for n in range(8)
        ),
        npc_voice_model="npc.onnx",
        request_id="e" * 32,
    )

    # The legacy narration/speech pair keeps only the first narration and the
    # last speech, which was the whole turn back when dialogue was one segment.
    # Anything concatenating that pair would now drop most of the reply.
    assert len(result["clips"]) > 1
    assert len(result["clip_paths"]) == len(result["clips"])
    assert result["clip_paths"] == sorted(result["clip_paths"], key=lambda p: p.name)


def test_first_audio_is_marked_once_and_lands_in_the_phases():
    flask = pytest.importorskip("flask")
    from request_timing import (
        HeaderFirstAudioMs,
        attach_timing_headers,
        begin_request_timing,
        log_timing,
        note_first_audio_ms,
    )

    app = flask.Flask(__name__)
    with app.test_request_context("/npc-voice-turn"):
        begin_request_timing()
        log_timing("req", "piper segmented tts", __import__("time").perf_counter())
        note_first_audio_ms()
        first = flask.g._timing_first_audio_ms
        # Later clips are the tail of the reply, not the wait before it.
        note_first_audio_ms()
        assert flask.g._timing_first_audio_ms == first

        response = attach_timing_headers(
            flask.make_response(""), "req", include_voice_phases=True
        )

    assert response.headers[HeaderFirstAudioMs] == str(first)
    phases = __import__("json").loads(response.headers["X-Timing-Phases"])
    assert phases["first_audio_ms"] == first
    assert phases["cumulative_ms"]["first_audio"] == first


def test_a_turn_that_never_produced_audio_reports_no_first_audio():
    flask = pytest.importorskip("flask")
    from request_timing import HeaderFirstAudioMs, attach_timing_headers, begin_request_timing, log_timing

    app = flask.Flask(__name__)
    with app.test_request_context("/npc-voice-turn"):
        begin_request_timing()
        log_timing("req", "llm npc response", __import__("time").perf_counter())
        response = attach_timing_headers(
            flask.make_response(""), "req", include_voice_phases=True
        )

    assert HeaderFirstAudioMs not in response.headers
    phases = __import__("json").loads(response.headers["X-Timing-Phases"])
    assert "first_audio_ms" not in phases

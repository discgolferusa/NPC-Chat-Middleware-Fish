"""Progressive event shape. No network."""

import base64

from fish_progressive import chunk_event, end_event, error_event, start_event, stream_id
import turn_progress as tp


def test_stream_events_use_request_and_segment():
    assert stream_id("req", 2) == "req:2"
    started = start_event("req", 2, sample_rate=44100, channels=1)
    assert started["event"] == "audioStreamStart"
    assert started["stream_id"] == "req:2"
    chunk = chunk_event("req", 2, b"\x01\x00", 0)
    assert base64.b64decode(chunk["pcm_base64"]) == b"\x01\x00"
    assert end_event("req", 2)["event"] == "audioStreamEnd"
    assert "Fish" in error_event("req", 2, "")["message"] or error_event("req", 2, "nope")["message"] == "nope"


def test_append_audio_stream_is_visible_on_progress():
    rid = "a" * 32
    tp.register_turn_progress(rid, route="/npc-voice-turn", npc_id="n", player_id="p", mode="voice")
    tp.append_audio_stream(rid, chunk_event(rid, 0, b"\x00\x00", 0))
    progress = tp.get_turn_progress(rid)
    assert progress["audio_stream"][0]["event"] == "audioStreamChunk"
    tp.complete_turn_progress(rid)
    before = len(tp.get_turn_progress(rid)["audio_stream"])
    tp.append_audio_stream(rid, end_event(rid, 0))
    assert len(tp.get_turn_progress(rid)["audio_stream"]) == before

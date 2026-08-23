"""Unit tests for request timing header helpers (no Whisper/server import)."""

from __future__ import annotations

import json
import time

import pytest

pytest.importorskip("flask")

from flask import Flask, Response

import request_timing


def test_attach_timing_headers_includes_total_and_steps():
    app = Flask(__name__)
    with app.test_request_context("/npc-turn", method="POST"):
        request_timing.begin_request_timing()
        step = time.perf_counter() - 0.05
        request_timing.log_timing("rid", "load npc context", step)
        step = time.perf_counter() - 0.12
        request_timing.log_timing("rid", "llm text npc response", step)
        resp = Response(response=b'{"ok":true}', status=200, mimetype="application/json")
        request_timing.attach_timing_headers(resp, "rid")
        assert "X-Timing-Total-Ms" in resp.headers
        assert int(resp.headers["X-Timing-Total-Ms"]) >= 0
        steps = resp.headers.get("X-Timing-Steps", "")
        assert "load npc context=" in steps
        assert "llm text npc response=" in steps
        assert resp.headers.get("X-Turn-Request-Id") == "rid"


def test_build_voice_timing_phases_groups_steps_and_cumulative():
    steps = [
        "save uploaded audio=100",
        "speech-to-text=400",
        "build prompt/messages=50",
        "llm npc response=2000",
        "split tts segments=10",
        "piper segmented tts=800",
    ]
    phases = request_timing.build_voice_timing_phases(steps)
    assert phases["pre_llm_ms"] == 550
    assert phases["llm_ms"] == 2000
    assert phases["post_llm_ms"] == 10
    assert phases["tts_ms"] == 800
    assert phases["cumulative_ms"]["pre_llm"] == 550
    assert phases["cumulative_ms"]["llm"] == 2550
    assert phases["cumulative_ms"]["post_llm"] == 2560
    assert phases["cumulative_ms"]["tts"] == 3360


def test_attach_timing_headers_includes_voice_phases_json():
    app = Flask(__name__)
    with app.test_request_context("/npc-voice-turn", method="POST"):
        request_timing.begin_request_timing()
        step = time.perf_counter()
        step = request_timing.log_timing("rid", "save uploaded audio", step)
        step = request_timing.log_timing("rid", "llm npc response", step)
        step = request_timing.log_timing("rid", "piper segmented tts", step)
        resp = Response(response=b"RIFF", status=200, mimetype="audio/wav")
        request_timing.attach_timing_headers(resp, "rid")
        raw = resp.headers.get("X-Timing-Phases")
        assert raw
        phases = json.loads(raw)
        assert phases["llm_ms"] >= 0
        assert "cumulative_ms" in phases
        assert phases["cumulative_ms"]["middleware"] >= phases["cumulative_ms"]["tts"]

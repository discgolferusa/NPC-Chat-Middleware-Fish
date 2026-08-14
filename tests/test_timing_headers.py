"""Unit tests for request timing header helpers (no Whisper/server import)."""

from __future__ import annotations

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

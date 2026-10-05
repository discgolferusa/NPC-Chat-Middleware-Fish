"""Unit tests for request timing header helpers (no Whisper/server import)."""

from __future__ import annotations

import json
import time

import pytest

pytest.importorskip("flask")

from flask import Flask, Response, g

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
        assert "X-Timing-Phases" not in resp.headers


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
    assert phases["stt_ms"] == 400
    assert phases["pre_llm_ms"] == 150
    assert phases["llm_ms"] == 2000
    assert phases["post_llm_ms"] == 10
    assert phases["tts_ms"] == 800
    assert phases["cumulative_ms"]["stt"] == 400
    assert phases["cumulative_ms"]["pre_llm"] == 550
    assert phases["cumulative_ms"]["llm"] == 2550
    assert phases["cumulative_ms"]["post_llm"] == 2560
    assert phases["cumulative_ms"]["tts"] == 3360


def test_attach_timing_headers_includes_stt_attribution():
    app = Flask(__name__)
    with app.test_request_context("/npc-voice-turn", method="POST"):
        request_timing.begin_request_timing()
        request_timing.note_request_stt(
            model_id="openrouter-assemblyai-universal-3-5-pro",
            model="assemblyai/universal-3-5-pro",
            provider="openrouter",
            transcription_ms=321,
            cost=0.00012,
            used_fallback=False,
        )
        step = time.perf_counter()
        step = request_timing.log_timing("rid", "speech-to-text", step)
        step = request_timing.log_timing("rid", "llm npc response", step)
        resp = Response(response=b"RIFF", status=200, mimetype="audio/wav")
        request_timing.attach_timing_headers(resp, "rid", include_voice_phases=True)
        assert resp.headers.get("X-Stt-Model-Id") == "openrouter-assemblyai-universal-3-5-pro"
        assert resp.headers.get("X-Stt-Provider") == "openrouter"
        assert resp.headers.get("X-Stt-Model") == "assemblyai/universal-3-5-pro"
        assert resp.headers.get("X-Stt-Transcription-Ms") == "321"
        assert resp.headers.get("X-Stt-Used-Fallback") == "0"
        phases = json.loads(resp.headers["X-Timing-Phases"])
        assert phases["stt_model_id"] == "openrouter-assemblyai-universal-3-5-pro"
        assert phases["stt_provider"] == "openrouter"
        assert phases["stt_ms"] >= 0
        assert phases["stt_cost_usd"] == 0.00012


def test_attach_timing_headers_voice_phases_only_when_requested():
    app = Flask(__name__)
    with app.test_request_context("/npc-voice-turn", method="POST"):
        request_timing.begin_request_timing()
        request_timing.note_request_llm_model(
            active_model_id="openrouter-gpt-oss-20b",
            selected_model_id="openrouter-gpt-oss-20b",
            purpose="chat",
        )
        step = time.perf_counter()
        step = request_timing.log_timing("rid", "save uploaded audio", step)
        step = request_timing.log_timing("rid", "llm npc response", step)
        step = request_timing.log_timing("rid", "piper segmented tts", step)
        resp = Response(response=b"RIFF", status=200, mimetype="audio/wav")
        request_timing.attach_timing_headers(resp, "rid", include_voice_phases=True)
        raw = resp.headers.get("X-Timing-Phases")
        assert raw
        phases = json.loads(raw)
        assert phases["llm_ms"] >= 0
        assert "cumulative_ms" in phases
        assert phases["cumulative_ms"]["middleware"] >= phases["cumulative_ms"]["tts"]
        assert resp.headers.get("X-Llm-Model-Id") == "openrouter-gpt-oss-20b"
        assert resp.headers.get("X-Llm-Selected-Model-Id") == "openrouter-gpt-oss-20b"
        assert phases["model_id"] == "openrouter-gpt-oss-20b"


def test_attach_timing_headers_uses_request_local_model_not_process_wide(monkeypatch):
    app = Flask(__name__)

    class FakeRouter:
        PURPOSE_CHAT = "chat"
        LAST_ACTIVE_MODEL_ID = {"chat": "wrong-stolen-model"}
        CURRENT_LLM_MODEL_ID = "wrong-stolen-model"

    monkeypatch.setitem(__import__("sys").modules, "llm_router", FakeRouter())

    with app.test_request_context("/npc-turn", method="POST"):
        request_timing.begin_request_timing()
        request_timing.note_request_llm_model(
            active_model_id="openrouter-deepseek-v4-flash",
            selected_model_id="openrouter-gpt-oss-20b",
            purpose="chat",
        )
        # Simulate another concurrent call overwriting process-wide state.
        FakeRouter.LAST_ACTIVE_MODEL_ID["chat"] = "wrong-stolen-model"
        resp = Response(response=b'{"ok":true}', status=200, mimetype="application/json")
        request_timing.attach_timing_headers(resp, "rid")
        assert resp.headers.get("X-Llm-Model-Id") == "openrouter-deepseek-v4-flash"
        assert resp.headers.get("X-Llm-Selected-Model-Id") == "openrouter-gpt-oss-20b"


def test_note_request_llm_model_first_chat_write_wins():
    app = Flask(__name__)
    with app.test_request_context("/npc-voice-turn", method="POST"):
        request_timing.begin_request_timing()
        request_timing.note_request_llm_model(
            active_model_id="npc-turn-model",
            selected_model_id="npc-primary",
            purpose="chat",
        )
        # Sync/async memory extraction must not steal attribution.
        request_timing.note_request_llm_model(
            active_model_id="memory-model",
            selected_model_id="memory-primary",
            purpose="chat",
        )
        assert g._timing_llm_model_id == "npc-turn-model"
        assert g._timing_llm_selected_model_id == "npc-primary"
        request_timing.log_timing("rid", "llm npc response", time.perf_counter() - 0.01)
        resp = Response(response=b"RIFF", status=200, mimetype="audio/wav")
        request_timing.attach_timing_headers(resp, "rid", include_voice_phases=True)
        assert resp.headers.get("X-Llm-Model-Id") == "npc-turn-model"
        phases = json.loads(resp.headers["X-Timing-Phases"])
        assert phases["model_id"] == "npc-turn-model"


def test_attach_timing_headers_includes_routed_provider():
    app = Flask(__name__)
    with app.test_request_context("/npc-voice-turn", method="POST"):
        request_timing.begin_request_timing()
        request_timing.note_request_llm_model(
            active_model_id="openrouter-gpt-oss-20b",
            selected_model_id="openrouter-gpt-oss-20b",
            provider="Amazon Bedrock",
            purpose="chat",
        )
        request_timing.log_timing("rid", "llm npc response", time.perf_counter() - 0.01)
        resp = Response(response=b"RIFF", status=200, mimetype="audio/wav")
        request_timing.attach_timing_headers(resp, "rid", include_voice_phases=True)
        assert resp.headers.get("X-Llm-Provider") == "Amazon Bedrock"
        assert json.loads(resp.headers["X-Timing-Phases"])["provider"] == "Amazon Bedrock"


def test_note_request_llm_model_provider_follows_first_attributed_model():
    app = Flask(__name__)
    with app.test_request_context("/npc-turn", method="POST"):
        request_timing.begin_request_timing()
        request_timing.note_request_llm_model(
            active_model_id="openrouter-nemotron-3.5-lightning-coreweave",
            selected_model_id="openrouter-nemotron-3.5-lightning-coreweave",
            provider="CoreWeave",
            purpose="chat",
        )
        # A later memory-extraction hop on another host must not relabel the turn.
        request_timing.note_request_llm_model(
            active_model_id="local",
            selected_model_id="local",
            provider="local",
            purpose="chat",
        )
        resp = Response(response=b'{"ok":true}', status=200, mimetype="application/json")
        request_timing.attach_timing_headers(resp, "rid")
        assert resp.headers.get("X-Llm-Model-Id") == "openrouter-nemotron-3.5-lightning-coreweave"
        assert resp.headers.get("X-Llm-Provider") == "CoreWeave"


def test_attach_timing_headers_omits_provider_when_unknown():
    app = Flask(__name__)
    with app.test_request_context("/npc-turn", method="POST"):
        request_timing.begin_request_timing()
        request_timing.note_request_llm_model(
            active_model_id="openrouter-gpt-oss-20b",
            selected_model_id="openrouter-gpt-oss-20b",
            purpose="chat",
        )
        resp = Response(response=b'{"ok":true}', status=200, mimetype="application/json")
        request_timing.attach_timing_headers(resp, "rid")
        assert "X-Llm-Provider" not in resp.headers


def test_note_request_llm_model_ignores_content_creation_purpose():
    app = Flask(__name__)
    with app.test_request_context("/npc-turn", method="POST"):
        request_timing.begin_request_timing()
        request_timing.note_request_llm_model(
            active_model_id="content-model",
            selected_model_id="content-model",
            purpose="content_creation",
        )
        assert getattr(g, "_timing_llm_model_id", None) is None
        resp = Response(response=b'{"ok":true}', status=200, mimetype="application/json")
        request_timing.attach_timing_headers(resp, "rid")
        assert "X-Llm-Model-Id" not in resp.headers

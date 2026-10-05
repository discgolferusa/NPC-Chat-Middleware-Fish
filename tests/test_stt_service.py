"""Unit + functional tests for OpenRouter STT catalog hierarchy."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

import stt_catalog
import stt_service
from stt_capabilities import get_capabilities
from stt_context import apply_context_to_request, build_stt_context
from stt_openrouter import STTProviderError, call_openrouter_transcription


@pytest.fixture()
def stt_tmp(tmp_path, monkeypatch):
    catalog = tmp_path / "stt_catalog.json"
    runtime = tmp_path / "stt_runtime_config.json"
    stt_catalog.configure(catalog_file=catalog, runtime_file=runtime)
    stt_catalog.load_or_seed()
    # Clear any account-wide OpenRouter cooldown from prior tests.
    with stt_service._ONLINE_DISABLED_LOCK:
        stt_service._ONLINE_DISABLED_UNTIL = 0.0
        stt_service._ONLINE_DISABLED_REASON = ""
    monkeypatch.setenv("STT_MAX_OPENROUTER_ATTEMPTS", "2")
    monkeypatch.setenv("STT_CHAIN_BUDGET_SECONDS", "30")
    yield tmp_path
    # Leave files for debugging on failure; next fixture call reconfigures.


def test_seed_default_primary_is_assemblyai(stt_tmp):
    assert stt_catalog.CURRENT_MODEL_ID == "openrouter-assemblyai-universal-3-5-pro"
    assert stt_catalog.PROVIDER_MODE == "openrouter_with_local_fallback"
    chain = stt_catalog.get_attempt_chain()
    assert chain[0] == "openrouter-assemblyai-universal-3-5-pro"
    assert "local-whisper-lite" in chain


def test_set_model_order_updates_runtime(stt_tmp):
    ordered = stt_catalog.set_model_order(
        [
            "openrouter-deepgram-nova-3",
            "local-whisper-lite",
        ],
        provider_mode="openrouter_with_local_fallback",
    )
    assert ordered[0] == "openrouter-deepgram-nova-3"
    assert stt_catalog.CURRENT_MODEL_ID == "openrouter-deepgram-nova-3"
    assert stt_catalog.FALLBACK_MODEL_IDS == ["local-whisper-lite"]
    runtime = json.loads((stt_tmp / "stt_runtime_config.json").read_text(encoding="utf-8"))
    assert runtime["model_id"] == "openrouter-deepgram-nova-3"


def test_provider_mode_local_filters_chain(stt_tmp):
    stt_catalog.set_model_order(
        [
            "openrouter-assemblyai-universal-3-5-pro",
            "local-whisper-lite",
        ],
        provider_mode="local",
    )
    assert stt_catalog.get_attempt_chain() == ["local-whisper-lite"]


def test_capabilities_assemblyai_supports_keyterms():
    caps = get_capabilities("assemblyai/universal-3-5-pro")
    assert caps.supports_keyterms
    assert caps.supports_context_prompt
    assert caps.use_top_level_keyterms


def test_capabilities_whisper_no_keyterms():
    caps = get_capabilities("openai/whisper-large-v3-turbo")
    assert not caps.supports_keyterms
    assert not caps.supports_context_prompt


def test_context_builder_trims_and_skips_secrets():
    ctx = build_stt_context(
        npc={
            "name": "Theodora Komiskey",
            "aliases": ["Komiskey"],
            "secrets": ["Knows about the egg"],
            "known_facts": ["Hidden agenda"],
        },
        location={"name": "Hadley's Hope", "sublocations": ["Atmospheric Processor"]},
        colocated_npcs=[
            ("michael_drapers", {"name": "Michael Drapers"}),
            ("miranda", {"name": "Miranda Reynolds"}),
        ],
        campaign={
            "name": "Alien Colonial",
            "factions": ["Weyland-Yutani", "Colonial Marines"],
            "species": ["Xenomorph"],
        },
        max_keyterms=20,
    )
    joined = " | ".join(ctx.keyterms)
    assert "Komiskey" in joined
    assert "Hadley's Hope" in joined
    assert "Weyland-Yutani" in joined
    assert "Knows about the egg" not in joined
    assert "Hidden agenda" not in joined
    assert len(ctx.keyterms) <= 20
    assert "tabletop RPG" in ctx.prompt


def test_apply_context_skips_keyterms_when_unsupported():
    caps = get_capabilities("openai/whisper-large-v3-turbo")
    ctx = build_stt_context(npc={"name": "Torin Ashhand"})
    payload = {"model": "openai/whisper-large-v3-turbo"}
    apply_context_to_request(payload, ctx, caps)
    assert "keyterms" not in payload
    assert "provider" not in payload
    assert payload.get("language") == "en"


def test_openrouter_success_normalized(stt_tmp, monkeypatch, tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF....WAVEfmt ")

    class _Resp:
        status_code = 200
        headers = {"X-Generation-Id": "gen-123"}
        text = "{}"

        def json(self):
            return {
                "text": "What does Komiskey know?",
                "usage": {"seconds": 4.2, "cost": 0.0001},
            }

    monkeypatch.setattr(
        "stt_openrouter.requests.post",
        lambda *a, **k: _Resp(),
    )
    monkeypatch.setattr("llm_router.OPENROUTER_API_KEY", "test-key", raising=False)
    monkeypatch.setattr(
        "llm_router._openrouter_api_root",
        lambda: "https://openrouter.ai/api/v1",
        raising=False,
    )
    monkeypatch.setattr(
        "llm_router._openrouter_auth_headers",
        lambda: {"Authorization": "Bearer test-key", "Content-Type": "application/json"},
        raising=False,
    )

    raw = call_openrouter_transcription(wav, model="assemblyai/universal-3-5-pro")
    assert raw.text.startswith("What does Komiskey")
    assert raw.request_id == "gen-123"
    assert raw.audio_duration_ms == 4200


def test_openrouter_timeout_is_retryable(stt_tmp, monkeypatch, tmp_path):
    import requests as req_mod

    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")

    def _raise(*a, **k):
        raise req_mod.Timeout("slow")

    monkeypatch.setattr("stt_openrouter.requests.post", _raise)
    monkeypatch.setattr("llm_router.OPENROUTER_API_KEY", "test-key", raising=False)
    monkeypatch.setattr(
        "llm_router._openrouter_api_root",
        lambda: "https://openrouter.ai/api/v1",
        raising=False,
    )
    monkeypatch.setattr(
        "llm_router._openrouter_auth_headers",
        lambda: {"Authorization": "Bearer x", "Content-Type": "application/json"},
        raising=False,
    )
    with pytest.raises(STTProviderError) as exc:
        call_openrouter_transcription(wav, model="deepgram/nova-3", timeout_seconds=1)
    assert exc.value.error_type == "timeout"
    assert exc.value.fallbackable


def test_openrouter_rate_limit(stt_tmp, monkeypatch, tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")

    class _Resp:
        status_code = 429
        headers = {}
        text = "rate limited"

        def json(self):
            return {"error": {"message": "rate limited"}}

    monkeypatch.setattr("stt_openrouter.requests.post", lambda *a, **k: _Resp())
    monkeypatch.setattr("llm_router.OPENROUTER_API_KEY", "test-key", raising=False)
    monkeypatch.setattr(
        "llm_router._openrouter_api_root",
        lambda: "https://openrouter.ai/api/v1",
        raising=False,
    )
    monkeypatch.setattr(
        "llm_router._openrouter_auth_headers",
        lambda: {"Authorization": "Bearer x", "Content-Type": "application/json"},
        raising=False,
    )
    with pytest.raises(STTProviderError) as exc:
        call_openrouter_transcription(wav, model="deepgram/nova-3")
    assert exc.value.error_type == "rate_limit"


def test_hierarchy_falls_back_to_local(stt_tmp, monkeypatch, tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF....")
    stt_catalog.set_model_order(
        [
            "openrouter-assemblyai-universal-3-5-pro",
            "local-whisper-lite",
        ],
        provider_mode="openrouter_with_local_fallback",
    )

    def _fail(*a, **k):
        raise STTProviderError("boom", error_type="timeout", retryable=True, fallbackable=True)

    class _FakeModel:
        def transcribe(self, *a, **k):
            return [SimpleNamespace(text=" Hello local.")], SimpleNamespace(duration=1.0)

    monkeypatch.setattr(stt_service, "_attempt_openrouter", _fail)
    monkeypatch.setattr(stt_service, "ensure_local_whisper", lambda **kw: _FakeModel())

    result = stt_service.transcribe(wav, pad_fn=lambda p: p, duration_fn=lambda p: 1.0)
    assert result.ok
    assert result.provider == "local"
    assert result.used_fallback
    assert "Hello local" in result.text


def test_hierarchy_fallback_disabled_stops(stt_tmp, monkeypatch, tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF....")
    stt_catalog.set_model_order(
        ["openrouter-assemblyai-universal-3-5-pro"],
        provider_mode="openrouter",
    )

    def _fail(*a, **k):
        raise STTProviderError(
            "timeout",
            error_type="timeout",
            retryable=True,
            fallbackable=True,
        )

    monkeypatch.setattr(stt_service, "_attempt_openrouter", _fail)
    result = stt_service.transcribe(wav)
    assert not result.ok
    assert result.error_type == "timeout"


def test_malformed_audio_does_not_fallback(stt_tmp, monkeypatch, tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF....")
    stt_catalog.set_model_order(
        [
            "openrouter-assemblyai-universal-3-5-pro",
            "local-whisper-lite",
        ]
    )

    def _fail(*a, **k):
        raise STTProviderError(
            "bad audio",
            error_type="malformed_audio",
            retryable=False,
            fallbackable=False,
        )

    local_called = {"n": 0}

    def _local(*a, **k):
        local_called["n"] += 1
        return "should not run", 1

    monkeypatch.setattr(stt_service, "_attempt_openrouter", _fail)
    monkeypatch.setattr(stt_service, "_transcribe_local", _local)
    result = stt_service.transcribe(wav)
    assert not result.ok
    assert result.error_type == "malformed_audio"
    assert local_called["n"] == 0


def test_auth_failure_skips_remaining_openrouter_and_uses_local(stt_tmp, monkeypatch, tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF....")
    stt_catalog.set_model_order(
        [
            "openrouter-assemblyai-universal-3-5-pro",
            "openrouter-deepgram-nova-3",
            "local-whisper-lite",
        ],
        provider_mode="openrouter_with_local_fallback",
    )
    openrouter_calls = {"n": 0}

    def _fail(*a, **k):
        openrouter_calls["n"] += 1
        raise STTProviderError(
            "bad key",
            error_type="authentication",
            retryable=False,
            fallbackable=True,
        )

    class _FakeModel:
        def transcribe(self, *a, **k):
            return [SimpleNamespace(text=" Local ok.")], SimpleNamespace(duration=1.0)

    monkeypatch.setattr(stt_service, "_attempt_openrouter", _fail)
    monkeypatch.setattr(stt_service, "ensure_local_whisper", lambda **kw: _FakeModel())
    result = stt_service.transcribe(wav, pad_fn=lambda p: p, duration_fn=lambda p: 1.0)
    assert result.ok
    assert result.provider == "local"
    assert openrouter_calls["n"] == 1


def test_cloud_transcript_not_filtered_by_whisper_hallucination(
    stt_tmp, monkeypatch, tmp_path
):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF....")
    stt_catalog.set_model_order(
        ["openrouter-assemblyai-universal-3-5-pro"],
        provider_mode="openrouter",
    )

    def _ok(*a, **k):
        return stt_service.STTResult(
            text="Yeah",
            provider="openrouter",
            model="assemblyai/universal-3-5-pro",
            model_id="openrouter-assemblyai-universal-3-5-pro",
            ok=True,
            transcription_ms=12,
            audio_duration_ms=900,
        )

    monkeypatch.setattr(stt_service, "_attempt_openrouter", _ok)
    # Whisper short-clip heuristic would reject "yeah"; cloud must keep it.
    result = stt_service.transcribe(
        wav,
        duration_fn=lambda p: 0.9,
        peak_fn=lambda p: 900,
        hallucination_fn=lambda text, **kw: True,
    )
    assert result.ok
    assert result.text == "Yeah"


def test_keyterms_trimmed_to_openrouter_char_limit():
    long_term = "X" * 140
    ctx = build_stt_context(npc={"name": long_term, "aliases": [long_term]})
    assert all(len(term) <= 100 for term in ctx.keyterms)
    caps = get_capabilities("assemblyai/universal-3-5-pro")
    payload = {"model": "assemblyai/universal-3-5-pro"}
    apply_context_to_request(payload, ctx, caps)
    for term in payload.get("keyterms") or []:
        assert 1 <= len(term) <= 100


def test_deepgram_uses_top_level_keyterms():
    caps = get_capabilities("deepgram/nova-3")
    assert caps.use_top_level_keyterms
    ctx = build_stt_context(npc={"name": "Komiskey"})
    payload = {"model": "deepgram/nova-3"}
    apply_context_to_request(payload, ctx, caps)
    assert "Komiskey" in (payload.get("keyterms") or [])
    assert "provider" not in payload


def test_stt_select_api(api, stt_tmp):
    client, server, headers = api
    # Point server module's catalog at tmp files already configured by stt_tmp.
    response = client.post(
        "/stt/select",
        headers=headers,
        json={
            "model_order": [
                "openrouter-deepgram-nova-3",
                "local-whisper-lite",
            ],
            "provider_mode": "openrouter_with_local_fallback",
        },
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["model_id"] == "openrouter-deepgram-nova-3"
    assert body["model_order"][0] == "openrouter-deepgram-nova-3"

    models = client.get("/stt/models", headers=headers)
    assert models.status_code == 200
    payload = models.get_json()
    assert payload["current_model_id"] == "openrouter-deepgram-nova-3"

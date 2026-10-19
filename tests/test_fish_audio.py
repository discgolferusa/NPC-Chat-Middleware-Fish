"""Fish provider, catalog, tags, and PCM decode. No live network."""

from __future__ import annotations

import io
import wave
from pathlib import Path

import pytest
import requests

from fish_config import FishConfigError, enforce_startup, load_settings, native_model_header
from fish_tags import canonical_text, is_allowed_tag, tts_markup
from fish_tts import FishSpeechError, FishVoiceError, pcm_to_wav, reset_client_for_tests, synthesize_wav
from fish_voices import clear_cache, get_voice, search_voices, to_voice_card
from fish_stream_fx import streaming_af_filter
from scripts.probe_fish_drama3 import measure_stream


class _FakeResponse:
    def __init__(self, *, status_code=200, headers=None, chunks=None, text="", json_body=None):
        self.status_code = status_code
        self.headers = headers or {}
        self._chunks = list(chunks or [])
        self.text = text
        self._json = json_body or {}
        self.closed = False

    def iter_content(self, chunk_size=4096):
        del chunk_size
        yield from self._chunks

    def json(self):
        return self._json

    def close(self):
        self.closed = True


def _pcm_tone(n=800):
    import array

    return array.array("h", [1000] * n).tobytes()


def test_defaults_are_fish_audio_drama3(monkeypatch):
    monkeypatch.delenv("FISH_TTS_PROVIDER", raising=False)
    monkeypatch.delenv("FISH_TTS_MODEL", raising=False)
    settings = load_settings()
    assert settings.provider == "fish-audio"
    assert settings.model == "fish-audio/drama-3"
    assert native_model_header(settings.model) == "drama-3"
    assert settings.progressive is False


def test_invalid_provider_and_empty_model_fail(monkeypatch):
    monkeypatch.setenv("FISH_TTS_PROVIDER", "piper")
    with pytest.raises(FishConfigError):
        load_settings()
    monkeypatch.setenv("FISH_TTS_PROVIDER", "fish")
    with pytest.raises(FishConfigError):
        load_settings()
    monkeypatch.setenv("FISH_TTS_PROVIDER", "fish-audio")
    monkeypatch.setenv("FISH_TTS_MODEL", "   ")
    with pytest.raises(FishConfigError):
        load_settings()


def test_startup_requires_fish_key(monkeypatch):
    monkeypatch.setenv("FISH_TTS_SKIP_STARTUP_CHECK", "0")
    monkeypatch.setenv("FISH_API_KEY", "")
    with pytest.raises(SystemExit):
        enforce_startup()


def test_openrouter_request_uses_full_slug(tmp_path, monkeypatch):
    monkeypatch.setenv("FISH_TTS_PROVIDER", "openrouter")
    monkeypatch.setenv("FISH_TTS_MODEL", "fish-audio/drama-3")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    monkeypatch.setenv("FISH_API_KEY", "fish-test")
    reset_client_for_tests()
    captured = {}

    def post(url, headers, json, stream, timeout):
        captured.update(url=url, headers=headers, json=json, stream=stream, timeout=timeout)
        return _FakeResponse(
            headers={"Content-Type": "audio/pcm; rate=44100; channels=1"},
            chunks=[_pcm_tone()],
        )

    out = tmp_path / "out.wav"
    metrics = synthesize_wav("Hello there.", "voice-9", out, http_post=post)
    assert captured["json"]["model"] == "fish-audio/drama-3"
    assert captured["json"]["voice"] == "voice-9"
    assert "reference_id" not in captured["json"]
    assert out.exists()
    assert metrics["provider"] == "openrouter"
    with wave.open(str(out), "rb") as handle:
        assert handle.getframerate() == 44100
        assert handle.getnchannels() == 1


def test_fish_audio_request_uses_header_and_reference_id(tmp_path, monkeypatch):
    monkeypatch.setenv("FISH_TTS_PROVIDER", "fish-audio")
    monkeypatch.setenv("FISH_TTS_MODEL", "fish-audio/drama-3")
    monkeypatch.setenv("FISH_API_KEY", "fish-test")
    reset_client_for_tests()
    captured = {}

    def post(url, headers, json, stream, timeout):
        captured.update(url=url, headers=headers, json=json)
        return _FakeResponse(status_code=429, text="slow down")

    with pytest.raises(FishSpeechError) as raised:
        synthesize_wav("Hello.", "voice-9", tmp_path / "x.wav", http_post=post)
    assert raised.value.status == 429
    assert captured["url"] == "https://api.fish.audio/v1/tts"
    assert captured["headers"]["model"] == "drama-3"
    assert captured["json"]["reference_id"] == "voice-9"
    assert "voice" not in captured["json"]


def test_missing_voice_does_not_pick_a_random_one(tmp_path, monkeypatch):
    monkeypatch.delenv("FISH_DEFAULT_VOICE_ID", raising=False)
    monkeypatch.setenv("FISH_API_KEY", "fish-test")
    calls = {"n": 0}

    def post(*_args, **_kwargs):
        calls["n"] += 1
        raise AssertionError("speech request should not be sent")

    with pytest.raises(FishVoiceError):
        synthesize_wav("Hello.", "", tmp_path / "x.wav", http_post=post)
    assert calls["n"] == 0


def test_failed_fish_response_is_not_retried(tmp_path, monkeypatch):
    monkeypatch.setenv("FISH_TTS_PROVIDER", "fish-audio")
    monkeypatch.setenv("FISH_API_KEY", "fish-test")
    calls = {"n": 0}

    def post(*_args, **_kwargs):
        calls["n"] += 1
        return _FakeResponse(status_code=500, text="nope")

    with pytest.raises(FishSpeechError):
        synthesize_wav("Hello.", "voice-9", tmp_path / "x.wav", http_post=post)
    assert calls["n"] == 1


def test_catalog_pagination_and_cache(monkeypatch):
    clear_cache()
    monkeypatch.setenv("FISH_API_KEY", "fish-test")
    monkeypatch.setenv("FISH_CATALOG_CACHE_SECONDS", "60")
    calls = {"n": 0}

    def get(url, params, headers, timeout):
        calls["n"] += 1
        assert url.endswith("/model")
        assert params["language"] == "en"
        assert params["licensed"] == "true"
        assert headers["Authorization"] == "Bearer fish-test"
        return _FakeResponse(
            json_body={
                "total": 2,
                "has_more": True,
                "items": [
                    {
                        "_id": "abc",
                        "type": "tts",
                        "title": "Harbor",
                        "description": "Low voice",
                        "tags": ["male"],
                        "languages": ["en"],
                        "state": "trained",
                        "licensed": True,
                        "visibility": "public",
                        "samples": [{"title": "s", "text": "hi", "task_id": "t", "audio": "https://cdn.example/a.mp3"}],
                        "author": {"_id": "u", "nickname": "Ada", "avatar": ""},
                        "like_count": 1,
                        "mark_count": 0,
                        "shared_count": 0,
                        "task_count": 3,
                    }
                ],
            }
        )

    page = search_voices(title="harbor", page=1, page_size=5, http_get=get)
    again = search_voices(title="harbor", page=1, page_size=5, http_get=get)
    assert calls["n"] == 1
    assert page["provider"] == "fish-audio"
    assert page["voices"][0]["id"] == "abc"
    assert page["voices"][0]["preview_url"] == "https://cdn.example/a.mp3"
    assert again["voices"][0]["title"] == "Harbor"
    assert "commercial-use" in page["licensing_note"]


def test_unavailable_voice_is_not_replaced():
    def get(url, headers, timeout):
        assert url.endswith("/model/gone")
        return _FakeResponse(status_code=404)

    assert get_voice("gone", http_get=get) is None


def test_dmca_voice_is_omitted():
    assert to_voice_card({"_id": "x", "type": "tts", "title": "X", "state": "trained", "dmca_taken_down": True}) is None


def test_tags_stay_out_of_canonical_text():
    raw = "[angry] I told you not to open that door. [whispering] There's something inside. [moaning] no"
    canon = canonical_text(raw)
    spoken = tts_markup(raw)
    assert "[angry]" not in canon
    assert "moaning" not in canon
    assert "[angry]" in spoken
    assert "[whispering]" in spoken
    assert "moaning" not in spoken
    assert is_allowed_tag("long-break")
    assert not is_allowed_tag("pause")


def test_streaming_filter_keeps_echo_and_drops_reverse_trim():
    graph = streaming_af_filter(fx_preset="cave", pitch_semitones=0, sample_rate=44100)
    assert "aecho" in graph
    assert "areverse" not in graph
    assert "dynaudnorm" not in graph
    assert "asetrate=22050" in graph or "aresample" in graph or "highpass" in graph


def test_probe_measure_marks_incremental_chunks():
    import time

    started = time.perf_counter()

    def chunks():
        yield b"abc"
        time.sleep(0.06)
        yield b"defg"

    measured = measure_stream(chunks(), started=started)
    assert measured["chunk_count"] == 2
    assert measured["byte_count"] == 7
    assert measured["incremental"] is True
    assert measured["ttfab_ms"] is not None


def test_pcm_wrap_roundtrip(tmp_path: Path):
    pcm = _pcm_tone(4)
    path = tmp_path / "t.wav"
    pcm_to_wav(pcm, path, sample_rate=22050, channels=1)
    with wave.open(str(path), "rb") as handle:
        assert handle.readframes(handle.getnframes()) == pcm


def test_tts_info_reports_env(api, monkeypatch):
    client, _, headers = api
    monkeypatch.setenv("FISH_TTS_PROVIDER", "fish-audio")
    monkeypatch.setenv("FISH_TTS_MODEL", "fish-audio/drama-3")
    res = client.get("/tts/info", headers=headers)
    assert res.status_code == 200
    body = res.get_json()
    assert body["speech_provider"] == "fish-audio"
    assert body["model"] == "fish-audio/drama-3"

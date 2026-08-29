"""
Warming Piper during a turn starves speech-to-text: the replica loaders take the
CPU that STT is running on, so a 3s clip took 10.5s to transcribe while the pool
filled to 10/10. These pin the warm happening at startup instead.
"""

from __future__ import annotations

from pathlib import Path

import server


def _capture_warm(monkeypatch) -> list[list[Path]]:
    calls: list[list[Path]] = []
    monkeypatch.setattr(
        server.piper_pool,
        "warm_voice_models",
        lambda paths, request_id=None: calls.append(list(paths)),
    )
    return calls


def test_startup_warms_the_narrator_voice(monkeypatch, tmp_path):
    calls = _capture_warm(monkeypatch)
    monkeypatch.setattr(
        server, "get_narrator_tts_settings", lambda: {"voice_model": "en_US-libritts-high.onnx"}
    )
    monkeypatch.setattr(server, "resolve_voice_model_path", lambda name: Path(str(name)))
    monkeypatch.setattr(server, "NPC_DIR", tmp_path)

    server.warm_piper_at_startup()

    assert calls == [[Path("en_US-libritts-high.onnx")]]


def test_startup_warm_can_be_turned_off(monkeypatch):
    calls = _capture_warm(monkeypatch)
    monkeypatch.setenv("PIPER_WARM_AT_STARTUP", "0")

    server.warm_piper_at_startup()

    assert calls == []


def test_startup_warm_is_skipped_for_subprocess_piper(monkeypatch):
    calls = _capture_warm(monkeypatch)
    monkeypatch.setenv("PIPER_USE_SUBPROCESS", "1")

    server.warm_piper_at_startup()

    assert calls == []


def test_an_unreadable_narrator_does_not_stop_the_server(monkeypatch, tmp_path):
    calls = _capture_warm(monkeypatch)

    def _boom():
        raise RuntimeError("no narrator settings")

    monkeypatch.setattr(server, "get_narrator_tts_settings", _boom)
    monkeypatch.setattr(server, "NPC_DIR", tmp_path)

    server.warm_piper_at_startup()

    assert calls == []

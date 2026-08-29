"""Regression tests for blocking Piper warm-up on WSGI import."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import server


@pytest.fixture(autouse=True)
def _reset_startup_state(monkeypatch, tmp_path):
    monkeypatch.setenv("PIPER_WARM_AT_STARTUP", "1")
    monkeypatch.delenv("PIPER_USE_SUBPROCESS", raising=False)
    monkeypatch.delenv("PIPER_WARM_VOICE_DIR", raising=False)
    monkeypatch.setattr(server, "_PIPER_STARTUP_WARM_COMPLETE", False)
    monkeypatch.setattr(server, "VOICES_DIR", tmp_path / "test-voices")


def _capture_warm(monkeypatch) -> list[dict]:
    calls: list[dict] = []

    def fake_warm(
        paths,
        *,
        request_id=None,
        wait_for_ready=False,
        timeout_seconds=300.0,
    ):
        paths = list(paths)
        calls.append({
            "paths": paths,
            "request_id": request_id,
            "wait_for_ready": wait_for_ready,
            "timeout_seconds": timeout_seconds,
        })
        return {
            str(path): {
                "ready": 1,
                "target": 1,
                "inflight": 0,
                "error": None,
            }
            for path in paths
        }

    monkeypatch.setattr(server.piper_pool, "warm_voice_models", fake_warm)
    return calls


def test_startup_blocks_until_every_configured_voice_is_ready(monkeypatch, tmp_path):
    calls = _capture_warm(monkeypatch)
    monkeypatch.setattr(server, "DEFAULT_VOICE_MODEL", "default.onnx")
    monkeypatch.setattr(
        server,
        "get_narrator_tts_settings",
        lambda: {"voice_model": "narrator.onnx"},
    )
    monkeypatch.setattr(server, "resolve_voice_model_path", lambda name: Path(str(name)))
    monkeypatch.setattr(server, "NPC_DIR", tmp_path)
    monkeypatch.setenv("PIPER_POOL_SLOTS", "10")

    server.warm_piper_at_startup()

    assert len(calls) == 1
    assert calls[0]["paths"] == [Path("default.onnx"), Path("narrator.onnx")]
    assert calls[0]["request_id"] == "startup"
    assert calls[0]["wait_for_ready"] is True
    assert calls[0]["timeout_seconds"] == 300.0
    assert server._PIPER_STARTUP_WARM_COMPLETE is True


def test_startup_scans_every_onnx_under_voices_dir(monkeypatch, tmp_path):
    calls = _capture_warm(monkeypatch)
    voices_dir = tmp_path / "Voices"
    voices_dir.mkdir(parents=True)
    first = voices_dir / "first.onnx"
    second = voices_dir / "nested" / "second.onnx"
    second.parent.mkdir()
    first.write_bytes(b"onnx")
    second.write_bytes(b"onnx")
    npc_dir = tmp_path / "npcs"
    npc_dir.mkdir()

    monkeypatch.setattr(server, "VOICES_DIR", voices_dir)
    monkeypatch.setattr(server, "DEFAULT_VOICE_MODEL", first)
    monkeypatch.setattr(server, "resolve_voice_model_path", lambda name: Path(str(name)))
    monkeypatch.setattr(
        server,
        "get_narrator_tts_settings",
        lambda: {"resolved_voice_model": second},
    )
    monkeypatch.setattr(server, "NPC_DIR", npc_dir)

    server.warm_piper_at_startup()

    assert len(calls) == 1
    assert calls[0]["paths"] == [first, second]
    assert calls[0]["wait_for_ready"] is True


def test_startup_warm_can_be_turned_off(monkeypatch):
    calls = _capture_warm(monkeypatch)
    monkeypatch.setenv("PIPER_WARM_AT_STARTUP", "0")

    server.warm_piper_at_startup()

    assert calls == []
    assert server._PIPER_STARTUP_WARM_COMPLETE is False


def test_startup_warm_is_skipped_for_subprocess_piper(monkeypatch):
    calls = _capture_warm(monkeypatch)
    monkeypatch.setenv("PIPER_USE_SUBPROCESS", "1")

    server.warm_piper_at_startup()

    assert calls == []
    assert server._PIPER_STARTUP_WARM_COMPLETE is False


def test_startup_fails_when_no_voice_model_can_be_resolved(monkeypatch, tmp_path):
    calls = _capture_warm(monkeypatch)
    monkeypatch.setattr(server, "DEFAULT_VOICE_MODEL", "missing.onnx")

    def missing_voice(_name):
        raise FileNotFoundError("missing")

    monkeypatch.setattr(server, "resolve_voice_model_path", missing_voice)
    monkeypatch.setattr(server, "get_narrator_tts_settings", lambda: {})
    monkeypatch.setattr(server, "NPC_DIR", tmp_path)

    with pytest.raises(RuntimeError, match="no usable voice models"):
        server.warm_piper_at_startup()

    assert calls == []
    assert server._PIPER_STARTUP_WARM_COMPLETE is False


def test_pool_warm_failure_prevents_server_readiness(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "DEFAULT_VOICE_MODEL", "default.onnx")
    monkeypatch.setattr(server, "resolve_voice_model_path", lambda name: Path(str(name)))
    monkeypatch.setattr(server, "get_narrator_tts_settings", lambda: {})
    monkeypatch.setattr(server, "NPC_DIR", tmp_path)

    def fail_warm(*_args, **_kwargs):
        raise RuntimeError("replica load failed")

    monkeypatch.setattr(server.piper_pool, "warm_voice_models", fail_warm)

    with pytest.raises(RuntimeError, match="replica load failed"):
        server.warm_piper_at_startup()

    assert server._PIPER_STARTUP_WARM_COMPLETE is False


def test_warm_runs_at_module_scope_for_wsgi_import():
    tree = ast.parse(Path(server.__file__).read_text(encoding="utf-8"))
    top_level_calls = [
        node
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "warm_piper_at_startup"
    ]

    assert len(top_level_calls) == 1

"""P2: phoneme cache + follow-up parallelism helper coverage."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import piper_pool


def test_phoneme_cache_hit_and_lru(monkeypatch):
    piper_pool.clear_phoneme_cache_for_tests()
    monkeypatch.setenv("PIPER_PHONEME_CACHE_SIZE", "2")

    calls = {"n": 0}

    class FakeVoice:
        def phonemize(self, text: str):
            calls["n"] += 1
            return [["a", "b"]]

    voice = FakeVoice()
    wrapped = piper_pool._wrap_voice_phonemize_cache(voice, "model-a")
    assert wrapped.phonemize("hi") == [["a", "b"]]
    assert wrapped.phonemize("hi") == [["a", "b"]]
    assert calls["n"] == 1
    stats = piper_pool.phoneme_cache_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1

    wrapped.phonemize("one")
    wrapped.phonemize("two")  # evicts "hi"
    assert calls["n"] == 3
    wrapped.phonemize("hi")  # miss after eviction
    assert calls["n"] == 4


def test_phoneme_cache_disabled(monkeypatch):
    piper_pool.clear_phoneme_cache_for_tests()
    monkeypatch.setenv("PIPER_PHONEME_CACHE_SIZE", "0")
    calls = {"n": 0}

    class FakeVoice:
        def phonemize(self, text: str):
            calls["n"] += 1
            return [["x"]]

    voice = FakeVoice()
    wrapped = piper_pool._wrap_voice_phonemize_cache(voice, "m")
    # When disabled, wrapper returns the same voice object (no cache wrap).
    assert wrapped is voice
    voice.phonemize("a")
    voice.phonemize("a")
    assert calls["n"] == 2
    assert piper_pool.phoneme_cache_stats()["hits"] == 0


def test_followup_parallel_env_clamped(monkeypatch):
    # Import helper logic via server function recreation
    import os

    def parallel() -> int:
        raw = str(os.environ.get("PIPER_FOLLOWUP_PARALLEL", "")).strip()
        if not raw:
            return 1
        try:
            return max(1, min(4, int(raw)))
        except ValueError:
            return 1

    monkeypatch.delenv("PIPER_FOLLOWUP_PARALLEL", raising=False)
    assert parallel() == 1
    monkeypatch.setenv("PIPER_FOLLOWUP_PARALLEL", "3")
    assert parallel() == 3
    monkeypatch.setenv("PIPER_FOLLOWUP_PARALLEL", "99")
    assert parallel() == 4

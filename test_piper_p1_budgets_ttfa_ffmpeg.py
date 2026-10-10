"""P1: explicit model slot budgets + TTFA-fast FFmpeg filter."""

from __future__ import annotations

from pathlib import Path

import piper_pool
from tts_processing import build_af_filter, ttfa_fast_ffmpeg_enabled


def test_parse_model_slot_budgets():
    got = piper_pool.parse_model_slot_budgets("en_US-libritts-high:8, lessac:2")
    assert got == {"en_us-libritts-high": 8, "lessac": 2}


def test_explicit_budgets_override_speaker_weights():
    weights = {
        str(Path("C:/Voices/en_US-libritts-high.onnx")): 904,
        str(Path("C:/Voices/en_US-lessac-medium.onnx")): 1,
    }
    # Without explicit budgets, libritts would dominate.
    proportional = piper_pool.allocate_replicas(weights, 10)
    assert proportional[str(Path("C:/Voices/en_US-libritts-high.onnx"))] >= 8

    import os

    os.environ["PIPER_MODEL_SLOT_BUDGETS"] = "libritts:8,lessac:2"
    try:
        got = piper_pool.allocate_replicas(weights, 10)
    finally:
        os.environ.pop("PIPER_MODEL_SLOT_BUDGETS", None)
    assert got[str(Path("C:/Voices/en_US-libritts-high.onnx"))] == 8
    assert got[str(Path("C:/Voices/en_US-lessac-medium.onnx"))] == 2


def test_ttfa_fast_filter_skips_heavy_stages():
    full = build_af_filter(ttfa_fast=False)
    fast = build_af_filter(ttfa_fast=True)
    assert "afftdn" in full
    assert "dynaudnorm" in full
    assert "afftdn" not in fast
    assert "dynaudnorm" not in fast
    assert "alimiter" in fast
    assert "highpass" in fast


def test_ttfa_fast_env(monkeypatch):
    monkeypatch.delenv("PIPER_TTFA_FAST_FFMPEG", raising=False)
    assert ttfa_fast_ffmpeg_enabled() is True
    monkeypatch.setenv("PIPER_TTFA_FAST_FFMPEG", "0")
    assert ttfa_fast_ffmpeg_enabled() is False

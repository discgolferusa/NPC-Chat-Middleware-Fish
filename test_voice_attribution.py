"""Unit + functional tests for MODEL_CARD voice attribution."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("flask")

from voice_attribution import build_voice_attributions, try_read_model_card


def test_try_read_model_card_prefers_named_file(tmp_path: Path):
    (tmp_path / "MODEL_CARD.md").write_text("FROM MD", encoding="utf-8")
    assert try_read_model_card(tmp_path) == "FROM MD"


def test_build_voice_attributions_local_then_root_fallback(tmp_path: Path):
    nested = tmp_path / "en" / "hero"
    nested.mkdir(parents=True)
    (nested / "hero.onnx").write_bytes(b"onnx")
    (nested / "MODEL_CARD").write_text("HERO CARD", encoding="utf-8")
    (tmp_path / "MODEL_CARD").write_text("ROOT CARD", encoding="utf-8")

    other = tmp_path / "en" / "villain"
    other.mkdir(parents=True)
    (other / "villain.onnx").write_bytes(b"onnx")

    rows = build_voice_attributions(tmp_path)
    assert len(rows) == 2
    by_id = {r["voice_id"]: r for r in rows}
    assert by_id["en/hero/hero.onnx"]["attribution_text"] == "HERO CARD"
    assert by_id["en/hero/hero.onnx"]["used_root_fallback"] is False
    assert by_id["en/hero/hero.onnx"]["name"] == "hero.onnx"
    assert by_id["en/villain/villain.onnx"]["attribution_text"] == "ROOT CARD"
    assert by_id["en/villain/villain.onnx"]["used_root_fallback"] is True


def test_build_voice_attributions_missing_dir_returns_empty(tmp_path: Path):
    assert build_voice_attributions(tmp_path / "missing") == []


def test_voices_attribution_endpoint(api, tmp_path, monkeypatch):
    client, server, headers = api

    root = tmp_path / "Voices"
    nested = root / "en" / "hero"
    nested.mkdir(parents=True)
    (nested / "hero.onnx").write_bytes(b"onnx")
    (nested / "MODEL_CARD").write_text("HERO CARD", encoding="utf-8")
    (root / "MODEL_CARD").write_text("ROOT CARD", encoding="utf-8")
    other = root / "en" / "villain"
    other.mkdir(parents=True)
    (other / "villain.onnx").write_bytes(b"onnx")

    monkeypatch.setattr(server, "VOICES_DIR", root)

    def _inventory(*, force: bool = False):
        # Mirror inventory shape used by /voices (path relative to root).
        return [
            {"path": "en/hero/hero.onnx", "label": "en/hero/hero.onnx"},
            {"path": "en/villain/villain.onnx", "label": "en/villain/villain.onnx"},
        ]

    monkeypatch.setattr(server, "refresh_voice_inventory", _inventory)

    res = client.get("/voices/attribution", headers=headers)
    assert res.status_code == 200
    body = res.get_json()
    voices = body.get("voices") or []
    assert len(voices) == 2
    by_id = {v["voice_id"]: v for v in voices}
    assert by_id["en/hero/hero.onnx"]["attribution_text"] == "HERO CARD"
    assert by_id["en/villain/villain.onnx"]["attribution_text"] == "ROOT CARD"
    assert by_id["en/villain/villain.onnx"]["used_root_fallback"] is True
    assert body.get("voices_dir") == str(root)

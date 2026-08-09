"""Unit + functional tests for MODEL_CARD voice attribution."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("flask")

from voice_attribution import (
    _card_search_dirs,
    build_voice_attributions,
    try_read_model_card,
)
import voice_attribution


def test_try_read_model_card_prefers_named_file(tmp_path: Path):
    (tmp_path / "MODEL_CARD.md").write_text("FROM MD", encoding="utf-8")
    assert try_read_model_card(tmp_path) == "FROM MD"


def test_card_search_dirs_uses_logical_parents_not_resolve(tmp_path: Path):
    """Walk must stay under voices_dir even when resolve() would escape via junction."""
    licensed = tmp_path / "licensed"
    nested = licensed / "en" / "hero"
    nested.mkdir(parents=True)

    dirs = _card_search_dirs(nested, tmp_path)
    assert dirs == [
        nested,
        licensed / "en",
        licensed,
        tmp_path,
    ]
    for directory in dirs:
        directory.relative_to(tmp_path)


def test_build_voice_attributions_caches_parent_card_reads(tmp_path: Path, monkeypatch):
    licensed = tmp_path / "licensed"
    for name in ("a", "b", "c"):
        folder = licensed / name
        folder.mkdir(parents=True)
        (folder / f"{name}.onnx").write_bytes(b"onnx")
    (licensed / "MODEL_CARD").write_text("PACK CARD", encoding="utf-8")

    reads = {"n": 0}
    real = try_read_model_card

    def counting(directory: Path):
        reads["n"] += 1
        return real(directory)

    monkeypatch.setattr(voice_attribution, "try_read_model_card", counting)
    rows = build_voice_attributions(tmp_path)
    assert len(rows) == 3
    assert all(r["attribution_text"] == "PACK CARD" for r in rows)
    # Unique dirs checked: a, b, c, licensed (root never reached once pack card hits).
    assert reads["n"] == 4


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


def test_build_voice_attributions_uses_licensed_parent_model_card(tmp_path: Path):
    """MODEL_CARD in an intermediate pack folder (e.g. licensed/) applies to nested onnx."""
    licensed = tmp_path / "licensed"
    nested = licensed / "en" / "en_US" / "ryan" / "medium"
    nested.mkdir(parents=True)
    (nested / "en_US-ryan-medium.onnx").write_bytes(b"onnx")
    (licensed / "MODEL_CARD").write_text("LICENSED PACK CARD", encoding="utf-8")

    rows = build_voice_attributions(tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["attribution_text"] == "LICENSED PACK CARD"
    assert row["used_root_fallback"] is True
    assert "No MODEL_CARD" not in row["attribution_text"]


def test_build_voice_attributions_prefers_closer_parent_over_root(tmp_path: Path):
    licensed = tmp_path / "licensed"
    nested = licensed / "en" / "hero"
    nested.mkdir(parents=True)
    (nested / "hero.onnx").write_bytes(b"onnx")
    (licensed / "MODEL_CARD").write_text("PACK CARD", encoding="utf-8")
    (tmp_path / "MODEL_CARD").write_text("ROOT CARD", encoding="utf-8")

    rows = build_voice_attributions(tmp_path)
    assert rows[0]["attribution_text"] == "PACK CARD"
    assert rows[0]["used_root_fallback"] is True


def test_build_voice_attributions_placeholder_does_not_set_root_fallback(tmp_path: Path):
    nested = tmp_path / "en" / "orphan"
    nested.mkdir(parents=True)
    (nested / "orphan.onnx").write_bytes(b"onnx")

    rows = build_voice_attributions(tmp_path)
    assert len(rows) == 1
    assert rows[0]["used_root_fallback"] is False
    assert "No MODEL_CARD" in rows[0]["attribution_text"]


def test_try_read_model_card_skips_invalid_utf8(tmp_path: Path):
    (tmp_path / "MODEL_CARD").write_bytes(b"\xff\xfe bad")
    (tmp_path / "MODEL_CARD.md").write_text("GOOD", encoding="utf-8")
    assert try_read_model_card(tmp_path) == "GOOD"


def test_build_voice_attributions_invalid_utf8_card_falls_through(tmp_path: Path):
    nested = tmp_path / "en" / "hero"
    nested.mkdir(parents=True)
    (nested / "hero.onnx").write_bytes(b"onnx")
    (nested / "MODEL_CARD").write_bytes(b"\xff\xfe")
    (tmp_path / "MODEL_CARD").write_text("ROOT OK", encoding="utf-8")

    rows = build_voice_attributions(tmp_path)
    assert len(rows) == 1
    assert rows[0]["attribution_text"] == "ROOT OK"
    assert rows[0]["used_root_fallback"] is True


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

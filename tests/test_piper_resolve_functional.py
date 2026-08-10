"""Functional coverage for Piper resolution wired through the Flask app module."""

from __future__ import annotations

from pathlib import Path


def test_health_still_ok_when_piper_missing(api, monkeypatch, tmp_path):
    """App should start and serve /health even if Piper is not installed."""
    client, server, _headers = api

    def _missing(**_kwargs):
        raise FileNotFoundError("no piper")

    monkeypatch.setattr(server, "resolve_piper_exe", _missing)
    monkeypatch.setattr(server, "PIPER_EXE", Path(tmp_path / "missing-piper"))

    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json() is not None

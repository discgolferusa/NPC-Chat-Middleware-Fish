"""Functional auth tests for Flask game API key gate."""

from __future__ import annotations

import importlib

import pytest

pytest.importorskip("flask")


@pytest.fixture()
def auth_client(monkeypatch):
    monkeypatch.setenv("GAME_API_KEY", "test-secret-key")
    monkeypatch.setenv("REQUIRE_GAME_API_KEY", "false")
    monkeypatch.delenv("VOICES_DIR", raising=False)

    import server

    importlib.reload(server)
    # game_api.env may override env at import time — pin the key used by the gate.
    server.GAME_API_KEY = "test-secret-key"
    server.REQUIRE_GAME_API_KEY = False
    server.app.config["TESTING"] = True
    with server.app.test_client() as client:
        yield client, server


def test_health_is_open_without_key(auth_client):
    client, _ = auth_client
    response = client.get("/health")
    assert response.status_code == 200


def test_protected_route_rejects_missing_key(auth_client):
    client, _ = auth_client
    response = client.get("/campaigns")
    assert response.status_code == 401
    assert response.get_json()["error"] == "Unauthorized"


def test_protected_route_accepts_bearer(auth_client):
    client, _ = auth_client
    response = client.get(
        "/campaigns",
        headers={"Authorization": "Bearer test-secret-key"},
    )
    assert response.status_code != 401


def test_protected_route_accepts_x_game_api_key(auth_client):
    client, _ = auth_client
    response = client.get(
        "/campaigns",
        headers={"X-Game-Api-Key": "test-secret-key"},
    )
    assert response.status_code != 401


def test_api_keys_match_helper(auth_client):
    _, server = auth_client
    assert server._api_keys_match("test-secret-key", "test-secret-key")
    assert not server._api_keys_match("wrong", "test-secret-key")

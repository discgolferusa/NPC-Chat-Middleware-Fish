"""Groq catalog freshness, purpose gating, and retired-id remapping.

The catalog in llm_router is the only source for the Admin -> LLM dropdowns, so a
stale entry here silently sends every NPC turn at a model Groq has shut down.
"""

from __future__ import annotations

import pytest

import llm_router

# Groq model ids reachable on the Free plan as of 2026-08-21
# (console.groq.com/docs/models + /docs/rate-limits). Speech-to-text, text-to-speech,
# prompt-guard classifiers and enterprise-only models are intentionally absent.
CURRENT_GROQ_FREE_PLAN_MODELS = frozenset(
    {
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-safeguard-20b",
        "qwen/qwen3.6-27b",
        "groq/compound",
        "groq/compound-mini",
    }
)

# Groq shutdown dates: llama3 2025-08-30, kimi-k2 2026-04-15, qwen3-32b and
# llama-4-scout 2026-07-17, llama-3.1-8b / llama-3.3-70b 2026-08-16.
SHUT_DOWN_GROQ_MODELS = (
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "qwen/qwen3-32b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "moonshotai/kimi-k2-instruct-0905",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
)


def _groq_entries() -> list[dict]:
    return [e for e in llm_router.LLM_MODELS.values() if e.get("provider") == "groq"]


def test_every_groq_entry_is_on_the_current_free_plan():
    for entry in _groq_entries():
        assert entry["model"] in CURRENT_GROQ_FREE_PLAN_MODELS, (
            f"{entry['id']} points at {entry['model']!r}, which Groq no longer serves"
        )


def test_catalog_has_no_shut_down_groq_models():
    served = {entry["model"] for entry in _groq_entries()}
    for retired in SHUT_DOWN_GROQ_MODELS:
        assert retired not in served


def test_groq_default_model_env_is_current():
    assert llm_router.GROQ_MODEL in CURRENT_GROQ_FREE_PLAN_MODELS


def test_every_groq_entry_declares_free_plan_limits():
    for entry in _groq_entries():
        limits = entry.get("limits")
        assert limits, f"{entry['id']} has no limits for preflight"
        for key in ("rpm", "rpd", "tpm"):
            assert isinstance(limits.get(key), int), f"{entry['id']} missing {key}"


def test_every_catalog_id_has_a_pricing_row():
    for model_id in llm_router.LLM_MODELS:
        assert model_id in llm_router.GROQ_ON_DEMAND_PRICES_USD


def test_pricing_table_has_no_orphan_rows():
    for model_id in llm_router.GROQ_ON_DEMAND_PRICES_USD:
        assert model_id in llm_router.LLM_MODELS


@pytest.mark.parametrize(
    ("retired_id", "replacement_id"),
    [
        ("groq-llama-3.1-8b-instant", "groq-gpt-oss-20b"),
        ("groq-llama-3.3-70b", "groq-gpt-oss-120b"),
        ("groq-qwen3-32b", "groq-qwen3.6-27b"),
        # Chains through the now-retired llama-3.3-70b alias.
        ("groq-llama-4-scout", "groq-gpt-oss-120b"),
    ],
)
def test_retired_ids_alias_onto_live_models(retired_id, replacement_id):
    assert llm_router._alias_model_id(retired_id) == replacement_id
    assert llm_router._resolve_model_id(retired_id) == replacement_id


def test_unknown_model_id_falls_back_to_local():
    assert llm_router._resolve_model_id("groq-does-not-exist") == "local"


def test_compound_is_offered_for_content_creation_only():
    chat_models = llm_router.models_for_purpose("chat")
    creation_models = llm_router.models_for_purpose("content_creation")

    assert "groq-compound" not in chat_models
    assert "groq-compound-mini" not in chat_models
    assert "groq-compound" in creation_models
    assert "groq-compound-mini" in creation_models
    # Plain chat models stay available to both dropdowns.
    for shared in ("local", "groq-gpt-oss-120b", "groq-gpt-oss-20b", "groq-qwen3.6-27b"):
        assert shared in chat_models
        assert shared in creation_models


def test_compound_has_the_largest_free_tpm_budget():
    """Import guidance points at groq-compound, so it must really be the roomiest."""
    tpm = {e["id"]: (e["limits"] or {}).get("tpm") or 0 for e in _groq_entries()}
    assert tpm["groq-compound"] == max(tpm.values())


def test_resolve_model_id_coerces_a_chat_selection_compound_cannot_serve():
    resolved = llm_router._resolve_model_id("groq-compound", default="local", purpose="chat")
    assert resolved == "local"
    assert (
        llm_router._resolve_model_id("groq-compound", default="local", purpose="content_creation")
        == "groq-compound"
    )


def test_zero_budget_advice_recommends_a_model_still_in_the_catalog(monkeypatch):
    monkeypatch.setattr(llm_router, "CONTENT_CREATION_MODEL_ID", "groq-gpt-oss-20b")
    monkeypatch.setattr(llm_router, "CONTENT_CREATION_PROVIDER_MODE", "groq")

    message = llm_router.content_creation_zero_budget_error_message()
    assert "groq-compound" in message
    assert "groq-llama" not in message


def test_tpm_breach_advice_recommends_a_model_still_in_the_catalog(monkeypatch):
    monkeypatch.setattr(llm_router, "usage_requests_last_60s", lambda **kwargs: 0)
    monkeypatch.setattr(llm_router, "usage_requests_last_24h", lambda **kwargs: 0)
    monkeypatch.setattr(llm_router, "usage_tokens_last_60s", lambda **kwargs: 0)

    reason = llm_router._would_exceed_groq_limits(
        "groq-gpt-oss-20b",
        999_999,
        purpose="content_creation",
    )
    assert reason is not None
    assert "groq-compound" in reason
    assert "groq-llama" not in reason


@pytest.fixture()
def isolated_selection(monkeypatch, tmp_path):
    """Keep select_llm_model off the developer's llm_runtime_config.json."""
    monkeypatch.setattr(
        llm_router, "LLM_RUNTIME_CONFIG_FILE", tmp_path / "llm_runtime_config.json"
    )
    monkeypatch.setattr(llm_router, "CURRENT_LLM_MODEL_ID", "local")
    monkeypatch.setattr(llm_router, "LLM_PROVIDER_MODE", "local")
    monkeypatch.setattr(llm_router, "CONTENT_CREATION_MODEL_ID", "local")
    monkeypatch.setattr(llm_router, "CONTENT_CREATION_PROVIDER_MODE", "local")
    return tmp_path


def test_select_rejects_compound_for_chat(isolated_selection):
    with pytest.raises(ValueError, match="not available for purpose chat"):
        llm_router.select_llm_model("groq-compound", provider_mode="groq", purpose="chat")
    assert llm_router.CURRENT_LLM_MODEL_ID == "local"


def test_select_accepts_compound_for_content_creation(isolated_selection):
    result = llm_router.select_llm_model(
        "groq-compound", provider_mode="groq", purpose="content_creation"
    )
    assert result["model_id"] == "groq-compound"
    assert result["selected_model"]["model"] == "groq/compound"


def test_select_remaps_a_retired_id(isolated_selection):
    result = llm_router.select_llm_model("groq-llama-3.3-70b", provider_mode="groq", purpose="chat")
    assert result["model_id"] == "groq-gpt-oss-120b"


def test_models_payload_exposes_purpose_filtered_lists(isolated_selection):
    payload = llm_router.get_llm_models_payload()

    chat_ids = [m["id"] for m in payload["purposes"]["chat"]["models"]]
    creation_ids = [m["id"] for m in payload["purposes"]["content_creation"]["models"]]

    assert "groq-compound" not in chat_ids
    assert "groq-compound" in creation_ids
    # Every entry advertises its purposes so older clients can filter the flat list.
    for model in payload["models"]:
        assert model["purposes"]


# --- functional: the HTTP surface Admin -> LLM actually calls ---


def test_llm_models_endpoint_returns_purpose_filtered_catalog(api):
    client, _server, headers = api

    response = client.get("/llm/models", headers=headers)
    assert response.status_code == 200
    payload = response.get_json()

    served = {m["model"] for m in payload["models"]}
    for retired in SHUT_DOWN_GROQ_MODELS:
        assert retired not in served

    chat_ids = [m["id"] for m in payload["purposes"]["chat"]["models"]]
    creation_ids = [m["id"] for m in payload["purposes"]["content_creation"]["models"]]
    assert "groq-compound" not in chat_ids
    assert "groq-compound" in creation_ids
    assert "groq-gpt-oss-120b" in chat_ids


def test_llm_select_endpoint_rejects_compound_for_chat(api, isolated_selection):
    client, _server, headers = api

    response = client.post(
        "/llm/select",
        headers=headers,
        json={"model_id": "groq-compound", "provider_mode": "groq", "purpose": "chat"},
    )
    assert response.status_code == 400


def test_llm_select_endpoint_accepts_compound_for_content_creation(api, isolated_selection):
    client, _server, headers = api

    response = client.post(
        "/llm/select",
        headers=headers,
        json={
            "model_id": "groq-compound",
            "provider_mode": "groq",
            "purpose": "content_creation",
        },
    )
    assert response.status_code == 200
    assert response.get_json()["model_id"] == "groq-compound"


def test_llm_select_endpoint_remaps_retired_id(api, isolated_selection):
    client, _server, headers = api

    response = client.post(
        "/llm/select",
        headers=headers,
        json={"model_id": "groq-llama-3.1-8b-instant", "provider_mode": "groq", "purpose": "chat"},
    )
    assert response.status_code == 200
    assert response.get_json()["model_id"] == "groq-gpt-oss-20b"

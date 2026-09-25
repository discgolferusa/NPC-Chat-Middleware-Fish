"""Tests for Admin-editable LLM catalog, precedence order, and OpenRouter pricing."""

from __future__ import annotations

import llm_catalog
import llm_router
import pytest


@pytest.fixture()
def isolated_selection(monkeypatch, tmp_path):
    catalog_path = tmp_path / "llm_catalog.json"
    monkeypatch.setattr(
        llm_router, "LLM_RUNTIME_CONFIG_FILE", tmp_path / "llm_runtime_config.json"
    )
    monkeypatch.setattr(llm_router, "LLM_CATALOG_FILE", catalog_path)
    monkeypatch.setattr(llm_catalog, "LLM_CATALOG_FILE", catalog_path)
    llm_catalog.load_or_seed(
        llm_models=llm_router.LLM_MODELS,
        prices=llm_router.ONLINE_ON_DEMAND_PRICES_USD,
        default_prices={
            "local": {"input": 0.0, "output": 0.0},
            "openrouter-gpt-oss-20b": {"input": 0.03, "output": 0.13},
            "openrouter-nemotron-3.5-lightning-coreweave": {"input": 0.10, "output": 0.25},
            "openrouter-nemotron-3.5-lightning-deepinfra": {"input": 0.08, "output": 0.20},
            "openrouter-deepseek-v4-flash-0731-coreweave": {"input": 0.13, "output": 0.28},
        },
        local_model=llm_router.LOCAL_LLM_MODEL,
        local_url=llm_router.LOCAL_LLM_URL,
    )
    monkeypatch.setattr(llm_router, "CURRENT_LLM_MODEL_ID", "local")
    monkeypatch.setattr(llm_router, "LLM_PROVIDER_MODE", "local")
    monkeypatch.setattr(llm_router, "CHAT_FALLBACK_MODEL_IDS", [])
    monkeypatch.setattr(llm_router, "CONTENT_CREATION_MODEL_ID", "local")
    monkeypatch.setattr(llm_router, "CONTENT_CREATION_PROVIDER_MODE", "local")
    monkeypatch.setattr(llm_router, "CONTENT_CREATION_FALLBACK_MODEL_IDS", [])
    return tmp_path


def test_make_catalog_id_includes_provider():
    assert (
        llm_catalog.make_catalog_id("nvidia/nemotron-3.5-lightning", "coreweave")
        == "openrouter-nvidia-nemotron-3-5-lightning-coreweave"
    )


def test_upsert_entry_pins_single_provider(isolated_selection, monkeypatch):
    monkeypatch.setattr(
        llm_catalog,
        "fetch_endpoint_pricing",
        lambda *a, **k: {
            "input": 0.11,
            "output": 0.22,
            "source": "openrouter",
            "fetched_at": "2026-01-01T00:00:00Z",
        },
    )
    entry = llm_router.upsert_llm_catalog_entry(
        {
            "model": "openai/gpt-oss-20b",
            "host_provider": "novita",
            "purposes": ["chat"],
            "enabled": True,
        },
        fetch_pricing=True,
    )
    assert entry["id"] == "openrouter-openai-gpt-oss-20b-novita"
    assert entry["provider_order"] == ["novita"]
    assert entry["host_provider"] == "novita"
    assert entry["purposes"] == ["chat"]
    assert entry["pricing"]["input"] == pytest.approx(0.11)
    assert entry["id"] in llm_router.LLM_MODELS
    assert llm_router.model_supports_purpose(entry["id"], "chat")
    assert not llm_router.model_supports_purpose(entry["id"], "content_creation")


def test_disable_entry_removes_from_purpose_lists(isolated_selection):
    llm_router.upsert_llm_catalog_entry(
        {
            "id": "openrouter-gpt-oss-20b",
            "model": "openai/gpt-oss-20b",
            "host_provider": "amazon-bedrock",
            "enabled": False,
            "purposes": ["chat", "content_creation"],
        },
        fetch_pricing=False,
    )
    assert "openrouter-gpt-oss-20b" not in llm_router.models_for_purpose("chat")
    assert llm_router.LLM_MODELS["openrouter-gpt-oss-20b"]["enabled"] is False


def test_set_model_order_updates_primary_and_fallbacks(isolated_selection):
    result = llm_router.set_llm_model_order(
        "chat",
        [
            "openrouter-nemotron-3.5-lightning-coreweave",
            "openrouter-nemotron-3.5-lightning-deepinfra",
            "local",
        ],
        provider_mode="openrouter_with_local_fallback",
    )
    assert result["model_id"] == "openrouter-nemotron-3.5-lightning-coreweave"
    assert result["fallback_model_ids"] == [
        "openrouter-nemotron-3.5-lightning-deepinfra",
        "local",
    ]
    assert result["model_order"][0] == "openrouter-nemotron-3.5-lightning-coreweave"
    assert llm_router.CURRENT_LLM_MODEL_ID == "openrouter-nemotron-3.5-lightning-coreweave"


def test_cannot_delete_local(isolated_selection):
    with pytest.raises(ValueError, match="local"):
        llm_router.delete_llm_catalog_entry("local")


def test_fetch_endpoint_pricing_matches_provider(monkeypatch):
    class FakeResp:
        status_code = 200
        content = b"{}"

        def json(self):
            return {
                "data": {
                    "endpoints": [
                        {
                            "provider_name": "DeepInfra",
                            "pricing": {"prompt": "0.00000008", "completion": "0.0000002"},
                        },
                        {
                            "provider_name": "CoreWeave",
                            "pricing": {"prompt": "0.0000001", "completion": "0.00000025"},
                        },
                    ]
                }
            }

    monkeypatch.setattr(
        llm_catalog, "_get_openrouter_api_root", lambda: "https://openrouter.ai/api/v1"
    )
    monkeypatch.setattr(llm_catalog, "_get_openrouter_headers", lambda: {})
    monkeypatch.setattr(llm_catalog.requests, "get", lambda *a, **k: FakeResp())

    pricing = llm_catalog.fetch_endpoint_pricing(
        "nvidia/nemotron-3.5-lightning", "coreweave"
    )
    assert pricing is not None
    assert pricing["input"] == pytest.approx(0.1)
    assert pricing["output"] == pytest.approx(0.25)
    assert pricing["source"] == "openrouter"


def test_select_model_order_via_select_llm_model(isolated_selection):
    result = llm_router.select_llm_model(
        "local",
        provider_mode="openrouter_with_local_fallback",
        purpose="content_creation",
        model_order=[
            "openrouter-gpt-oss-20b",
            "local",
        ],
    )
    assert result["model_id"] == "openrouter-gpt-oss-20b"
    assert result["fallback_model_ids"] == ["local"]
    assert result["model_order"][:2] == ["openrouter-gpt-oss-20b", "local"]

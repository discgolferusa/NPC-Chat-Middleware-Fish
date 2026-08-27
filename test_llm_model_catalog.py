"""OpenRouter catalog, credit-threshold fallback, and retired-id remapping.

The catalog in llm_router is the only source for the Admin -> LLM dropdowns.
"""

from __future__ import annotations

import pytest

import llm_router


def _online_entries() -> list[dict]:
    return [e for e in llm_router.LLM_MODELS.values() if e.get("provider") == "openrouter"]


def test_catalog_offers_openrouter_gpt_oss_20b():
    entry = llm_router.LLM_MODELS["openrouter-gpt-oss-20b"]
    assert entry["model"] == "openai/gpt-oss-20b"
    assert entry["online"] is True
    assert "chat" in entry["purposes"]
    assert "content_creation" in entry["purposes"]


@pytest.mark.parametrize(
    ("model_id", "openrouter_model", "label_fragment", "provider_slug"),
    [
        (
            "openrouter-nemotron-3.5-lightning-coreweave",
            "nvidia/nemotron-3.5-lightning",
            "Nemotron 3.5 Lightning",
            "coreweave",
        ),
        (
            "openrouter-nemotron-3.5-lightning-deepinfra",
            "nvidia/nemotron-3.5-lightning",
            "Nemotron 3.5 Lightning",
            "deepinfra",
        ),
        (
            "openrouter-deepseek-v4-flash-0731-coreweave",
            "deepseek/deepseek-v4-flash-0731",
            "DeepSeek V4 Flash 0731",
            "coreweave",
        ),
    ],
)
def test_catalog_offers_provider_pinned_openrouter_models(
    model_id, openrouter_model, label_fragment, provider_slug
):
    entry = llm_router.LLM_MODELS[model_id]
    assert entry["model"] == openrouter_model
    assert entry["provider"] == "openrouter"
    assert entry["provider_order"] == [provider_slug]
    assert entry["online"] is True
    assert label_fragment in entry["label"]
    assert entry["provider_label"]
    assert entry["provider_label"] in entry["label"]
    assert "chat" in entry["purposes"]
    assert "content_creation" in entry["purposes"]


def test_default_openrouter_model_env_matches_catalog():
    assert llm_router.OPENROUTER_MODEL == "openai/gpt-oss-20b"


def test_every_catalog_id_has_a_pricing_row():
    for model_id in llm_router.LLM_MODELS:
        assert model_id in llm_router.ONLINE_ON_DEMAND_PRICES_USD


def test_pricing_table_has_no_orphan_rows():
    for model_id in llm_router.ONLINE_ON_DEMAND_PRICES_USD:
        assert model_id in llm_router.LLM_MODELS


@pytest.mark.parametrize(
    ("retired_id", "replacement_id"),
    [
        ("groq-llama-3.1-8b-instant", "openrouter-gpt-oss-20b"),
        ("groq-llama-3.3-70b", "openrouter-gpt-oss-20b"),
        ("groq-gpt-oss-120b", "openrouter-gpt-oss-20b"),
        ("groq-gpt-oss-20b", "openrouter-gpt-oss-20b"),
        ("groq-compound", "openrouter-gpt-oss-20b"),
        ("groq-qwen3-32b", "openrouter-gpt-oss-20b"),
        ("groq-llama-4-scout", "openrouter-gpt-oss-20b"),
    ],
)
def test_retired_ids_alias_onto_openrouter(retired_id, replacement_id):
    assert llm_router._alias_model_id(retired_id) == replacement_id
    assert llm_router._resolve_model_id(retired_id) == replacement_id


def test_unknown_model_id_falls_back_to_local():
    assert llm_router._resolve_model_id("openrouter-does-not-exist") == "local"


def test_groq_provider_modes_alias_to_openrouter():
    assert llm_router._normalize_provider_mode("groq") == "openrouter"
    assert (
        llm_router._normalize_provider_mode("groq_with_local_fallback")
        == "openrouter_with_local_fallback"
    )


def test_models_for_purpose_include_openrouter_and_local():
    chat_models = llm_router.models_for_purpose("chat")
    creation_models = llm_router.models_for_purpose("content_creation")
    for shared in (
        "local",
        "openrouter-gpt-oss-20b",
        "openrouter-nemotron-3.5-lightning-coreweave",
        "openrouter-nemotron-3.5-lightning-deepinfra",
        "openrouter-deepseek-v4-flash-0731-coreweave",
    ):
        assert shared in chat_models
        assert shared in creation_models



def test_credit_snapshot_from_key_limit():
    snapshot = llm_router._credit_snapshot_from_key_payload(
        {"limit": 10.0, "limit_remaining": 1.5, "usage": 8.5, "is_free_tier": True}
    )
    assert snapshot is not None
    assert snapshot["usage_ratio"] == pytest.approx(0.85)
    assert snapshot["total_credits"] == 10.0
    assert snapshot["total_usage"] == 8.5


def test_credit_snapshot_from_key_usage_and_budget(monkeypatch):
    monkeypatch.setattr(llm_router, "OPENROUTER_CREDIT_BUDGET_USD", 25.0)
    snapshot = llm_router._credit_snapshot_from_key_payload(
        {"limit": None, "limit_remaining": None, "usage": 20.5, "is_free_tier": False}
    )
    assert snapshot is not None
    assert snapshot["source"] == "key+budget"
    assert snapshot["usage_ratio"] == pytest.approx(0.82)
    assert snapshot["total_credits"] == 25.0
    assert snapshot["total_usage"] == 20.5


def test_credit_snapshot_null_limit_without_budget_returns_none(monkeypatch):
    monkeypatch.setattr(llm_router, "OPENROUTER_CREDIT_BUDGET_USD", 0.0)
    assert llm_router._credit_snapshot_from_key_payload(
        {"limit": None, "limit_remaining": None, "usage": 12.0}
    ) is None


def test_fetch_credit_snapshot_negative_cache(monkeypatch):
    calls = {"n": 0}

    def fake_live():
        calls["n"] += 1
        return None

    monkeypatch.setattr(llm_router, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(llm_router, "_fetch_openrouter_credit_snapshot_live", fake_live)
    monkeypatch.setattr(llm_router, "_CREDITS_SNAPSHOT", None)
    monkeypatch.setattr(llm_router, "_CREDITS_LOOKUP_UNTIL", None)

    assert llm_router.fetch_openrouter_credit_snapshot() is None
    assert llm_router.fetch_openrouter_credit_snapshot() is None
    assert calls["n"] == 1


def test_fetch_credit_snapshot_failed_refresh_preserves_good_cache(monkeypatch):
    good = {
        "source": "key+budget",
        "total_credits": 10.0,
        "total_usage": 9.0,
        "usage_ratio": 0.9,
    }
    monkeypatch.setattr(llm_router, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(llm_router, "_CREDITS_SNAPSHOT", dict(good))
    monkeypatch.setattr(llm_router, "_CREDITS_LOOKUP_UNTIL", None)
    monkeypatch.setattr(llm_router, "_fetch_openrouter_credit_snapshot_live", lambda: None)

    result = llm_router.fetch_openrouter_credit_snapshot(force=True)
    assert result == good
    assert llm_router._CREDITS_SNAPSHOT == good


def test_fetch_credit_snapshot_returns_local_copy_under_lock(monkeypatch):
    """Return value must not re-read globals outside the lock."""
    good = {
        "source": "key",
        "total_credits": 10.0,
        "total_usage": 5.0,
        "usage_ratio": 0.5,
    }
    monkeypatch.setattr(llm_router, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(llm_router, "_CREDITS_SNAPSHOT", None)
    monkeypatch.setattr(llm_router, "_CREDITS_LOOKUP_UNTIL", None)

    original_lock = llm_router._CREDITS_LOCK

    class RaceLock:
        def __enter__(self):
            original_lock.__enter__()
            return self

        def __exit__(self, *args):
            return original_lock.__exit__(*args)

    lock = RaceLock()
    monkeypatch.setattr(llm_router, "_CREDITS_LOCK", lock)

    def fake_live():
        return dict(good)

    monkeypatch.setattr(llm_router, "_fetch_openrouter_credit_snapshot_live", fake_live)

    result = llm_router.fetch_openrouter_credit_snapshot(force=True)
    assert result == good
    # Mutating globals after fetch must not change the returned dict.
    llm_router._CREDITS_SNAPSHOT = None
    assert result == good


def test_fetch_credit_snapshot_skips_credits_without_management_key(monkeypatch):
    calls = {"key": 0, "credits": 0}

    class FakeResp:
        status_code = 200
        content = b'{"data":{"limit":null,"usage":1.0}}'

        def json(self):
            return {"data": {"limit": None, "usage": 1.0}}

    def fake_get(url, **kwargs):
        if url.endswith("/key"):
            calls["key"] += 1
            return FakeResp()
        if url.endswith("/credits"):
            calls["credits"] += 1
            raise AssertionError("/credits must not be called without management key")
        raise AssertionError(url)

    monkeypatch.setattr(llm_router, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(llm_router, "OPENROUTER_MANAGEMENT_API_KEY", "")
    monkeypatch.setattr(llm_router, "OPENROUTER_CREDIT_BUDGET_USD", 0.0)
    monkeypatch.setattr(llm_router, "_CREDITS_SNAPSHOT", None)
    monkeypatch.setattr(llm_router, "_CREDITS_LOOKUP_UNTIL", None)
    monkeypatch.setattr(llm_router.requests, "get", fake_get)

    assert llm_router.fetch_openrouter_credit_snapshot(force=True) is None
    assert calls["key"] == 1
    assert calls["credits"] == 0


def test_credits_status_payload_marks_fallback_only_in_fallback_mode(monkeypatch):
    monkeypatch.setattr(
        llm_router,
        "fetch_openrouter_credit_snapshot",
        lambda force=False: {
            "source": "key+budget",
            "total_credits": 10.0,
            "total_usage": 9.0,
            "usage_ratio": 0.9,
        },
    )
    strict = llm_router._credits_status_payload(provider_mode="openrouter")
    assert strict["above_threshold"] is True
    assert strict["fallback_applies"] is False
    assert strict["credit_fallback_active"] is False

    fallback = llm_router._credits_status_payload(provider_mode="openrouter_with_local_fallback")
    assert fallback["above_threshold"] is True
    assert fallback["fallback_applies"] is True
    assert fallback["credit_fallback_active"] is True


def test_credit_snapshot_from_credits_endpoint():
    snapshot = llm_router._credit_snapshot_from_credits_payload(
        {"total_credits": 100.0, "total_usage": 81.0}
    )
    assert snapshot is not None
    assert snapshot["usage_ratio"] == pytest.approx(0.81)


def test_credit_threshold_triggers_fallback_reason(monkeypatch):
    monkeypatch.setattr(llm_router, "OPENROUTER_CREDIT_FALLBACK_RATIO", 0.8)
    monkeypatch.setattr(
        llm_router,
        "fetch_openrouter_credit_snapshot",
        lambda force=False: {
            "source": "key",
            "total_credits": 10.0,
            "total_usage": 8.1,
            "usage_ratio": 0.81,
        },
    )
    reason = llm_router.openrouter_credit_threshold_reason()
    assert reason is not None
    assert "80%" in reason or "credit usage" in reason.casefold()


def test_credit_threshold_below_ratio_is_none(monkeypatch):
    monkeypatch.setattr(llm_router, "OPENROUTER_CREDIT_FALLBACK_RATIO", 0.8)
    monkeypatch.setattr(
        llm_router,
        "fetch_openrouter_credit_snapshot",
        lambda force=False: {
            "source": "key",
            "total_credits": 10.0,
            "total_usage": 5.0,
            "usage_ratio": 0.5,
        },
    )
    assert llm_router.openrouter_credit_threshold_reason() is None


def test_would_exceed_online_limits_uses_credit_threshold(monkeypatch):
    monkeypatch.setattr(
        llm_router,
        "openrouter_credit_threshold_reason",
        lambda: "OpenRouter credit usage is at 85% of available credits (threshold 80%).",
    )
    reason = llm_router._would_exceed_online_limits("openrouter-gpt-oss-20b", 100)
    assert reason is not None
    assert "85%" in reason


@pytest.fixture()
def isolated_selection(monkeypatch, tmp_path):
    monkeypatch.setattr(
        llm_router, "LLM_RUNTIME_CONFIG_FILE", tmp_path / "llm_runtime_config.json"
    )
    monkeypatch.setattr(llm_router, "CURRENT_LLM_MODEL_ID", "local")
    monkeypatch.setattr(llm_router, "LLM_PROVIDER_MODE", "local")
    monkeypatch.setattr(llm_router, "CHAT_FALLBACK_MODEL_IDS", [])
    monkeypatch.setattr(llm_router, "CONTENT_CREATION_MODEL_ID", "local")
    monkeypatch.setattr(llm_router, "CONTENT_CREATION_PROVIDER_MODE", "local")
    monkeypatch.setattr(llm_router, "CONTENT_CREATION_FALLBACK_MODEL_IDS", [])
    return tmp_path


def test_select_accepts_openrouter_model(isolated_selection):
    result = llm_router.select_llm_model(
        "openrouter-gpt-oss-20b",
        provider_mode="openrouter_with_local_fallback",
        purpose="chat",
    )
    assert result["model_id"] == "openrouter-gpt-oss-20b"
    assert result["provider_mode"] == "openrouter_with_local_fallback"


def test_select_persists_fallback_model_ids(isolated_selection):
    result = llm_router.select_llm_model(
        "openrouter-nemotron-3.5-lightning-coreweave",
        provider_mode="openrouter_with_local_fallback",
        purpose="chat",
        fallback_model_ids=[
            "openrouter-nemotron-3.5-lightning-deepinfra",
            "openrouter-deepseek-v4-flash-0731-coreweave",
            "local",
        ],
    )
    assert result["fallback_model_ids"] == [
        "openrouter-nemotron-3.5-lightning-deepinfra",
        "openrouter-deepseek-v4-flash-0731-coreweave",
        "local",
    ]
    assert llm_router.CHAT_FALLBACK_MODEL_IDS == result["fallback_model_ids"]


def test_select_fallback_ids_exclude_primary(isolated_selection):
    result = llm_router.select_llm_model(
        "openrouter-gpt-oss-20b",
        provider_mode="openrouter_with_local_fallback",
        purpose="chat",
        fallback_model_ids=["openrouter-gpt-oss-20b", "local"],
    )
    assert result["fallback_model_ids"] == ["local"]


def test_select_accepts_legacy_groq_mode_alias(isolated_selection):
    result = llm_router.select_llm_model(
        "openrouter-gpt-oss-20b",
        provider_mode="groq_with_local_fallback",
        purpose="chat",
    )
    assert result["provider_mode"] == "openrouter_with_local_fallback"


def test_select_remaps_a_retired_id(isolated_selection):
    result = llm_router.select_llm_model(
        "groq-llama-3.3-70b", provider_mode="openrouter", purpose="chat"
    )
    assert result["model_id"] == "openrouter-gpt-oss-20b"


def test_models_payload_exposes_openrouter_modes(isolated_selection):
    payload = llm_router.get_llm_models_payload()
    mode_ids = {m["id"] for m in payload["provider_modes"]}
    assert mode_ids == {"local", "openrouter", "openrouter_with_local_fallback"}
    chat_ids = [m["id"] for m in payload["purposes"]["chat"]["models"]]
    assert "openrouter-gpt-oss-20b" in chat_ids
    assert "groq-compound" not in chat_ids


def test_generate_falls_back_to_local_when_credits_exceed_threshold(monkeypatch):
    calls = {"local": 0}

    monkeypatch.setattr(llm_router, "CURRENT_LLM_MODEL_ID", "openrouter-gpt-oss-20b")
    monkeypatch.setattr(llm_router, "LLM_PROVIDER_MODE", "openrouter_with_local_fallback")
    monkeypatch.setattr(llm_router, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(llm_router, "ONLINE_DISABLED_UNTIL", None)
    monkeypatch.setattr(
        llm_router,
        "openrouter_credit_threshold_reason",
        lambda: "OpenRouter credit usage is at 90% of available credits (threshold 80%).",
    )

    def fake_local(messages, **kwargs):
        calls["local"] += 1
        return "local-ok", {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}

    monkeypatch.setattr(llm_router, "call_local_chat_completions", fake_local)
    monkeypatch.setattr(llm_router, "record_llm_usage", lambda **kwargs: None)

    result = llm_router.generate_llm_response([{"role": "user", "content": "hi"}])
    assert result == "local-ok"
    assert calls["local"] == 1
    assert llm_router.LAST_FALLBACK_REASON["chat"] == "openrouter_credit_threshold"


def test_generate_strict_openrouter_raises_on_credit_threshold(monkeypatch):
    monkeypatch.setattr(llm_router, "CURRENT_LLM_MODEL_ID", "openrouter-gpt-oss-20b")
    monkeypatch.setattr(llm_router, "LLM_PROVIDER_MODE", "openrouter")
    monkeypatch.setattr(llm_router, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(llm_router, "ONLINE_DISABLED_UNTIL", None)
    monkeypatch.setattr(
        llm_router,
        "openrouter_credit_threshold_reason",
        lambda: "OpenRouter credit usage is at 90% of available credits (threshold 80%).",
    )

    with pytest.raises(llm_router.LLMProviderError, match="credit usage"):
        llm_router.generate_llm_response([{"role": "user", "content": "hi"}])


# --- functional: the HTTP surface Admin -> LLM actually calls ---


def test_llm_models_endpoint_returns_openrouter_catalog(api):
    client, _server, headers = api

    response = client.get("/llm/models", headers=headers)
    assert response.status_code == 200
    payload = response.get_json()

    served = {m["id"] for m in payload["models"]}
    assert "openrouter-gpt-oss-20b" in served
    assert "openrouter-nemotron-3.5-lightning-coreweave" in served
    assert "openrouter-nemotron-3.5-lightning-deepinfra" in served
    assert "openrouter-deepseek-v4-flash-0731-coreweave" in served
    assert "local" in served
    assert "groq-compound" not in served

    by_id = {m["id"]: m for m in payload["models"]}
    assert by_id["openrouter-nemotron-3.5-lightning-coreweave"]["provider_order"] == ["coreweave"]
    assert by_id["openrouter-nemotron-3.5-lightning-deepinfra"]["provider_order"] == ["deepinfra"]
    assert by_id["openrouter-deepseek-v4-flash-0731-coreweave"]["provider_order"] == ["coreweave"]
    assert "—" in by_id["openrouter-gpt-oss-20b"]["label"]

    for purpose in ("chat", "content_creation"):
        purpose_ids = {m["id"] for m in payload["purposes"][purpose]["models"]}
        assert "openrouter-nemotron-3.5-lightning-coreweave" in purpose_ids
        assert "openrouter-nemotron-3.5-lightning-deepinfra" in purpose_ids
        assert "openrouter-deepseek-v4-flash-0731-coreweave" in purpose_ids
        assert "fallback_model_ids" in payload["purposes"][purpose]
    mode_ids = {m["id"] for m in payload["provider_modes"]}
    assert "openrouter_with_local_fallback" in mode_ids
    assert "groq" not in mode_ids


@pytest.mark.parametrize(
    "model_id",
    [
        "openrouter-nemotron-3.5-lightning-coreweave",
        "openrouter-nemotron-3.5-lightning-deepinfra",
        "openrouter-deepseek-v4-flash-0731-coreweave",
    ],
)
@pytest.mark.parametrize("purpose", ["chat", "content_creation"])
def test_llm_select_endpoint_accepts_coreweave_models(api, isolated_selection, model_id, purpose):
    client, _server, headers = api

    response = client.post(
        "/llm/select",
        headers=headers,
        json={
            "model_id": model_id,
            "provider_mode": "openrouter_with_local_fallback",
            "purpose": purpose,
            "fallback_model_ids": ["local"],
        },
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["model_id"] == model_id
    assert body["fallback_model_ids"] == ["local"]


def test_llm_select_endpoint_accepts_openrouter(api, isolated_selection):
    client, _server, headers = api

    response = client.post(
        "/llm/select",
        headers=headers,
        json={
            "model_id": "openrouter-gpt-oss-20b",
            "provider_mode": "openrouter_with_local_fallback",
            "purpose": "chat",
        },
    )
    assert response.status_code == 200
    assert response.get_json()["model_id"] == "openrouter-gpt-oss-20b"


def test_llm_select_endpoint_remaps_retired_id(api, isolated_selection):
    client, _server, headers = api

    response = client.post(
        "/llm/select",
        headers=headers,
        json={
            "model_id": "groq-llama-3.1-8b-instant",
            "provider_mode": "openrouter",
            "purpose": "chat",
        },
    )
    assert response.status_code == 200
    assert response.get_json()["model_id"] == "openrouter-gpt-oss-20b"


def test_llm_status_includes_credits_block(api, monkeypatch):
    client, _server, headers = api
    monkeypatch.setattr(
        llm_router,
        "fetch_openrouter_credit_snapshot",
        lambda force=False: {
            "source": "key",
            "total_credits": 10.0,
            "total_usage": 2.0,
            "usage_ratio": 0.2,
            "limit": 10.0,
            "limit_remaining": 8.0,
        },
    )

    response = client.get("/llm/status", headers=headers)
    assert response.status_code == 200
    payload = response.get_json()
    credits = payload["purposes"]["chat"]["credits"]
    assert credits["fallback_threshold"] == pytest.approx(llm_router.OPENROUTER_CREDIT_FALLBACK_RATIO)
    assert credits["above_threshold"] is False
    assert credits["credit_fallback_active"] is False
    assert credits["fallback_applies"] is True
    assert credits["usage_ratio"] == pytest.approx(0.2)


def test_openrouter_provider_order_parsed_from_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_PROVIDER_ORDER", "amazon-bedrock, novita")
    assert llm_router._openrouter_provider_order() == ["amazon-bedrock", "novita"]


def test_openrouter_provider_order_defaults_to_bedrock_then_novita(monkeypatch):
    monkeypatch.delenv("OPENROUTER_PROVIDER_ORDER", raising=False)
    monkeypatch.setattr(llm_router, "OPENROUTER_PROVIDER_ORDER_RAW", "amazon-bedrock,novita")
    assert llm_router._openrouter_provider_order() == ["amazon-bedrock", "novita"]


def test_call_openrouter_includes_provider_preferences(monkeypatch):
    monkeypatch.setattr(llm_router, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_PROVIDER_ORDER", "amazon-bedrock,novita")
    monkeypatch.setenv("OPENROUTER_PROVIDER_ALLOW_FALLBACKS", "false")

    captured: dict = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                "provider": "Amazon Bedrock",
            }

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(llm_router.requests, "post", fake_post)

    content, usage, response = llm_router.call_openrouter_chat_completions(
        [{"role": "user", "content": "hi"}],
        "openai/gpt-oss-20b",
    )
    assert content == "Hello"
    assert captured["json"]["provider"] == {
        "order": ["amazon-bedrock", "novita"],
        "allow_fallbacks": False,
        "only": ["amazon-bedrock", "novita"],
    }


def test_call_openrouter_allow_fallbacks_omits_only(monkeypatch):
    monkeypatch.setattr(llm_router, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_PROVIDER_ORDER", "amazon-bedrock,novita")
    monkeypatch.setenv("OPENROUTER_PROVIDER_ALLOW_FALLBACKS", "true")

    captured: dict = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

    def fake_post(url, headers, json, timeout):
        captured["json"] = json
        return FakeResponse()

    monkeypatch.setattr(llm_router.requests, "post", fake_post)
    llm_router.call_openrouter_chat_completions([{"role": "user", "content": "hi"}], "openai/gpt-oss-20b")
    assert captured["json"]["provider"]["allow_fallbacks"] is True
    assert "only" not in captured["json"]["provider"]


def test_call_openrouter_uses_catalog_provider_order_for_coreweave(monkeypatch):
    monkeypatch.setattr(llm_router, "OPENROUTER_API_KEY", "test-key")
    # Env still prefers Bedrock; catalog pin must win for CoreWeave models.
    monkeypatch.setenv("OPENROUTER_PROVIDER_ORDER", "amazon-bedrock,novita")
    monkeypatch.setenv("OPENROUTER_PROVIDER_ALLOW_FALLBACKS", "true")

    captured: dict = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                "provider": "CoreWeave",
            }

    def fake_post(url, headers, json, timeout):
        captured["json"] = json
        return FakeResponse()

    monkeypatch.setattr(llm_router.requests, "post", fake_post)
    llm_router.call_openrouter_chat_completions(
        [{"role": "user", "content": "hi"}],
        "nvidia/nemotron-3.5-lightning",
        model_id="openrouter-nemotron-3.5-lightning-coreweave",
    )
    assert captured["json"]["model"] == "nvidia/nemotron-3.5-lightning"
    assert captured["json"]["provider"] == {
        "order": ["coreweave"],
        "allow_fallbacks": False,
        "only": ["coreweave"],
    }
    assert captured["json"].get("reasoning_effort") == "low"


def test_call_openrouter_pins_deepseek_coreweave(monkeypatch):
    monkeypatch.setattr(llm_router, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_PROVIDER_ORDER", "amazon-bedrock,novita")

    captured: dict = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

    def fake_post(url, headers, json, timeout):
        captured["json"] = json
        return FakeResponse()

    monkeypatch.setattr(llm_router.requests, "post", fake_post)
    llm_router.call_openrouter_chat_completions(
        [{"role": "user", "content": "hi"}],
        "deepseek/deepseek-v4-flash-0731",
        model_id="openrouter-deepseek-v4-flash-0731-coreweave",
    )
    assert captured["json"]["provider"]["order"] == ["coreweave"]
    assert captured["json"]["provider"]["only"] == ["coreweave"]
    assert captured["json"]["provider"]["allow_fallbacks"] is False
    assert captured["json"].get("reasoning_effort") == "low"


def test_call_openrouter_pins_nemotron_deepinfra(monkeypatch):
    monkeypatch.setattr(llm_router, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_PROVIDER_ORDER", "amazon-bedrock,novita")

    captured: dict = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

    def fake_post(url, headers, json, timeout):
        captured["json"] = json
        return FakeResponse()

    monkeypatch.setattr(llm_router.requests, "post", fake_post)
    llm_router.call_openrouter_chat_completions(
        [{"role": "user", "content": "hi"}],
        "nvidia/nemotron-3.5-lightning",
        model_id="openrouter-nemotron-3.5-lightning-deepinfra",
    )
    assert captured["json"]["provider"] == {
        "order": ["deepinfra"],
        "allow_fallbacks": False,
        "only": ["deepinfra"],
    }


def test_generate_tries_online_fallback_before_local(monkeypatch):
    calls = {"openrouter": [], "local": 0}
    usage_records = []

    monkeypatch.setattr(llm_router, "CURRENT_LLM_MODEL_ID", "openrouter-nemotron-3.5-lightning-coreweave")
    monkeypatch.setattr(llm_router, "LLM_PROVIDER_MODE", "openrouter_with_local_fallback")
    monkeypatch.setattr(
        llm_router,
        "CHAT_FALLBACK_MODEL_IDS",
        ["openrouter-nemotron-3.5-lightning-deepinfra", "local"],
    )
    monkeypatch.setattr(llm_router, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(llm_router, "ONLINE_DISABLED_UNTIL", None)
    monkeypatch.setattr(llm_router, "openrouter_credit_threshold_reason", lambda: None)
    monkeypatch.setattr(llm_router, "_would_exceed_online_limits", lambda *a, **k: None)
    monkeypatch.setattr(llm_router, "record_llm_usage", lambda **kwargs: usage_records.append(kwargs))

    class FailPrimary:
        status_code = 503
        text = "provider down"

        def json(self):
            return {}

    class OkFallback:
        status_code = 200
        text = ""

        def json(self):
            return {
                "choices": [{"message": {"content": "fallback-ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                "provider": "DeepInfra",
            }

    def fake_openrouter(messages, model_name, **kwargs):
        mid = kwargs.get("model_id")
        calls["openrouter"].append(mid)
        if mid == "openrouter-nemotron-3.5-lightning-coreweave":
            return "", {}, FailPrimary()
        return "fallback-ok", {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}, OkFallback()

    def fake_local(messages, **kwargs):
        calls["local"] += 1
        return "local-ok", {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}

    monkeypatch.setattr(llm_router, "call_openrouter_chat_completions", fake_openrouter)
    monkeypatch.setattr(llm_router, "call_local_chat_completions", fake_local)

    result = llm_router.generate_llm_response([{"role": "user", "content": "hi"}])
    assert result == "fallback-ok"
    assert calls["openrouter"] == [
        "openrouter-nemotron-3.5-lightning-coreweave",
        "openrouter-nemotron-3.5-lightning-deepinfra",
    ]
    assert calls["local"] == 0
    assert llm_router.LAST_ACTIVE_MODEL_ID["chat"] == "openrouter-nemotron-3.5-lightning-deepinfra"
    # Per-model 5xx must not circuit-break OpenRouter for later turns.
    assert llm_router.ONLINE_DISABLED_UNTIL is None
    success = [r for r in usage_records if r.get("success")]
    assert len(success) == 1
    assert success[0]["model_id"] == "openrouter-nemotron-3.5-lightning-deepinfra"
    assert success[0].get("attribute_cost_to_selected") is False


def test_generate_after_primary_5xx_still_tries_openrouter_next_turn(monkeypatch):
    """Regression: a prior primary 5xx must not force later turns to local-only."""
    calls = {"openrouter": []}

    monkeypatch.setattr(llm_router, "CURRENT_LLM_MODEL_ID", "openrouter-nemotron-3.5-lightning-coreweave")
    monkeypatch.setattr(llm_router, "LLM_PROVIDER_MODE", "openrouter_with_local_fallback")
    monkeypatch.setattr(
        llm_router,
        "CHAT_FALLBACK_MODEL_IDS",
        ["openrouter-nemotron-3.5-lightning-deepinfra", "local"],
    )
    monkeypatch.setattr(llm_router, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(llm_router, "ONLINE_DISABLED_UNTIL", None)
    monkeypatch.setattr(llm_router, "openrouter_credit_threshold_reason", lambda: None)
    monkeypatch.setattr(llm_router, "_would_exceed_online_limits", lambda *a, **k: None)
    monkeypatch.setattr(llm_router, "record_llm_usage", lambda **kwargs: None)

    class FailPrimary:
        status_code = 503
        text = "provider down"

        def json(self):
            return {}

    class OkFallback:
        status_code = 200
        text = ""

        def json(self):
            return {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

    def fake_openrouter(messages, model_name, **kwargs):
        mid = kwargs.get("model_id")
        calls["openrouter"].append(mid)
        if mid == "openrouter-nemotron-3.5-lightning-coreweave":
            return "", {}, FailPrimary()
        return "ok", {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}, OkFallback()

    monkeypatch.setattr(llm_router, "call_openrouter_chat_completions", fake_openrouter)
    monkeypatch.setattr(
        llm_router,
        "call_local_chat_completions",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("local should not run")),
    )

    assert llm_router.generate_llm_response([{"role": "user", "content": "one"}]) == "ok"
    assert llm_router.ONLINE_DISABLED_UNTIL is None
    assert llm_router.generate_llm_response([{"role": "user", "content": "two"}]) == "ok"
    # Both turns attempt primary then DeepInfra — not skipped via online_unavailable.
    assert calls["openrouter"] == [
        "openrouter-nemotron-3.5-lightning-coreweave",
        "openrouter-nemotron-3.5-lightning-deepinfra",
        "openrouter-nemotron-3.5-lightning-coreweave",
        "openrouter-nemotron-3.5-lightning-deepinfra",
    ]


def test_openrouter_success_does_not_clear_402_cooldown(monkeypatch):
    """In-flight work that already passed the disabled gate must not wipe Retry-After."""
    from datetime import timedelta

    disabled_until = llm_router._utc_now() + timedelta(minutes=5)
    monkeypatch.setattr(llm_router, "CURRENT_LLM_MODEL_ID", "openrouter-gpt-oss-20b")
    monkeypatch.setattr(llm_router, "LLM_PROVIDER_MODE", "openrouter_with_local_fallback")
    monkeypatch.setattr(llm_router, "CHAT_FALLBACK_MODEL_IDS", ["local"])
    monkeypatch.setattr(llm_router, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(llm_router, "ONLINE_DISABLED_UNTIL", None)
    monkeypatch.setattr(llm_router, "openrouter_credit_threshold_reason", lambda: None)
    monkeypatch.setattr(llm_router, "_would_exceed_online_limits", lambda *a, **k: None)
    monkeypatch.setattr(llm_router, "record_llm_usage", lambda **kwargs: None)

    class OkResponse:
        status_code = 200
        text = ""

        def json(self):
            return {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

    def fake_openrouter(messages, model_name, **kwargs):
        # Another request hit 402 while this in-flight call was already past the gate.
        llm_router.ONLINE_DISABLED_UNTIL = disabled_until
        return (
            "ok",
            {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            OkResponse(),
        )

    monkeypatch.setattr(llm_router, "call_openrouter_chat_completions", fake_openrouter)

    assert llm_router.generate_llm_response([{"role": "user", "content": "hi"}]) == "ok"
    assert llm_router.ONLINE_DISABLED_UNTIL == disabled_until

    # Online fallback success must preserve the cooldown as well.
    monkeypatch.setattr(llm_router, "ONLINE_DISABLED_UNTIL", disabled_until)
    monkeypatch.setattr(
        llm_router,
        "CHAT_FALLBACK_MODEL_IDS",
        ["openrouter-gpt-oss-20b", "local"],
    )
    content = llm_router._fallback_after_failure(
        [{"role": "user", "content": "hi"}],
        reason="openrouter_http_error",
        selected_model_id="openrouter-nemotron-3.5-lightning-coreweave",
        temperature=None,
        max_tokens=180,
        purpose="chat",
        respect_online_disabled=False,
    )
    assert content == "ok"
    assert llm_router.ONLINE_DISABLED_UNTIL == disabled_until

def test_llm_status_exposes_openrouter_provider_order(api, monkeypatch):
    client, _server, headers = api
    monkeypatch.setenv("OPENROUTER_PROVIDER_ORDER", "amazon-bedrock,novita")
    monkeypatch.setenv("OPENROUTER_PROVIDER_ALLOW_FALLBACKS", "false")

    response = client.get("/llm/status", headers=headers)
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["openrouter"]["provider_order"] == ["amazon-bedrock", "novita"]
    assert payload["openrouter"]["provider_allow_fallbacks"] is False


def test_secrets_file_loads_provider_order(monkeypatch, tmp_path):
    secrets = tmp_path / "llm_secrets.env"
    secrets.write_text(
        "OPENROUTER_API_KEY=from-file\nOPENROUTER_PROVIDER_ORDER=amazon-bedrock,novita\n"
        "OPENROUTER_PROVIDER_ALLOW_FALLBACKS=false\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(llm_router, "LLM_SECRETS_FILE", secrets)
    monkeypatch.setattr(llm_router, "OPENROUTER_API_KEY", "")
    monkeypatch.setattr(llm_router, "_OPENROUTER_PROVIDER_ORDER_ENV_SET", False)
    monkeypatch.setattr(llm_router, "OPENROUTER_PROVIDER_ORDER_RAW", "")
    monkeypatch.delenv("OPENROUTER_PROVIDER_ORDER", raising=False)
    monkeypatch.delenv("OPENROUTER_PROVIDER_ALLOW_FALLBACKS", raising=False)

    llm_router._load_secrets_file()
    assert llm_router.OPENROUTER_API_KEY == "from-file"
    assert llm_router.OPENROUTER_PROVIDER_ORDER_RAW == "amazon-bedrock,novita"
    assert llm_router.OPENROUTER_PROVIDER_ALLOW_FALLBACKS is False

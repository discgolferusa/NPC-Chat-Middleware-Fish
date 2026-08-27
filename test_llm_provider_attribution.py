"""Per-request attribution of the upstream host that served an LLM call.

OpenRouter picks the host at call time, so latency/cost breakdowns must record
the routed provider rather than the catalog's static label whenever available.
"""

from __future__ import annotations

import json

import pytest

import llm_router


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self) -> dict:
        return self._payload


@pytest.fixture(autouse=True)
def _reset_routed_provider():
    llm_router._clear_routed_provider()
    yield
    llm_router._clear_routed_provider()


def test_attribution_prefers_routed_provider_over_catalog_label():
    llm_router._note_routed_provider("Novita")
    assert (
        llm_router._attribution_provider("openrouter", "openrouter-gpt-oss-20b") == "Novita"
    )


def test_attribution_falls_back_to_catalog_label_when_response_omits_provider():
    assert (
        llm_router._attribution_provider("openrouter", "openrouter-gpt-oss-20b")
        == "Amazon Bedrock / Novita"
    )
    assert (
        llm_router._attribution_provider(
            "openrouter", "openrouter-nemotron-3.5-lightning-coreweave"
        )
        == "CoreWeave"
    )


def test_attribution_for_local_ignores_stale_routed_provider():
    llm_router._note_routed_provider("CoreWeave")
    assert llm_router._attribution_provider("local", "local") == "local"


def test_openrouter_call_records_routed_provider(monkeypatch):
    payload = {
        "provider": "Amazon Bedrock",
        "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
    }
    monkeypatch.setattr(llm_router, "_openrouter_auth_headers", lambda: {})
    monkeypatch.setattr(
        llm_router.requests, "post", lambda *_a, **_kw: _FakeResponse(payload)
    )

    content, usage, _response = llm_router.call_openrouter_chat_completions(
        [{"role": "user", "content": "hi"}],
        "openai/gpt-oss-20b",
        model_id="openrouter-gpt-oss-20b",
    )

    assert content == "hello"
    assert usage["total_tokens"] == 13
    assert llm_router._routed_provider() == "Amazon Bedrock"
    assert (
        llm_router._attribution_provider("openrouter", "openrouter-gpt-oss-20b")
        == "Amazon Bedrock"
    )


def test_failed_openrouter_call_does_not_keep_previous_routed_provider(monkeypatch):
    llm_router._note_routed_provider("CoreWeave")
    monkeypatch.setattr(llm_router, "_openrouter_auth_headers", lambda: {})
    monkeypatch.setattr(
        llm_router.requests,
        "post",
        lambda *_a, **_kw: _FakeResponse({"error": "boom"}, status_code=503),
    )

    content, _usage, response = llm_router.call_openrouter_chat_completions(
        [{"role": "user", "content": "hi"}],
        "nvidia/nemotron-3.5-lightning",
        model_id="openrouter-nemotron-3.5-lightning-deepinfra",
    )

    assert content == ""
    assert response.status_code == 503
    assert llm_router._routed_provider() is None
    # Attribution degrades to the catalog label for the model that was tried.
    assert (
        llm_router._attribution_provider(
            "openrouter", "openrouter-nemotron-3.5-lightning-deepinfra"
        )
        == "DeepInfra"
    )


def test_set_active_binds_provider_to_request_timing(monkeypatch):
    pytest.importorskip("flask")
    from flask import Flask

    import request_timing

    llm_router._note_routed_provider("CoreWeave")
    app = Flask(__name__)
    with app.test_request_context("/npc-turn", method="POST"):
        request_timing.begin_request_timing()
        llm_router._set_active(
            "openrouter", "openrouter-nemotron-3.5-lightning-coreweave", "chat"
        )
        from flask import g

        assert g._timing_llm_model_id == "openrouter-nemotron-3.5-lightning-coreweave"
        assert g._timing_llm_provider == "CoreWeave"

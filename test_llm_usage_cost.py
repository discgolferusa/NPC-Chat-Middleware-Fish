"""Alpha LLM cost report attribution tests."""

from __future__ import annotations

import llm_router


def test_fallback_usage_attributed_to_selected_online_model(tmp_path, monkeypatch):
    usage_file = tmp_path / "llm_usage.json"
    monkeypatch.setattr(llm_router, "LLM_USAGE_FILE", usage_file)
    monkeypatch.setattr(llm_router, "_request_owner_user_id", lambda: "user-alpha")

    llm_router.record_llm_usage(
        provider="local",
        model_id="local",
        model="npc",
        prompt_tokens=8000,
        completion_tokens=400,
        total_tokens=8400,
        success=True,
        fallback=True,
        purpose="content_creation",
        selected_model_id="groq-gpt-oss-20b",
        attribute_cost_to_selected=True,
    )

    report = llm_router.get_llm_usage_cost_report(hours=24)
    assert report["totals"]["requests"] == 1
    assert report["totals"]["prompt_tokens"] == 8000
    assert report["totals"]["completion_tokens"] == 400
    assert report["by_model"][0]["model_id"] == "groq-gpt-oss-20b"
    assert report["by_model"][0]["fallback_attributed_requests"] == 1
    # 8000/1e6 * 0.075 + 400/1e6 * 0.30
    assert report["by_model"][0]["estimated_total_cost_usd"] == 0.00072
    assert report["by_user"][0]["owner_user_id"] == "user-alpha"
    assert report["by_user"][0]["estimated_total_cost_usd"] == 0.00072


def test_local_only_usage_has_zero_cost(tmp_path, monkeypatch):
    usage_file = tmp_path / "llm_usage.json"
    monkeypatch.setattr(llm_router, "LLM_USAGE_FILE", usage_file)
    monkeypatch.setattr(llm_router, "_request_owner_user_id", lambda: "user-local")

    llm_router.record_llm_usage(
        provider="local",
        model_id="local",
        model="npc",
        prompt_tokens=1000,
        completion_tokens=100,
        total_tokens=1100,
        success=True,
        fallback=False,
        purpose="chat",
        selected_model_id="local",
    )

    report = llm_router.get_llm_usage_cost_report(hours=24)
    assert report["by_model"][0]["model_id"] == "local"
    assert report["by_model"][0]["estimated_total_cost_usd"] == 0.0


def test_failed_rows_excluded_from_cost_report(tmp_path, monkeypatch):
    usage_file = tmp_path / "llm_usage.json"
    monkeypatch.setattr(llm_router, "LLM_USAGE_FILE", usage_file)
    monkeypatch.setattr(llm_router, "_request_owner_user_id", lambda: "user-x")

    llm_router.record_llm_usage(
        provider="groq",
        model_id="groq-gpt-oss-120b",
        model="openai/gpt-oss-120b",
        prompt_tokens=5000,
        completion_tokens=0,
        total_tokens=5000,
        success=False,
        fallback=True,
        purpose="chat",
        selected_model_id="groq-gpt-oss-120b",
    )

    report = llm_router.get_llm_usage_cost_report(hours=24)
    assert report["totals"]["requests"] == 0
    assert report["by_model"] == []

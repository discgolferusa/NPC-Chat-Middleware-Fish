"""Tests for local llama.cpp context budgeting in llm_router."""

from __future__ import annotations

import llm_router


def test_local_content_creation_max_input_chars_respects_context(monkeypatch):
    monkeypatch.setattr(llm_router, "LOCAL_LLM_CONTEXT_TOKENS", 8192)
    monkeypatch.setattr(llm_router, "CONTENT_CREATION_MODEL_ID", "local")
    monkeypatch.setattr(llm_router, "CONTENT_CREATION_PROVIDER_MODE", "local")
    budget = llm_router.content_creation_max_input_chars()
    assert 0 < budget < 120_000


def test_zero_budget_error_message_is_local_specific(monkeypatch):
    monkeypatch.setattr(llm_router, "LOCAL_LLM_CONTEXT_TOKENS", 1024)
    monkeypatch.setattr(llm_router, "CONTENT_CREATION_MODEL_ID", "local")
    monkeypatch.setattr(llm_router, "CONTENT_CREATION_PROVIDER_MODE", "local")
    assert llm_router.content_creation_max_input_chars() == 0
    message = llm_router.content_creation_zero_budget_error_message()
    assert "LOCAL_LLM_CONTEXT_TOKENS" in message
    assert "Groq tokens-per-minute" not in message


def test_zero_budget_error_message_is_groq_specific(monkeypatch):
    monkeypatch.setattr(llm_router, "CONTENT_CREATION_MODEL_ID", "groq-llama-3.3-70b")
    monkeypatch.setattr(llm_router, "CONTENT_CREATION_PROVIDER_MODE", "groq")
    message = llm_router.content_creation_zero_budget_error_message()
    assert "Groq tokens-per-minute" in message
    assert "LOCAL_LLM_CONTEXT_TOKENS" not in message


def test_fit_messages_to_local_context_shrinks_oversized_prompt(monkeypatch):
    monkeypatch.setattr(llm_router, "LOCAL_LLM_CONTEXT_TOKENS", 8192)
    huge = "word " * 50_000
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": huge},
    ]
    fitted, max_tokens = llm_router._fit_messages_to_local_context(messages, 6144)
    prompt_tokens = llm_router._estimate_prompt_tokens(fitted)
    assert prompt_tokens + max_tokens <= llm_router.LOCAL_LLM_CONTEXT_TOKENS
    assert len(fitted[1]["content"]) < len(huge)
    # After shrink, reclaim remaining context for completion (not stuck at 256).
    assert max_tokens > 1000


def test_fit_messages_preserves_small_chat_max_tokens(monkeypatch):
    monkeypatch.setattr(llm_router, "LOCAL_LLM_CONTEXT_TOKENS", 8192)
    messages = [
        {"role": "system", "content": "You are an NPC."},
        {"role": "user", "content": "Hello there."},
    ]
    _fitted, max_tokens = llm_router._fit_messages_to_local_context(messages, 180)
    assert max_tokens == 180

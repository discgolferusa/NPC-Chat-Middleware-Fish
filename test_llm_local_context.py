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
    monkeypatch.setattr(llm_router, "CONTENT_CREATION_MODEL_ID", "groq-gpt-oss-120b")
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


def test_fit_messages_keeps_tail_not_head(monkeypatch):
    """Import places NPC sections / forward excerpts at the end — do not end-chop."""
    monkeypatch.setattr(llm_router, "LOCAL_LLM_CONTEXT_TOKENS", 8192)
    marker = "NON-PLAYER CHARACTERS\nFULL NAME: Albert Johns\nJOHNS, OFFICER"
    # Far larger than an 8k context so the fit loop must shrink.
    head = "INTRO FILLER " * 20_000
    content = f"{head}\n\n{marker}"
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": content},
    ]
    fitted, _max_tokens = llm_router._fit_messages_to_local_context(messages, 6144)
    assert "FULL NAME: Albert Johns" in fitted[1]["content"]
    assert "truncated to fit local model context" in fitted[1]["content"]
    # Head filler should be what was dropped.
    assert fitted[1]["content"].index("Albert Johns") > fitted[1]["content"].index("truncated")


def test_fit_messages_reserve_matches_import_share_not_full_6144(monkeypatch):
    monkeypatch.setattr(llm_router, "LOCAL_LLM_CONTEXT_TOKENS", 8192)
    share = llm_router._local_import_completion_reserve_tokens()
    assert share < 6144
    # Prompt sized just over (ctx - full 6144) but under (ctx - share) should not shrink
    # if we only reserve the import share.
    slack = 128
    target_prompt_tokens = llm_router.LOCAL_LLM_CONTEXT_TOKENS - share - slack - 50
    # chars ≈ tokens * 4
    body = "x" * (target_prompt_tokens * 4)
    messages = [{"role": "user", "content": body}]
    fitted, max_tokens = llm_router._fit_messages_to_local_context(messages, 6144)
    # Should not need aggressive shrink; completion reclaim uses remaining room.
    assert "truncated to fit local model context" not in fitted[0]["content"]
    assert max_tokens >= share


def test_fit_messages_preserves_small_chat_max_tokens(monkeypatch):
    monkeypatch.setattr(llm_router, "LOCAL_LLM_CONTEXT_TOKENS", 8192)
    messages = [
        {"role": "system", "content": "You are an NPC."},
        {"role": "user", "content": "Hello there."},
    ]
    _fitted, max_tokens = llm_router._fit_messages_to_local_context(messages, 180)
    assert max_tokens == 180

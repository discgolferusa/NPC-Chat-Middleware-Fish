"""Spoken-reply completion budget: admin-tunable, with safe floors.

Replies were being cut off mid-sentence because the chat budget was a hard-coded
180 tokens, and the stage-direction pass then appended a period, so a truncation
reached the table looking like a finished sentence with nothing in the log.
"""

import llm_router
import prompt_caps


def _caps(**overrides):
    data = dict(prompt_caps.DEFAULT_PROMPT_CAPS)
    data.update(overrides)
    return data


def test_chat_replies_use_the_admin_cap(monkeypatch):
    monkeypatch.setattr(llm_router, "get_prompt_caps", lambda: _caps(chat_reply_max_tokens=900))
    assert llm_router._effective_max_tokens("openrouter-llama-3.3-70b", None) == 900


def test_reasoning_models_get_their_own_cap(monkeypatch):
    # Chain-of-thought is spent from the same budget before any visible text.
    monkeypatch.setattr(
        llm_router,
        "get_prompt_caps",
        lambda: _caps(chat_reply_max_tokens=512, chat_reply_reasoning_max_tokens=2048),
    )
    assert llm_router._effective_max_tokens("openrouter-gpt-oss-20b", None) == 2048


def test_zero_or_missing_cap_falls_back_instead_of_muting_the_npc(monkeypatch):
    # 0 is the caps form's "unset"; honouring it literally would truncate every
    # reply to nothing.
    monkeypatch.setattr(llm_router, "get_prompt_caps", lambda: _caps(chat_reply_max_tokens=0))
    assert llm_router._effective_max_tokens("local", None) == llm_router.DEFAULT_MAX_TOKENS

    monkeypatch.setattr(llm_router, "get_prompt_caps", dict)
    assert llm_router._effective_max_tokens("local", None) == llm_router.DEFAULT_MAX_TOKENS


def test_an_unreadable_caps_file_does_not_break_generation(monkeypatch):
    def boom():
        raise OSError("ServerSettings is not mounted")

    monkeypatch.setattr(llm_router, "get_prompt_caps", boom)
    assert llm_router._effective_max_tokens("local", None) == llm_router.DEFAULT_MAX_TOKENS


def test_an_explicit_request_still_wins(monkeypatch):
    monkeypatch.setattr(llm_router, "get_prompt_caps", lambda: _caps(chat_reply_max_tokens=900))
    assert llm_router._effective_max_tokens("local", 64) == 64


def test_content_creation_is_unaffected_by_the_reply_cap(monkeypatch):
    monkeypatch.setattr(llm_router, "get_prompt_caps", lambda: _caps(chat_reply_max_tokens=64))
    assert (
        llm_router._effective_max_tokens("local", None, purpose=llm_router.PURPOSE_CONTENT_CREATION)
        == llm_router.CONTENT_CREATION_DEFAULT_MAX_TOKENS
    )


def test_a_truncated_reply_says_so_in_the_log(capsys):
    llm_router._warn_if_length_capped("openrouter", "some/model", "length", 512, 512)
    out = capsys.readouterr().out
    assert "hit the token cap" in out
    assert "max_tokens=512" in out
    assert "Prompt size caps" in out


def test_a_normal_reply_stays_quiet(capsys):
    llm_router._warn_if_length_capped("local", "npc", "stop", 120, 512)
    assert capsys.readouterr().out == ""


def test_the_caps_file_round_trips_the_reply_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(prompt_caps, "SERVER_SETTINGS_DIR", tmp_path)
    monkeypatch.setattr(prompt_caps, "PROMPT_CAPS_FILE", tmp_path / "prompt_caps.yaml")
    monkeypatch.setattr(prompt_caps, "_CACHE", {"modified_at": None, "data": {}})

    saved = prompt_caps.save_prompt_caps({"chat_reply_max_tokens": 777})
    assert saved["chat_reply_max_tokens"] == 777
    # Keys the admin form did not send keep their defaults rather than becoming 0.
    assert saved["chat_reply_reasoning_max_tokens"] == (
        prompt_caps.DEFAULT_PROMPT_CAPS["chat_reply_reasoning_max_tokens"]
    )
    assert prompt_caps.get_prompt_caps()["chat_reply_max_tokens"] == 777

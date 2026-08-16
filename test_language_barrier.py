"""Unit tests for language-barrier detection and forced replies."""

from __future__ import annotations

from language_barrier import (
    build_language_barrier_response,
    detect_language_barrier,
    enforce_language_barrier_text,
    generate_npc_reply_text,
    language_barrier_status,
)


SLAADI_RULES = [
    "This NPC only understands Language Slaadi. They do not understand Common/English.",
    "Until DM channel context confirms a speaker knows Slaadi, do NOT answer the meaning of their words.",
    "Never invent or speak fake Slaadi, gibberish syllables, or alien words.",
    "Every such reply MUST end with exactly this final line: [DM CHECK: does this speaker know Language Slaadi]",
]


def test_detect_language_barrier_from_only_understands_rule():
    npc = {"name": "Xill", "llm_rules": SLAADI_RULES}
    barrier = detect_language_barrier(npc)
    assert barrier is not None
    assert barrier["language"].lower() == "slaadi"
    assert "slaadi" in barrier["claim"].lower()


def test_detect_language_barrier_absent_without_rules():
    assert detect_language_barrier({"name": "Varra", "llm_rules": ["Stay stern."]}) is None


def test_language_barrier_status_unknown_without_ruling():
    npc = {"name": "Xill", "llm_rules": SLAADI_RULES}
    resolved = language_barrier_status(npc, {"rulings": []})
    assert resolved is not None
    assert resolved["status"] == "unknown"


def test_language_barrier_status_confirmed_from_dm_ruling():
    npc = {"name": "Xill", "llm_rules": SLAADI_RULES}
    dm_context = {
        "rulings": [
            {
                "verdict": "confirmed",
                "claim_summary": "does this speaker know Language Slaadi",
                "ruling_text": "Yes, the wizard speaks Slaadi.",
            }
        ]
    }
    resolved = language_barrier_status(npc, dm_context)
    assert resolved is not None
    assert resolved["status"] == "confirmed"


def test_language_barrier_status_denied_from_dm_ruling():
    npc = {"name": "Xill", "llm_rules": SLAADI_RULES}
    dm_context = {
        "rulings": [
            {
                "verdict": "denied",
                "claim_summary": "does this speaker know Language Slaadi",
                "ruling_text": "No.",
            }
        ]
    }
    resolved = language_barrier_status(npc, dm_context)
    assert resolved is not None
    assert resolved["status"] == "denied"


def test_build_response_includes_dm_check_when_unknown():
    text = build_language_barrier_response(
        {"name": "Xill"},
        language="Slaadi",
        include_dm_check=True,
    )
    assert text.startswith("*Xill")
    assert "[DM CHECK: does this speaker know Language Slaadi]" in text
    # Stage only before the tag — no free spoken dialogue line.
    spoken_body = text.split("[DM CHECK:", 1)[0]
    assert spoken_body.count("*") == 2
    assert "hello" not in spoken_body.lower()


def test_generate_skips_llm_when_barrier_unknown():
    npc = {"name": "Xill", "llm_rules": SLAADI_RULES}
    calls = {"n": 0}

    def _ask(_messages):
        calls["n"] += 1
        return "I totally understand you in Common."

    text = generate_npc_reply_text(npc, {"rulings": []}, [], _ask)
    assert calls["n"] == 0
    assert "[DM CHECK:" in text
    assert "understand you" not in text.lower()


def test_generate_allows_llm_when_confirmed():
    npc = {"name": "Xill", "llm_rules": SLAADI_RULES}
    dm_context = {
        "rulings": [
            {
                "verdict": "confirmed",
                "claim_summary": "does this speaker know Language Slaadi",
                "ruling_text": "Yes",
            }
        ]
    }
    text = generate_npc_reply_text(
        npc,
        dm_context,
        [],
        lambda _m: "Welcome, speaker of Slaadi.",
    )
    assert text == "Welcome, speaker of Slaadi."


def test_enforce_strips_spoken_dialogue_and_keeps_stage():
    npc = {"name": "Xill", "llm_rules": SLAADI_RULES}
    raw = "*Xill tilts its head.* Blargh snikto! What do you want? [DM CHECK: ignored]"
    text = enforce_language_barrier_text(npc, {"rulings": []}, raw)
    assert text.startswith("*Xill tilts its head.*")
    assert "Blargh" not in text
    assert "What do you want" not in text
    assert "[DM CHECK: does this speaker know Language Slaadi]" in text

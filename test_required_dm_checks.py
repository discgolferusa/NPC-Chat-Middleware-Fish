"""Unit tests for generic DM CHECK templates embedded in llm_rules."""

from __future__ import annotations

from required_dm_checks import (
    ensure_required_dm_check_tags,
    extract_dm_check_templates_from_rules,
    format_required_dm_checks_prompt_block,
    generate_npc_reply_text,
    required_dm_checks,
)


LANGUAGE_RULES = [
    "Until the DM rules, do not treat the speaker as understood.",
    "End unsettled replies with "
    "[DM CHECK: Does the speaker know the necessary language to talk to the NPC]",
    "[DM Ruling: Success] Respond to the speaker in English.",
    "[DM Ruling: Failure] You do not understand the speaker; answer only by showing "
    "you cannot understand them, using a short fantasy-language line.",
]


def test_extract_dm_check_from_rules():
    claims = extract_dm_check_templates_from_rules({"llm_rules": LANGUAGE_RULES})
    assert len(claims) == 1
    assert "necessary language" in claims[0].lower()


def test_required_until_ruled():
    npc = {"llm_rules": LANGUAGE_RULES}
    assert required_dm_checks(npc, {"rulings": []})
    assert "REQUIRED DM CHECK" in format_required_dm_checks_prompt_block(npc, {"rulings": []})


def test_not_required_after_matching_ruling():
    npc = {"llm_rules": LANGUAGE_RULES}
    dm_context = {
        "rulings": [
            {
                "verdict": "confirmed",
                "claim_summary": "Does the speaker know the necessary language to talk to the NPC",
                "ruling_text": "Yes",
            }
        ]
    }
    assert required_dm_checks(npc, dm_context) == []
    assert format_required_dm_checks_prompt_block(npc, dm_context) == ""


def test_ensure_appends_missing_tag_without_rewriting_speech():
    npc = {"llm_rules": LANGUAGE_RULES}
    raw = "*Tilts its head.* Krah-vesh?"
    out = ensure_required_dm_check_tags(npc, {"rulings": []}, raw)
    assert out.startswith("*Tilts its head.* Krah-vesh?")
    assert out.strip().endswith(
        "[DM CHECK: Does the speaker know the necessary language to talk to the NPC]"
    )


def test_ensure_noop_when_tag_already_present():
    npc = {"llm_rules": LANGUAGE_RULES}
    raw = (
        "Krah?\n"
        "[DM CHECK: Does the speaker know the necessary language to talk to the NPC]"
    )
    assert ensure_required_dm_check_tags(npc, {"rulings": []}, raw) == raw


def test_no_templates_means_no_enforcement():
    npc = {"llm_rules": ["Stay stern.", "This NPC knows Language Elvish as well as Common."]}
    assert extract_dm_check_templates_from_rules(npc) == []
    assert required_dm_checks(npc, {"rulings": []}) == []
    assert ensure_required_dm_check_tags(npc, {"rulings": []}, "Hello.") == "Hello."


def test_generate_still_calls_llm():
    npc = {"llm_rules": LANGUAGE_RULES}
    calls = {"n": 0}

    def _ask(_messages):
        calls["n"] += 1
        return "Confused stare."

    text = generate_npc_reply_text(npc, {"rulings": []}, [], _ask)
    assert calls["n"] == 1
    assert "Confused stare." in text
    assert "[DM CHECK:" in text

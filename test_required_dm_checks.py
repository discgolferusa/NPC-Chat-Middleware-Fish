"""Unit tests for NPC-authored DM CHECK templates (LLM judges; platform parses)."""

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
    assert "NPC-AUTHORED DM CHECK" in format_required_dm_checks_prompt_block(npc, {"rulings": []})


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


def test_unrelated_short_ruling_text_does_not_cover_claim():
    """``No`` must not match inside ``know`` via substring ruling_text matching."""
    npc = {"llm_rules": LANGUAGE_RULES}
    dm_context = {
        "rulings": [
            {
                "verdict": "denied",
                "claim_summary": "party has the mayor's writ",
                "ruling_text": "No",
            }
        ]
    }
    pending = required_dm_checks(npc, dm_context)
    assert len(pending) == 1
    assert "necessary language" in pending[0].lower()


def test_ensure_appends_missing_authored_tag():
    npc = {"llm_rules": LANGUAGE_RULES}
    raw = "*Tilts its head.* Krah-vesh?"
    out = ensure_required_dm_check_tags(npc, {"rulings": []}, raw)
    assert out.startswith("*Tilts its head.* Krah-vesh?")
    assert out.strip().endswith(
        "[DM CHECK: Does the speaker know the necessary language to talk to the NPC]"
    )


def test_ensure_noop_when_exact_tag_already_present():
    npc = {"llm_rules": LANGUAGE_RULES}
    raw = (
        "Krah?\n"
        "[DM CHECK: Does the speaker know the necessary language to talk to the NPC]"
    )
    assert ensure_required_dm_check_tags(npc, {"rulings": []}, raw) == raw


def test_ensure_accepts_near_match_without_duplicating():
    npc = {"llm_rules": LANGUAGE_RULES}
    raw = (
        "Krah?\n"
        "[DM CHECK: does this speaker know the necessary language to talk to the NPC]"
    )
    assert ensure_required_dm_check_tags(npc, {"rulings": []}, raw) == raw


def test_no_templates_means_no_forced_tag():
    npc = {"llm_rules": ["Stay stern.", "This NPC knows Language Elvish as well as Common."]}
    assert extract_dm_check_templates_from_rules(npc) == []
    assert required_dm_checks(npc, {"rulings": []}) == []
    assert ensure_required_dm_check_tags(npc, {"rulings": []}, "Hello.") == "Hello."


def test_generate_does_not_invent_check_from_player_text():
    """Without an authored template, omitted tags stay omitted (LLM responsibility)."""
    npc = {"llm_rules": ["Be stern."]}
    calls = {"n": 0}

    def _ask(_messages):
        calls["n"] += 1
        return "I cannot simply take your word for that."

    text = generate_npc_reply_text(
        npc,
        {"rulings": []},
        [],
        _ask,
        player_text="We have a letter from the mayor allowing us to enter.",
    )
    assert calls["n"] == 1
    assert text == "I cannot simply take your word for that."
    assert "[DM CHECK:" not in text


def test_generate_still_enforces_authored_template():
    npc = {"llm_rules": LANGUAGE_RULES}
    text = generate_npc_reply_text(
        npc,
        {"rulings": []},
        [],
        lambda _m: "Confused stare.",
        player_text="Hello",
    )
    assert "Confused stare." in text
    assert "[DM CHECK:" in text
    assert "necessary language" in text.lower()

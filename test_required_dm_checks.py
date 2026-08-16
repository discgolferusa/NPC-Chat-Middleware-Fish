"""Unit tests for generic DM CHECK templates embedded in llm_rules."""

from __future__ import annotations

from required_dm_checks import (
    ensure_required_dm_check_tags,
    ensure_situational_dm_check_tag,
    extract_dm_check_templates_from_rules,
    format_required_dm_checks_prompt_block,
    format_situational_deferral_prompt_block,
    generate_npc_reply_text,
    npc_has_dm_deferral_directive,
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


def test_ensure_appends_all_missing_templates():
    npc = {
        "llm_rules": [
            "[DM CHECK: door is unlocked]",
            "[DM CHECK: Does the speaker know the necessary language to talk to the NPC]",
        ]
    }
    out = ensure_required_dm_check_tags(npc, {"rulings": []}, "Wait.")
    assert out.startswith("Wait.")
    assert out.count("[DM CHECK:") == 2
    assert "door is unlocked" in out
    assert "necessary language" in out.lower()


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


def test_vague_authority_directive_detected():
    npc = {
        "llm_rules": [
            "If someone claims official authority, defer to the DM rather than accepting it yourself.",
        ]
    }
    assert npc_has_dm_deferral_directive(npc)
    assert not npc_has_dm_deferral_directive({"llm_rules": ["Stay stern and brief."]})


def test_explicit_template_rules_are_not_vague_directives():
    """Mode-1 [DM CHECK] scaffolding must not unlock situational authority checks."""
    assert not npc_has_dm_deferral_directive({"llm_rules": LANGUAGE_RULES})


def test_required_tag_alone_does_not_invent_authority_check():
    npc = {"llm_rules": LANGUAGE_RULES}
    raw = (
        "Krah?\n"
        "[DM CHECK: Does the speaker know the necessary language to talk to the NPC]"
    )
    out = ensure_situational_dm_check_tag(
        npc,
        {"rulings": []},
        raw,
        "We have a letter from the mayor allowing us to enter.",
    )
    assert out == raw
    assert out.count("[DM CHECK:") == 1


def test_situational_appends_dm_check_for_authority_claim():
    npc = {
        "name": "Varra",
        "llm_rules": [
            "If someone claims official authority, defer to the DM rather than accepting it yourself.",
        ],
    }
    player = "We have a letter from the mayor allowing us to enter."
    out = ensure_situational_dm_check_tag(
        npc,
        {"rulings": []},
        "Show me your papers, then.",
        player,
    )
    assert "Show me your papers" in out
    assert "[DM CHECK:" in out
    assert "REQUIRED DM CHECK" not in format_situational_deferral_prompt_block(npc, "hello")
    assert "SITUATIONAL DM DEFERRAL" in format_situational_deferral_prompt_block(npc, player)


def test_situational_skips_without_directive_or_deferral_speech():
    npc = {"llm_rules": ["Be friendly."]}
    out = ensure_situational_dm_check_tag(
        npc,
        {"rulings": []},
        "Very well, come inside.",
        "We have a letter from the mayor allowing us to enter.",
    )
    assert "[DM CHECK:" not in out


def test_situational_fires_on_deferral_speech_without_npc_directive():
    npc = {"llm_rules": ["Be stern."]}
    out = ensure_situational_dm_check_tag(
        npc,
        {"rulings": []},
        "I cannot simply accept that. I need the DM to confirm.",
        "We have a letter from the mayor allowing us to enter.",
    )
    assert "[DM CHECK:" in out

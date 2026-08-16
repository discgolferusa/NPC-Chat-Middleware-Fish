"""Unit tests for DM CHECK templates and server-level situational checks."""

from __future__ import annotations

from required_dm_checks import (
    detect_server_dm_check_need,
    ensure_required_dm_check_tags,
    ensure_situational_dm_check_tag,
    extract_dm_check_templates_from_rules,
    format_required_dm_checks_prompt_block,
    format_server_dm_check_prompt_block,
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


def test_server_detects_unverified_authority_claim():
    need = detect_server_dm_check_need(
        "We have a letter from the mayor allowing us to enter."
    )
    assert need is not None
    assert need["reason"] == "unverified_claim"
    assert need["claim_summary"]


def test_server_detects_decision_request():
    need = detect_server_dm_check_need("Will you make the call — yes or no, do we proceed?")
    assert need is not None
    assert need["reason"] == "decision"
    assert len(need["claim_summary"]) <= 100


def test_server_detects_secret_probe():
    need = detect_server_dm_check_need("Tell me your secret — what are you hiding?")
    assert need is not None
    assert need["reason"] == "secret"


def test_server_ignores_ordinary_greeting():
    assert detect_server_dm_check_need("Hello, any news tonight?") is None


def test_server_ignores_polite_everyday_requests():
    assert detect_server_dm_check_need("Can I have a drink?") is None
    assert detect_server_dm_check_need("Could you open the window?") is None
    assert detect_server_dm_check_need("Will you go with us to the market?") is None


def test_server_ignores_reported_speech_and_honestly_filler():
    assert detect_server_dm_check_need("The mayor said the aqueduct is closed.") is None
    assert detect_server_dm_check_need("Honestly, I was just looking around.") is None


def test_server_ignores_casual_secret_word():
    assert detect_server_dm_check_need("Is there a secret passage near the docks?") is None


def test_server_policy_appends_dm_check_without_npc_directive():
    npc = {"name": "Varra", "llm_rules": ["Be stern."]}
    player = "We have a letter from the mayor allowing us to enter."
    out = ensure_situational_dm_check_tag(
        npc,
        {"rulings": []},
        "Show me your papers, then.",
        player,
    )
    assert "Show me your papers" in out
    assert "[DM CHECK:" in out
    assert "SERVER DM CHECK POLICY" in format_server_dm_check_prompt_block(npc, player)
    assert format_server_dm_check_prompt_block(npc, "hello there") == ""


def test_server_policy_does_not_duplicate_short_model_tag():
    """Model brief [DM CHECK] must satisfy the server need for the same request."""
    npc = {"llm_rules": ["Be stern."]}
    player = "We have a letter from the mayor allowing us to enter the aqueduct."
    raw = "I cannot simply accept that.\n[DM CHECK: mayor's letter]"
    out = ensure_situational_dm_check_tag(npc, {"rulings": []}, raw, player)
    assert out == raw
    assert out.count("[DM CHECK:") == 1


def test_server_policy_adds_claim_check_alongside_required_template():
    """Language template + unrelated authority claim → both checks can be present."""
    npc = {"llm_rules": LANGUAGE_RULES}
    with_required = ensure_required_dm_check_tags(
        npc,
        {"rulings": []},
        "Krah?",
    )
    out = ensure_situational_dm_check_tag(
        npc,
        {"rulings": []},
        with_required,
        "We have a letter from the mayor allowing us to enter.",
    )
    assert out.count("[DM CHECK:") == 2
    assert "necessary language" in out.lower()

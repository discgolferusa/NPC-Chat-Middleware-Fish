"""Unit tests for NPC-authored DM CHECK templates (LLM judges; platform parses)."""

from __future__ import annotations

from required_dm_checks import (
    ensure_required_dm_check_tags,
    enforce_unruled_check_spoken_behavior,
    extract_dm_check_templates_from_rules,
    format_required_dm_checks_prompt_block,
    generate_npc_reply_text,
    log_llm_prompt_to_console,
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

SLAAD_RULES = [
    "Until the DM rules, do not treat the speaker as understood.",
    "'End unsettled replies with [DM CHECK: Does the speaker know the necessary language to talk to the NPC]'",
    "'[DM Ruling: Success] Respond to the speaker in English.'",
    "'[DM Ruling: Failure] You do not understand them; reply only to show that, in a short fantasy-language line.'",
]


def test_extract_dm_check_from_rules():
    claims = extract_dm_check_templates_from_rules({"llm_rules": LANGUAGE_RULES})
    assert len(claims) == 1
    assert "necessary language" in claims[0].lower()


def test_extract_preserves_trailing_quotes_and_possessives():
    """Normal [DM CHECK:…] claims must keep authored ' / \" at the ends."""
    npc = {
        "llm_rules": [
            'End with [DM CHECK: confirm the mayor\'s "writ of passage"]',
        ]
    }
    claims = extract_dm_check_templates_from_rules(npc)
    assert len(claims) == 1
    assert claims[0] == 'confirm the mayor\'s "writ of passage"'
    out = ensure_required_dm_check_tags(npc, {"rulings": []}, "Hold.")
    assert out.endswith('[DM CHECK: confirm the mayor\'s "writ of passage"]')


def test_extract_finds_dm_even_when_yaml_split_mapping():
    """Look for [DM… even after YAML turns the tag into a one-key mapping."""
    import yaml

    raw = (
        "llm_rules:\n"
        "- Stay stern.\n"
        "- If someone claims official authority, defer to the DM. "
        "End with [DM CHECK: player claims official authority or permission "
        "that needs DM confirmation]\n"
    )
    data = yaml.safe_load(raw)
    assert isinstance(data["llm_rules"][1], dict)
    assert any("[DM" in str(k).upper() for k in data["llm_rules"][1].keys())
    claims = extract_dm_check_templates_from_rules(data)
    assert claims
    assert "official authority" in claims[0].lower() or "permission" in claims[0].lower()
    out = ensure_required_dm_check_tags(data, {"rulings": []}, "You may pass.")
    assert "[DM CHECK:" in out


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


def test_claim_match_ignores_stopword_boilerplate_overlap():
    from required_dm_checks import _claim_matches

    language = "Does the speaker know the necessary language to talk to the NPC"
    # Shares does/the/to/speaker/know boilerplate-ish tokens but is a different check.
    authority = "Does the speaker know whether the party has the mayor writ to enter"
    assert not _claim_matches(language, authority)

    near = "does this speaker know the necessary language to talk to the NPC"
    assert _claim_matches(language, near)


def test_claim_match_short_secrets_template_near_paraphrase():
    """Auto-seeded secrets check has few content tokens; paraphrases must still match."""
    from required_dm_checks import _claim_matches, _content_claim_tokens, required_dm_checks

    template = "the player is probing one of this NPC's secrets"
    assert len(_content_claim_tokens(template)) == 3

    paraphrase = "player is probing one of the NPC secrets"
    assert _claim_matches(template, paraphrase)

    npc = {
        "llm_rules": [
            "If asked about secrets, end with "
            "[DM CHECK: the player is probing one of this NPC's secrets]"
        ]
    }
    dm_context = {
        "rulings": [
            {
                "verdict": "denied",
                "claim_summary": "player is probing one of the NPC secrets",
                "ruling_text": "No",
            }
        ]
    }
    assert required_dm_checks(npc, dm_context) == []

    # Unrelated short claim must not clear the secrets template.
    bad_context = {
        "rulings": [
            {
                "verdict": "confirmed",
                "claim_summary": "player is probing the locked door",
                "ruling_text": "Yes",
            }
        ]
    }
    assert required_dm_checks(npc, bad_context)


def test_no_templates_means_no_forced_tag():
    npc = {"llm_rules": ["Stay stern.", "This NPC knows Language Elvish as well as Common."]}
    assert extract_dm_check_templates_from_rules(npc) == []
    assert required_dm_checks(npc, {"rulings": []}) == []
    assert ensure_required_dm_check_tags(npc, {"rulings": []}, "Hello.") == "Hello."


def test_generate_logs_llm_prompt_to_console(capsys):
    messages = [
        {"role": "system", "content": "You are Captain Varra."},
        {"role": "user", "content": "Here is the letter from the city council."},
    ]
    generate_npc_reply_text(
        {"llm_rules": ["Be stern."]},
        {"rulings": []},
        messages,
        lambda _m: "Stand aside.",
    )
    captured = capsys.readouterr().out
    assert "[LLM-PROMPT] npc-reply messages=2" in captured
    assert "You are Captain Varra." in captured
    assert "Here is the letter from the city council." in captured
    assert "[LLM-PROMPT] === end ===" in captured


def test_log_llm_prompt_uses_single_print(monkeypatch):
    """One print call so threaded Flask workers cannot interleave prompt lines."""
    prints: list[str] = []

    def _capture(text="", *args, **kwargs):
        prints.append(text if isinstance(text, str) else str(text))

    monkeypatch.setattr("builtins.print", _capture)
    log_llm_prompt_to_console(
        [
            {"role": "system", "content": "System A"},
            {"role": "user", "content": "User B"},
        ]
    )
    assert len(prints) == 1
    assert prints[0].startswith("[LLM-PROMPT] npc-reply messages=2")
    assert "System A" in prints[0]
    assert "User B" in prints[0]
    assert prints[0].rstrip().endswith("[LLM-PROMPT] === end ===")


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


def test_generate_can_skip_required_dm_check_enforcement():
    npc = {"llm_rules": LANGUAGE_RULES}
    text = generate_npc_reply_text(
        npc,
        {"rulings": []},
        [],
        lambda _m: "Very well.",
        enforce_required_dm_checks=False,
    )
    assert text == "Very well."
    assert "[DM CHECK:" not in text


def test_slaad_quoted_yaml_rules_extract_language_template():
    npc = {"llm_rules": SLAAD_RULES}
    claims = extract_dm_check_templates_from_rules(npc)
    assert len(claims) == 1
    assert "necessary language" in claims[0].lower()


def test_format_includes_language_barrier_block_for_slaad():
    npc = {"llm_rules": SLAAD_RULES}
    block = format_required_dm_checks_prompt_block(npc, {"rulings": []})
    assert "LANGUAGE BARRIER" in block
    assert "Do NOT respond in English" in block


def test_enforce_strips_english_comprehension_when_language_check_unruled():
    npc = {"llm_rules": SLAAD_RULES}
    raw = "*Snarls.* Yes, I understand you perfectly. How may I help?"
    tagged = ensure_required_dm_check_tags(npc, {"rulings": []}, raw)
    enforced = enforce_unruled_check_spoken_behavior(npc, {"rulings": []}, tagged)
    assert "understand you" not in enforced.lower()
    assert "Skraa-vek" in enforced
    assert "[DM CHECK:" in enforced
    assert "necessary language" in enforced.lower()


def test_generate_rewrites_english_for_unruled_language_barrier():
    npc = {"llm_rules": SLAAD_RULES}
    text = generate_npc_reply_text(
        npc,
        {"rulings": []},
        [],
        lambda _m: "Hello, I can help you with that.",
    )
    assert "help you" not in text.lower()
    assert "Skraa-vek" in text
    assert "[DM CHECK:" in text


def test_generate_allows_english_after_language_success_ruling():
    npc = {"llm_rules": SLAAD_RULES}
    dm_context = {
        "rulings": [
            {
                "verdict": "confirmed",
                "claim_summary": "Does the speaker know the necessary language to talk to the NPC",
                "ruling_text": "Yes, they speak the language.",
            }
        ]
    }
    text = generate_npc_reply_text(
        npc,
        dm_context,
        [],
        lambda _m: "Welcome, traveler. What do you need?",
    )
    assert "Welcome, traveler" in text
    assert "[DM CHECK:" not in text


def test_language_comprehension_claim_rejects_generic_understand():
    from required_dm_checks import _is_language_comprehension_claim

    assert not _is_language_comprehension_claim("does the player understand the secret")
    assert not _is_language_comprehension_claim("party understands the mayor's plan")
    assert _is_language_comprehension_claim(
        "Does the speaker know the necessary language to talk to the NPC"
    )
    assert _is_language_comprehension_claim(
        "does this speaker know Language Slaadi"
    )


def test_generic_understand_npc_is_not_rewritten_to_fantasy_language():
    """Failure + understand in rules must not trigger barrier rewrite without barrier setup."""
    from required_dm_checks import enforce_unruled_check_spoken_behavior

    npc = {
        "llm_rules": [
            "End unsettled replies with [DM CHECK: does the player understand the secret]",
            "[DM Ruling: Success] Reveal the secret.",
            "[DM Ruling: Failure] You do not understand what they mean.",
        ],
    }
    raw = "Yes, I understand completely. Here is the answer."
    tagged = ensure_required_dm_check_tags(npc, {"rulings": []}, raw)
    enforced = enforce_unruled_check_spoken_behavior(npc, {"rulings": []}, tagged)
    assert "Skraa-vek" not in enforced
    assert "understand completely" in enforced.lower()

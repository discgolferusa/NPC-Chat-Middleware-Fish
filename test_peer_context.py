"""Unit tests for peer NPC public-facts formatting with gender/pronouns."""

from __future__ import annotations

from peer_context import format_peer_public_facts_block
from npc_pronouns import get_npc_pronoun_context


def test_get_npc_pronoun_context_structured():
    ctx = get_npc_pronoun_context(
        {
            "gender": "female",
            "pronouns": {
                "subject": "she",
                "object": "her",
                "possessive_adjective": "her",
                "possessive_pronoun": "hers",
                "reflexive": "herself",
            },
        }
    )
    assert ctx["gender"] == "female"
    assert ctx["subject"] == "she"
    assert "she/her" in ctx["display"]


def test_get_npc_pronoun_context_string_shorthand():
    ctx = get_npc_pronoun_context({"gender": "male", "pronouns": "he/him"})
    assert ctx["subject"] == "he"
    assert ctx["object"] == "him"


def test_format_peer_public_facts_includes_identity(monkeypatch):
    peer = {
        "name": "Captain Varra",
        "gender": "female",
        "pronouns": {
            "subject": "she",
            "object": "her",
            "possessive_adjective": "her",
            "possessive_pronoun": "hers",
            "reflexive": "herself",
        },
        "public_facts": ["Commands the harbor watch."],
    }

    monkeypatch.setattr(
        "peer_context.lookup_document",
        lambda kind, peer_id: peer if kind == "npc" and peer_id == "captain_varra" else None,
    )

    block = format_peer_public_facts_block(
        ["captain_varra"],
        caps={"peer_public_facts_max_chars": 2000},
    )

    assert "Public knowledge about Captain Varra (captain_varra):" in block
    assert "Gender/identity: female" in block
    assert "Pronouns:" in block
    assert "subject=she" in block
    assert "Commands the harbor watch." in block
    assert "must NEVER replace" in block
    assert "YOUR gender/identity" in block

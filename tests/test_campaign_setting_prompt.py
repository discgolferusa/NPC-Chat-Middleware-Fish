"""Campaign genre / RPG system in party context and stay-in-setting LLM rule."""

from __future__ import annotations

import party_campaign as pc


def test_normalize_campaign_genre_aliases():
    assert pc.normalize_campaign_genre("Fantasy") == "fantasy"
    assert pc.normalize_campaign_genre("sci-fi") == "sci_fi"
    assert pc.normalize_campaign_genre("science fiction") == "sci_fi"
    assert pc.normalize_campaign_genre("") == ""
    assert pc.normalize_campaign_genre("cyberpunk") == "other"


def test_format_party_context_includes_genre_and_rpg_system():
    prompt = pc.format_party_context_for_prompt(
        {
            "campaign_id": "cronus",
            "party_id": "marines",
            "party_label": "Marines",
            "campaign": {
                "campaign_id": "cronus",
                "name": "Cronus Ops",
                "description": "Salvage gone wrong.",
                "genre": "sci_fi",
                "rpg_system": "Alien RPG",
            },
            "party_members": [],
            "speaking_member": {},
        }
    )
    assert "Campaign genre/setting: sci-fi" in prompt
    assert "RPG system: Alien RPG" in prompt
    assert "Campaign summary: Salvage gone wrong." in prompt


def test_format_campaign_setting_rule_requires_setting():
    rule = pc.format_campaign_setting_rule(
        {
            "campaign": {
                "genre": "horror",
                "rpg_system": "Call of Cthulhu",
            }
        }
    )
    assert "Campaign setting rule (REQUIRED" in rule
    assert "Genre/setting: horror" in rule
    assert "RPG system: Call of Cthulhu" in rule
    assert "Do not introduce anachronisms" in rule


def test_format_campaign_setting_rule_empty_when_unset():
    assert pc.format_campaign_setting_rule({"campaign": {}}) == ""
    assert pc.format_campaign_setting_rule(None) == ""

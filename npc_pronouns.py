"""Normalize NPC gender/pronoun fields for prompt grounding."""

from __future__ import annotations


def get_npc_pronoun_context(npc: dict) -> dict:
    """
    Return normalized NPC gender/pronoun context for prompt grounding.

    Existing NPC YAML files do not need these fields; old files fall back to
    neutral they/them phrasing until explicit values are added.
    """
    gender = (npc.get("gender") or npc.get("identity") or "unspecified")
    if isinstance(gender, str):
        gender = gender.strip() or "unspecified"
    else:
        gender = str(gender or "unspecified").strip() or "unspecified"

    pronouns = npc.get("pronouns", {}) or {}

    # Allow either a structured YAML block or a simple string like "he/him".
    if isinstance(pronouns, str):
        raw = pronouns.strip()
        lowered = raw.lower()

        if lowered in ("he/him", "he/him/his"):
            pronouns = {
                "subject": "he",
                "object": "him",
                "possessive_adjective": "his",
                "possessive_pronoun": "his",
                "reflexive": "himself",
            }
        elif lowered in ("she/her", "she/her/hers"):
            pronouns = {
                "subject": "she",
                "object": "her",
                "possessive_adjective": "her",
                "possessive_pronoun": "hers",
                "reflexive": "herself",
            }
        elif lowered in ("they/them", "they/them/theirs"):
            pronouns = {
                "subject": "they",
                "object": "them",
                "possessive_adjective": "their",
                "possessive_pronoun": "theirs",
                "reflexive": "themself",
            }
        else:
            pronouns = {"display": raw}

    if not isinstance(pronouns, dict):
        pronouns = {}

    subject = pronouns.get("subject", "they")
    obj = pronouns.get("object", "them")
    possessive_adjective = pronouns.get("possessive_adjective", "their")
    possessive_pronoun = pronouns.get("possessive_pronoun", "theirs")
    reflexive = pronouns.get("reflexive", "themself")

    display = pronouns.get("display")
    if not display:
        display = f"{subject}/{obj}/{possessive_adjective}/{possessive_pronoun}/{reflexive}"

    return {
        "gender": gender,
        "subject": subject,
        "object": obj,
        "possessive_adjective": possessive_adjective,
        "possessive_pronoun": possessive_pronoun,
        "reflexive": reflexive,
        "display": display,
    }

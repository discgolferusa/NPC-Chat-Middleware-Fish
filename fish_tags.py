"""Fish performance tags. Canonical dialogue never keeps them.

The allowlist is the S2 emotion page. ``drama-3`` tag support is unverified;
unknown brackets are removed from both the transcript and the spoken request.
"""

from __future__ import annotations

import re

# Interiors, compared case-insensitively. Multi-word tags keep their spaces.
ALLOWED_TAGS = frozenset(
    {
        "happy",
        "sad",
        "angry",
        "excited",
        "calm",
        "nervous",
        "confident",
        "surprised",
        "satisfied",
        "delighted",
        "scared",
        "worried",
        "upset",
        "frustrated",
        "depressed",
        "empathetic",
        "embarrassed",
        "disgusted",
        "moved",
        "proud",
        "relaxed",
        "grateful",
        "curious",
        "sarcastic",
        "disdainful",
        "unhappy",
        "anxious",
        "hysterical",
        "indifferent",
        "uncertain",
        "doubtful",
        "confused",
        "disappointed",
        "regretful",
        "guilty",
        "ashamed",
        "jealous",
        "envious",
        "hopeful",
        "optimistic",
        "pessimistic",
        "nostalgic",
        "lonely",
        "bored",
        "contemptuous",
        "sympathetic",
        "compassionate",
        "determined",
        "resigned",
        "in a hurry tone",
        "shouting",
        "screaming",
        "whispering",
        "soft tone",
        "emphasis",
        "laughing",
        "chuckling",
        "sobbing",
        "crying loudly",
        "sighing",
        "groaning",
        "panting",
        "gasping",
        "yawning",
        "snoring",
        "clear throat",
        "audience laughing",
        "background laughter",
        "crowd laughing",
        "break",
        "long-break",
    }
)

_BRACKET = re.compile(r"\[([^\[\]]{1,80})\]")


def _normalize_tag(inner: str) -> str:
    return " ".join((inner or "").strip().lower().split())


def is_allowed_tag(inner: str) -> bool:
    return _normalize_tag(inner) in ALLOWED_TAGS


def canonical_text(text: str) -> str:
    """Drop every square-bracket cue so transcripts and memory stay spoken words."""

    return strip_all_brackets(text)


def strip_all_brackets(text: str) -> str:
    """Remove bracket cues without joining separate lines into one sentence."""

    cleaned = _BRACKET.sub("", text or "")
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned)
    return cleaned.strip()


def tts_markup(text: str) -> str:
    """Keep allowlisted cues. Remove every other bracket, including its interior."""

    def _keep(match: re.Match[str]) -> str:
        inner = match.group(1)
        if is_allowed_tag(inner):
            return f"[{_normalize_tag(inner)}]"
        return ""

    cleaned = _BRACKET.sub(_keep, text or "")
    return re.sub(r"\s+", " ", cleaned).strip()

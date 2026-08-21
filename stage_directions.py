"""
Enforce the stage-direction placement contract from ServerSettings/llm_rules.yaml.

The global rules ask for at most one nonverbal stage direction, wrapped in single
asterisks, at the start of the reply. Models follow that most of the time but not
reliably, and the prompt cannot guarantee it. A beat that lands mid-reply reaches
the player as an interrupted line and splits TTS into an extra narrator clip in
the middle of the NPC's speech, so placement is normalized in code instead.
"""

from __future__ import annotations

import re

# Same shape the TTS segmenter matches, so normalization and segmentation agree
# on what counts as a stage direction.
STAGE_DIRECTION_PATTERN = re.compile(r"\*([^*]{1,800})\*")

_SENTENCE_ENDINGS = ".!?…"


def _end_sentence(body: str) -> str:
    body = body.strip()
    if not body:
        return ""
    if body.endswith("...") or body[-1] in _SENTENCE_ENDINGS:
        return body
    return body + "."


def _tidy_seam(text: str, *, opens_at_seam: bool) -> str:
    """
    Clean up the punctuation left behind where a stage direction was lifted out.

    ``opens_at_seam`` says the remainder starts where a direction used to be, so its
    leading punctuation is seam debris. Without that guard the model's own opening
    ellipsis or dash reads as debris too, and the hesitation it marks is dropped.

    A leading newline always survives: it is what keeps a trailing ``[DM CHECK: ...]``
    tag on its own line when the rest of the reply was beats.
    """
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,;:])(\s*[,;:])+", r"\1", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    if opens_at_seam:
        # A beat can leave debris on either side of the newline it opened, so strip
        # across that newline while keeping it: it is what holds a trailing
        # [DM CHECK: ...] tag on its own line.
        text = re.sub(r"^[ \t,;:.\-–—]*(\n?)[ \t,;:.\-–—]*", r"\1", text, count=1)
    leading_newline = "\n" if text[:1] == "\n" else ""
    return leading_newline + text.strip()


def normalize_stage_directions(text: str) -> str:
    """
    Collapse asterisk-wrapped stage directions into one direction at the start.

    Merged directions keep their original order, so a reply that opens and closes
    with a beat becomes a single opening beat rather than silently losing the
    second one. Replies with no direction, or with one direction already leading,
    are returned byte-for-byte so well-formed output is never reflowed.

    Newlines survive because a trailing ``[DM CHECK: ...]`` tag has to stay on its
    own final line.
    """
    source = text or ""
    matches = [m for m in STAGE_DIRECTION_PATTERN.finditer(source) if m.group(1).strip()]
    if not matches:
        return source

    if len(matches) == 1 and not source[: matches[0].start()].strip():
        return source

    direction = " ".join(_end_sentence(m.group(1)) for m in matches).strip()
    remainder = _tidy_seam(
        STAGE_DIRECTION_PATTERN.sub(" ", source),
        opens_at_seam=not source[: matches[0].start()].strip(),
    )

    if not remainder:
        return f"*{direction}*"
    if remainder.startswith("\n"):
        return f"*{direction}*{remainder}"
    return f"*{direction}* {remainder}"
